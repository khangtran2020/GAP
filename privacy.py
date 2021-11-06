import logging
import torch
import numpy as np
import torch.nn.functional as F
from autodp.autodp_core import Mechanism
import autodp.mechanism_zoo as mechanisms
from autodp.transformer_zoo import AmplificationBySampling, Composition
from torch_geometric.utils import remove_self_loops, add_self_loops
from scipy.optimize import minimize_scalar

class NullMechanism(Mechanism):
    def __init__(self):
        super().__init__()

    def update(self, _):
        self.__init__()


class GaussianMechanism(mechanisms.ExactGaussianMechanism):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        super().__init__(sigma=noise_scale)

    def update(self, noise_scale):
        self.__init__(noise_scale)

    def perturb(self, data, sensitivity):
        std = self.params['sigma'] * sensitivity
        return torch.normal(mean=data, std=std) if std else data

    def normalize(self, data):
        if self.noise_scale == 0:
            return data
        else:
            return F.normalize(data, p=2, dim=-1)

    def clip(self, data, c):
        if self.noise_scale == 0:
            return data
        else:
            return (c / data.norm(p=2, dim=-1, keepdim=True)).clamp(max=1) * data

    def get_approxDP(self, delta):
        if self.noise_scale == 0:
            return np.inf
        else:
            return super().get_approxDP(delta)


class LaplaceMechanism(mechanisms.LaplaceMechanism):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        super().__init__(b=noise_scale)

    def update(self, noise_scale):
        self.__init__(noise_scale)

    def perturb(self, data, sensitivity):
        scale = self.params['b'] * sensitivity
        return torch.distributions.Laplace(loc=data, scale=scale).sample() if scale else data

    def normalize(self, data):
        if self.noise_scale == 0:
            return data
        else:
            return F.normalize(data, p=1, dim=-1)

    def clip(self, data, c):
        if self.noise_scale == 0:
            return data
        else:
            return (c / data.norm(p=1, dim=-1, keepdim=True)).clamp(max=1) * data

    def get_approxDP(self, delta):
        if self.noise_scale == 0:
            return np.inf
        else:
            return super().get_approxDP(delta)
    

supported_mechanisms = {
    'laplace': LaplaceMechanism,
    'gaussian': GaussianMechanism
}


class TopMFilter(Mechanism):
    def __init__(self, noise_scale):
        super().__init__()
        self.noise_scale = noise_scale
        self.mech_edges = LaplaceMechanism(self.noise_scale)         # for individual edge perturbation
        self.mech_count = LaplaceMechanism(self.noise_scale * 9)     # for edge-count perturbation
        composed_mech = Composition()([self.mech_edges, self.mech_count], [1,1])
        self.set_all_representation(composed_mech)

    def update(self, noise_scale):
        self.__init__(noise_scale)

    def get_approxDP(self, delta):
        if self.noise_scale == 0:
            return np.inf
        else:
            return super().get_approxDP(delta)

    def build_mechanism(self, noise_scale, epochs, sampling_rate) -> Mechanism:
        if noise_scale == 0.0:
            return NullMechanism()

        self.update(noise_scale)
        
        if sampling_rate == 1.0:    
            return self
        else:
            subsample = AmplificationBySampling(PoissonSampling=True)
            subsampled_mech = subsample(self, prob=sampling_rate, improved_bound_flag=True)
            complex_mech = Composition()([subsampled_mech], [epochs])
            return complex_mech

    def perturb(self, data):
        if self.noise_scale == 0.0:
            return data

        data.edge_index, _ = remove_self_loops(data.edge_index)
        n = data.num_nodes
        m = data.num_edges
        m_pert = self.mech_count.perturb(m, sensitivity=1)
        m_pert = round(m_pert.item())
        eps_edges = 1 / self.mech_edges.noise_scale
        
        theta = np.log((n * (n-1) / m_pert) - 1) / (2 * eps_edges) + 0.5
        if theta > 1:
            theta = np.log((n * (n-1) / (2 * m_pert)) + 0.5 * (np.exp(eps_edges) - 1)) / eps_edges

        loc = torch.ones_like(data.edge_index[0]).float()
        sample = self.mech_edges.perturb(loc, sensitivity=1)
        edges_to_be_removed = data.edge_index[:, sample < theta]

        edge_index_with_self_loops, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        adjmat = self.to_sparse_adjacency(edge_index_with_self_loops, num_nodes=n)
        ### adjmat has m+n entries ###

        while True:
            adjmat = adjmat.coalesce()
            nnz = adjmat.values().size(0)
            num_remaining_edges = m_pert + n + edges_to_be_removed.size(1) - nnz

            if num_remaining_edges <= 0:
                break

            edges_to_be_added = torch.randint(n, size=(2, num_remaining_edges), device=adjmat.device)
            adjmat = adjmat + self.to_sparse_adjacency(edges_to_be_added, num_nodes=n)

        adjmat = (adjmat.bool().int() - self.to_sparse_adjacency(edges_to_be_removed, num_nodes=n)).coalesce()
        edge_index, values = adjmat.indices(), adjmat.values()
        data.edge_index = edge_index[:, values > 0].contiguous()
        data.edge_index, _ = remove_self_loops(data.edge_index)
        return data

    @staticmethod
    def to_sparse_adjacency(edge_index, num_nodes):
        return torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones_like(edge_index[0]).float(),
            size=(num_nodes, num_nodes),
            device=edge_index.device
        )


class Calibrator:
    def __init__(self, mechanism_cls):
        self.mechanism_cls = mechanism_cls

    def calibrate(self, eps, delta):
        mechanism = self.mechanism_cls(noise_scale=1)

        if eps == np.inf or isinstance(mechanism, NullMechanism):
            return 0.0
        else:
            logging.info('calibrating noise to privacy budget...')
            noise_scale = self.eps_delta_calibrator(eps, delta)
            logging.info(f'noise scale: {noise_scale:.4f}')
            return noise_scale
    
    def eps_delta_calibrator(self, eps, delta):
        fn_err = lambda x: abs(eps - self.mechanism_cls(x).get_approxDP(delta))
        results = minimize_scalar(fn_err, method='bounded', bounds=[0,1000], tol=1e-8, options={'maxiter': 1000000})

        if results.success and results.fun < 1e-3:
            return results.x
        else:
            raise RuntimeError(f"eps_delta_calibrator fails to find a parameter:\n{results}")
