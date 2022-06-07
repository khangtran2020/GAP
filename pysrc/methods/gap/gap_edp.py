import numpy as np
import torch
import logging
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from pysrc.console import console
from pysrc.methods.gap import GAP
from pysrc.privacy.algorithms import PMA
from pysrc.classifiers.base import Metrics


class EdgePrivGAP (GAP):
    """edge-private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, dict(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 dict(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 **kwargs:      Annotated[dict,  dict(help='extra options passed to base class', bases=[GAP])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)
        
        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                logging.info('delta = %.0e', delta)
            
            self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            logging.info(f'noise scale: {self.noise_scale:.4f}\n')


    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        x = matmul(adj_t, x)
        x = self.pma_mechanism(x, sensitivity=1)
        return x
