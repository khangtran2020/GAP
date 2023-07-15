from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class TrainTestSplit(BaseTransform):
    def __call__(self, data: Data):
        # mask = data.y.new_zeros(data.num_nodes, dtype=bool)
        # mask[data.edge_index[0]] = True
        # mask[data.edge_index[1]] = True
        # data = data.subgraph(mask)
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        tr_data = data.subgraph(train_mask)
        va_data = data.subgraph(val_mask)
        te_data = data.subgraph(test_mask)
        tr_data.train_mask = tr_data.y.new_ones(tr_data.num_nodes, dtype=bool)
        va_data.val_mask = va_data.y.new_ones(va_data.num_nodes, dtype=bool)
        te_data.test_mask = te_data.y.new_ones(te_data.num_nodes, dtype=bool)
        return tr_data, va_data, te_data
