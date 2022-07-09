import torch
from typing import Annotated, Literal, Optional, Union
import torch.nn.functional as F
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core.console import console
from core.methods.base import NodeClassificationBase
from core.classifiers import Encoder, MultiMLPClassifier
from core.classifiers.base import Metrics, Stage


class GAP (NodeClassificationBase):
    """Non-private GAP method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hops:            Annotated[int,   dict(help='number of hops', option='-k')] = 2,
                 hidden_dim:      Annotated[int,   dict(help='dimension of the hidden layers')] = 16,
                 encoder_layers:  Annotated[int,   dict(help='number of encoder MLP layers')] = 2,
                 base_layers:     Annotated[int,   dict(help='number of base MLP layers')] = 1,
                 head_layers:     Annotated[int,   dict(help='number of head MLP layers')] = 1,
                 combine:         Annotated[str,   dict(help='combination type of transformed hops', choices=MultiMLPClassifier.supported_combinations)] = 'cat',
                 activation:      Annotated[str,   dict(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, dict(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  dict(help='if true, then model uses batch normalization')] = True,
                 batch_size:      Annotated[Union[Literal['full'], int],   
                                                   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 encoder_epochs:  Annotated[int,   dict(help='number of epochs for encoder pre-training (ignored if encoder_layers=0)')] = 100,
                 **kwargs:        Annotated[dict,  dict(help='extra options passed to base class', bases=[NodeClassificationBase])]
                 ):

        super().__init__(num_classes, **kwargs)

        if encoder_layers == 0 and encoder_epochs > 0:
            console.warning('encoder_layers is 0, setting encoder_epochs to 0')
            encoder_epochs = 0

        self.hops = hops
        self.encoder_layers = encoder_layers
        self.encoder_epochs = encoder_epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        activation_fn = self.supported_activations[activation]

        self._encoder = Encoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self._classifier = MultiMLPClassifier(
            num_inputs=hops+1,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def classifier(self) -> MultiMLPClassifier:
        return self._classifier

    def reset_parameters(self):
        self._encoder.reset_parameters()
        super().reset_parameters()

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        self.data = data
        if self.encoder_layers > 0:
            self.data = self._pretrain_encoder(self.data, prefix=prefix)
        self.data = self._compute_aggregations(self.data)
        return super().fit(self.data, prefix=prefix)

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if data is None or data == self.data:
            data = self.data
        else:
            data.x = self._encoder.predict(data)
            data = self._compute_aggregations(data)

        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is None or data == self.data:
            data = self.data
        else:
            data.x = self._encoder.predict(data)
            data = self._compute_aggregations(data)

        return super().predict(data)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        return matmul(adj_t, x)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def _pretrain_encoder(self, data: Data, prefix: str) -> Data:
        console.info('pretraining encoder')
        self._encoder.to(self.device)

        self.trainer.fit(
            model=self._encoder,
            epochs=self.encoder_epochs,
            optimizer=self._configure_encoder_optimizer(), 
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=f'{prefix}encoder/',
        )

        self.trainer.reset()
        data.x = self._encoder.predict(data)
        return data

    def _compute_aggregations(self, data: Data) -> Data:
        with console.status('computing aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self._aggregate(x, data.adj_t)
                x = self._normalize(x)
                x_list.append(x)

            data.x = torch.stack(x_list, dim=-1)
        return data

    def data_loader(self, data, stage: Stage) -> DataLoader:
        mask = data[f'{stage}_mask']
        x = data.x[mask]
        y = data.y[mask]
        if self.batch_size == 'full' or (stage != 'train' and self.full_batch_eval):
            return [(x, y)]
        else:
            return DataLoader(
                dataset=TensorDataset(x, y),
                batch_size=self.batch_size, 
                shuffle=True
            )

    def _configure_encoder_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self._encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
