import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch.nn import Parameter
from torch.nn import LayerNorm
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.nn.conv import MessagePassing
from utils import reset, uniform


class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class NodeLevelLayerNorm(LayerNorm):
    r"""
    Applies node level layer normalization over a batch of graph data.
    LayerNorm in/out: [N, **] number of examples, etc.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(normalized_shape))
            self.bias = Parameter(torch.Tensor(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return torch.functional.F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return 'normalized_shape={normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class DirectedMessagePassing(MessagePassing):
    r"""
    Applies Directed Message Passing over message m.
    Shape:
        - Input: batch of graph data
        - Output: batch of graph data
    """

    def __init__(self, in_channels, out_channels, nn,
                 aggr='add', root_weight=True, bias=True, **kwargs):
        super(DirectedMessagePassing, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr
        self.root = Parameter(torch.Tensor(in_channels, out_channels)) if root_weight else None
        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, m, edge_indem, edge_attr):
        m = m.unsqueeze(-1) if m.dim() == 1 else m
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_indem, m=m, pseudo=pseudo)

    def message(self, m_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(m_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, m):
        if self.root is not None: aggr_out = aggr_out + torch.mm(m, self.root)
        if self.bias is not None: aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class ResidualMessagePassingBlock(torch.nn.Module):
    def __init__(self,
                 in_dim=15,
                 out_dim=64,
                 edge_in_dim=7, ):
        super(ResidualMessagePassingBlock, self).__init__()
        edge_nn = Sequential(Linear(edge_in_dim, 64), ReLU(), Linear(64, in_dim * in_dim))
        self.mp = NNConv(in_dim, in_dim, edge_nn, aggr='mean', root_weight=True)
        edge_nn2 = Sequential(Linear(edge_in_dim, 64), ReLU(), Linear(64, in_dim * in_dim))
        self.dmp = DirectedMessagePassing(in_dim, in_dim, edge_nn2, aggr='mean', root_weight=True)
        self.fu = GRU(in_dim, in_dim)
        self.lin = torch.nn.Linear(in_dim, out_dim)
        self.bn = NodeLevelBatchNorm(num_features=out_dim)
        self.relu = torch.nn.ReLU()
        self.short_cut = Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, data):
        # self.fu.flatten_parameters()
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # phase 1
        h_gru = x.unsqueeze(0)
        for i in range(3):
            m = self.mp(x, edge_index, edge_attr)  # 去除 F.relu
            m = self.dmp(m, edge_index, edge_attr)
            x, h_gru = self.fu(m.unsqueeze(0), h_gru)
            x = x.squeeze(0)

        # phase 2
        res = self.relu(self.bn(self.lin(x)))

        # phase 3
        data.x = res + self.short_cut(x)
        return data


class BGNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=18,
                 edge_input_dim=7,
                 hidden_dim0=64,
                 hidden_dim1=64,
                 hidden_dim2=32,
                 hidden_dim3=32,
                 hidden_dim4=16,
                 hidden_dim5=16,
                 output_dim=1):
        super(BGNN, self).__init__()
        self.ffnn = torch.nn.Linear(node_input_dim, hidden_dim0)
        self.resmpblock0 = ResidualMessagePassingBlock(hidden_dim0, hidden_dim1, edge_input_dim)
        self.resmpblock1 = ResidualMessagePassingBlock(hidden_dim1, hidden_dim2, edge_input_dim)
        self.resmpblock2 = ResidualMessagePassingBlock(hidden_dim2, hidden_dim3, edge_input_dim)
        self.resmpblock3 = ResidualMessagePassingBlock(hidden_dim3, hidden_dim4, edge_input_dim)
        self.resmpblock4 = ResidualMessagePassingBlock(hidden_dim4, hidden_dim5, edge_input_dim)
        self.set2set = Set2Set(hidden_dim5, processing_steps=3)
        self.ffnn_out = torch.nn.Linear(hidden_dim5 * 2, output_dim)

    def forward(self, data):
        data.x = F.relu(self.ffnn(data.x))
        data = self.resmpblock0(data)
        data = self.resmpblock1(data)
        data = self.resmpblock2(data)
        data = self.resmpblock3(data)
        data = self.resmpblock4(data)
        x = self.set2set(data.x, data.batch)
        x = self.ffnn_out(x)
        return x.view(-1)


if __name__ == '__main__':
    from torch_geometric.data import Data

    data = Data(x=torch.rand([100, 18]),
                edge_attr=torch.rand([200, 7]),
                edge_index=torch.ones([2, 200]).long(),
                y=torch.ones([200]),
                batch=torch.zeros([200]).long())
    resmpblock0 = ResidualMessagePassingBlock(18, 18)
    print(resmpblock0(data))
