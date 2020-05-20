import math
import random
import os
import torch
import numpy as np
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # (y_true, y_pred
from scipy.stats import spearmanr, kendalltau, pearsonr  # no matter order


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def metrics(y_true, y_pred, info='test'):  # true, pred np.float
    return {
        info + ' mse': mean_squared_error(y_true, y_pred),
        info + ' rmse': mean_squared_error(y_true, y_pred) ** 0.5,
        info + ' mae': mean_absolute_error(y_true, y_pred),
        info + ' r^2': r2_score(y_true, y_pred),
        info + ' pearson r': pearsonr(y_true.tolist(), y_pred.tolist())[0],
        info + ' spearman rho': spearmanr(y_true.tolist(), y_pred.tolist())[0],
        info + ' kendall tau': kendalltau(y_true.tolist(), y_pred.tolist())[0],
    }


def get_latest_ckpt(file_dir='./ckpt/'):
    filelist = os.listdir(file_dir)
    filelist.sort(key=lambda fn: os.path.getmtime(file_dir + fn) if not os.path.isdir(file_dir + fn) else 0)
    print('The latest ckpt is {}'.format(filelist[-1]))
    return file_dir + filelist[-1]
