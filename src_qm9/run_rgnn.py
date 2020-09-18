import sys
import warnings
import torch
import torch_geometric.transforms as T
from utils import Complete
from dataset import QM9Dataset
from trainer import Trainer
from model import RGNN

warnings.filterwarnings("ignore")
task = int(sys.argv[1])
gpu = int(sys.argv[2])
option = {
    'train_epoch': 400,
    'train_batch': 64,
    'task': task,  # [0~11]
    'lr': 1e-4,
    'lr_scheduler_patience': 5,

    'parallel': False,
    'cuda_devices': [gpu],  # works when parallel=True
    'early_stop_patience': -1,  # -1 for no early stop
}


class SpecifyTarget(object):
    def __call__(self, data):
        data.y = data.y[option['task']].view(-1)
        return data


print('Load Dataset...')
transform = T.Compose([SpecifyTarget(), Complete(), T.Distance(norm=True)])
path = '../qm9_dataset'
dataset = QM9Dataset(root=path, transform=transform).shuffle()

print('Split the dataset...')
one_tenth = len(dataset) // 10
test_dataset = dataset[: one_tenth]
valid_dataset = dataset[one_tenth: one_tenth * 2]
train_dataset = dataset[one_tenth * 2:]
assert len(train_dataset) + len(valid_dataset) + len(test_dataset) == len(dataset)

print('Training init...')
model = RGNN()
trainer = Trainer(option, model, \
                  train_dataset, valid_dataset, test_dataset)
trainer.train()

print('Testing')
trainer.load_best_ckpt()
trainer.valid_iterations(mode='test')





