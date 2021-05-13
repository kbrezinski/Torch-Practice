
import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dataset(dataset_len: int, seed: int = 2021) -> np.ndarray:
    np.random.seed(seed)
    x = np.random.rand(dataset_len, 1)
    y = 1 + 2 * x + .1 * np.random.randn(dataset_len, 1)
    return x, y

def init_weights(models):
    for model in models:
        _init_weights(model)

def _init_weights(m):
    if isinstance(m, list):
        for mm in m:
            _init_weights(mm)
    else:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)
        self.activation = nn.ReLU()
        init_weights(self.modules())
        self.loss_mse_fn = nn.MSELoss().to(device)

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, x, y):
        x = self.linear(x)
        x = self.activation(x)
        self.loss_mse = self.loss_mse_fn(x, y)
        return x

def produce_dataset(dataset_len=100):
    np.random.seed(2021)
    x = np.random.rand(dataset_len, 1)
    y = 1 + 2 * x + .1 * np.random.randn(dataset_len, 1)

    return x, y

class MyDataLoader(Dataset):
    def __init__(self):
        self.x, self.y = produce_dataset(dataset_len=100)
    def __getitem__(self, index):
        return(self.x[index], self.y[index])
    def __len__(self):
        return len(self.x)

def logger(
        exp_path,
        exp_name,
        work_dir,
        resume=False):

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    writer = SummaryWriter(os.path.join(exp_path, exp_name))
    log_file = os.path.join(exp_path, exp_name + '.txt')

    cfg_file = open('./config.py', 'r')
    cfg_lines = cfg_file.readlines()

    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n' * 4)

    if resume:
        ## copy over new directory files
        pass

    return writer, log_file
