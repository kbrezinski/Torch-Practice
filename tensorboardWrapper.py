
import os
import torch
import torch.nn as nn

from config import cfg
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (logger, create_dataset, MyModel, MyDataLoader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    def __init__(self, x, y):

        self.train_dl = DataLoader(MyDataLoader(), batch_size=64)

        self.net = MyModel().to(device)
        self.optimizer = torch.optim.Adam(
                            self.net.parameters(),
                            lr=1e-1,
                            weight_decay=1e-4)

        self.writer, self.log_file = logger(
            exp_path = cfg.EXP_PATH,
            exp_name = cfg.EXP_NAME,
            work_dir = os.path.abspath(__file__))

    def __call__(self):
        for epoch in range(1_000):
            self.train()

    def train(self):
        self.net.train()
        for i, data in enumerate(self.train_dl, 0):
            x, y = data
            x = x.to(device)
            y = y.to(device)

            y_pred = self.net(x, y)
            loss = self.net.loss

    def validate(self):
        self.net.eval()

x, y = create_dataset(dataset_len=1_000)
trainer = Trainer(x, y)
trainer()
