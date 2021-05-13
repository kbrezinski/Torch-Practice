
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * x

class myDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return(self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

## Prelude
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Produce the dataset iterable
x, y = produce_dataset() ## numpy dataset
dataset = myDataset(x, y) ## torch dataset_len
train_ds, val_ds = random_split(dataset, [80, 20])

params = {'batch_size':64, 'shuffle':True}
train_loader = DataLoader(train_ds, **params)
val_loader = DataLoader(val_ds, **params)

## Create the model and functions
lr = 1e-1
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')

## Create training function as wrapper
def step(model, loss_fn, optimizer, train=True):
    def train_step(x, y):

        model.train() if train else model.eval()

        y_pred = model(x)
        loss = loss_fn(y, y_pred)

        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()
    return train_step

## Perform training
train_step = step(model, loss_fn, optimizer)
val_step = step(model, loss_fn, optimizer, train=False)
losses = []
val_losses = []

for epoch in range(100):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = train_step(x, y)
        losses.append(loss)

    if epoch % 10 == 9:
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = val_step(x, y)
                val_losses.append(loss)
