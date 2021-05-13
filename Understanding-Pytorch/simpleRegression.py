
import numpy as np

def produce_dataset(train_split=.8, dataset_len=100):

    np.random.seed(2021)
    x = np.random.rand(dataset_len, 1)
    y = 1 + 2 * x + .1 * np.random.randn(dataset_len, 1)

    idx = np.arange(dataset_len)
    np.random.shuffle(idx)

    split_idx = int(dataset_len * train_split)
    train_idx, val_idx = idx[:split_idx], idx[split_idx:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return (x_train, y_train), (x_val, y_val)

a = np.random.randn(1)
b = np.random.randn(1)

lr = 1e-1
n_epochs = 1000

train_ds, val_ds = produce_dataset()
x_train, y_train = train_ds

for epoch in range(n_epochs):

    y_pred = a + b * x_train
    error = (y_train - y_pred)
    loss = (error ** 2).mean()

    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    a -= lr * a_grad
    b -= lr * b_grad
