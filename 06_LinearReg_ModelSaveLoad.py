#%% Packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

# %% Data Import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

# %% Visualize the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

# %% Convert data into tensors
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

# %% Create dataset and dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)

# %% Model Class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
    
input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_dim, output_dim)
model.train()

# %% Loss Function
loss_fun = nn.MSELoss()

# %% Optimizer
LR = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %% Check trainloader returns
# for i, (X, y) in enumerate(train_loader):
#     print(f"{i}th batch")
#     print(X)
#     print(y)

# %% Training
losses = []
slope, bias = [], []

NUM_EPOCHS = 1000
BATCH_SIZE = 2

for epoch in range(NUM_EPOCHS):
    for i, (X, y) in enumerate(train_loader):
        # set gradients to zero
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # compute loss
        loss = loss_fun(y_pred, y)
        losses.append(loss.item())

        # Backprop
        loss.backward()

        # update weights
        optimizer.step()

    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # store loss
    losses.append(float(loss.data))

    # print loss
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.data}')

# %% Model State Dict
model.state_dict()

# %% Save Model state dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# %% Load a model
model1 = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
model1.load_state_dict(torch.load('model_state_dict.pth'))
                      
model1.state_dict()