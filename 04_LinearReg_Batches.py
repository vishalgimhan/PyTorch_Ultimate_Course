#%% Packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns

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

# %% Training
losses = []
slope, bias = [], []

NUM_EPOCHS = 1000
BATCH_SIZE = 2

for epoch in range(NUM_EPOCHS):
    for i in range(0, X.shape[0], BATCH_SIZE):
        # set gradients to zero
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X[i:i+BATCH_SIZE])

        # compute loss
        loss = loss_fun(y_pred, y_true[i:i+BATCH_SIZE])
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

# %%
len(bias)
NUM_EPOCHS

# %% Visualize Model Training
sns.scatterplot(x=range(len(losses)), y=losses)

# %% Visualize Bias Development
sns.lineplot(x=range(NUM_EPOCHS), y=bias)

# %% Visualize Slope development
sns.lineplot(x=range(NUM_EPOCHS), y=slope)

# %% Check results
model.eval()
y_pred = [i[0] for i in model(X).data.numpy()]
y = [i[0] for i in y_true.data.numpy()]
sns.scatterplot(x=X_list, y=y)
sns.lineplot(x=X_list, y=y_pred, color='red')

# %%
import hiddenlayer as hl
graph = hl.build_graph(model, X)
# %%
