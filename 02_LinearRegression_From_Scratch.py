#%% Packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

# %% Import Data
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

# %% Visualize the Model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

# %% Convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values.tolist()
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)

X = torch.from_numpy(X_np)
y = torch.tensor(y_list)

# %% Training
w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    for i in range(len(X)):
        # forward pass
        y_pred = X[i] * w + b

        # calculate loss
        loss_tensor = torch.pow(y_pred - y[i], 2)

        # backward_pass
        loss_tensor.backward()

        # extract losses
        loss_value = loss_tensor.data[0]

        # update weights and biases
        with torch.no_grad():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            w.grad.zero_()
            b.grad.zero_()
    print(loss_value)


# %% Check results
print(f"Weight: {w.item()}, Bias: {b.item()}")

# %%
y_pred = (torch.tensor(X_list) * w + b).detach().numpy()

# %%
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')

# %% (Statistical) Linear Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_np, y_list)
print(f"Slope: {reg.coef_}, Intercept: {reg.intercept_}")

# %% Create Graph Visualization
import os
from torchviz import make_dot
os.environ['PATH'] += os.pathsep + 'C:/Program FIles (x86)/GraphViz/bin'

make_dot(loss_tensor)

# %%
