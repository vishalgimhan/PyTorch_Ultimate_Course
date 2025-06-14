# %% Packages
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter

# %% Data Prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2)

# %% Dataset and Dataloader
class MultiLabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# %% Instance of dataset
multilabel_train_data = MultiLabelDataset(X_train, y_train)
multilabel_test_data = MultiLabelDataset(X_test, y_test)

# %% Train Loader
train_loader = DataLoader(multilabel_train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(multilabel_test_data, batch_size=32, shuffle=True)

# %% Model
class MultiLabelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# %%
input_dim = multilabel_train_data.X.shape[1]
output_dim = multilabel_train_data.y.shape[1]

# %% Model instance
model = MultiLabelNetwork(input_size=input_dim, hidden_size=20, output_size=output_dim)

# %% Loss function, OPtimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
slope, bias = [], []
num_epochs = 100

# %% training loop
for epoch in range(num_epochs):
    for j, (X, y) in enumerate(train_loader):

        # optimization xero grad
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)

        # losses
        loss = loss_fn(y_pred, y)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
    
    # print losses and loss at every 10th epoch
    if epoch%10 == 0:
        print(f'epoch: {epoch}, loss: {loss.data.item()}')
        losses.append(loss.item())

# %% Losses
sns.scatterplot(x=range(len(losses)), y=losses)

# %% Test momdel
with torch.no_grad(): 
    y_test_pred = model(X_test).round()

# %% Naive Classifier Accuracy
# Convert y_test tensor to list of strings
y_test_str = [str(i) for i in y_test.detach().numpy()]
Counter(y_test_str)

# most common class count
most_common_count = Counter(y_test_str).most_common()[0][1]
print(f"Naive Classifier Accuracy: {most_common_count/len(y_test_str)*100}%")

# %% Test accuracy
accuracy_score(y_test, y_test_pred)