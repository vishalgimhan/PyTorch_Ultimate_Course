 # %% Packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# %% Import dataset
iris = load_iris()
X = iris.data
y = iris.target

# %% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% Convert to Float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% Dtaset Class
class IrisData(Dataset):
    def __init__(self, X_train, X_test):
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
    
# %% Dataloader
iris_data = IrisData(X_train, y_train)
train_loader = DataLoader(iris_data, batch_size = 32, shuffle = True)

# %% Check Dims
print(f"X Shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}")

# %% Define class
class MultiClassNet(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES) #in and out
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x
    
# %% Hyperparameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN_FEATURES = 6
NUM_CLASSES = len(iris_data.y.unique())

# %% Model instance
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN_FEATURES)

# %% Loss Function
criterion = nn.CrossEntropyLoss()

# %% Optimizer
LR = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %% Training
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    for X, y in train_loader:
        # initialize gradients
        optimizer.zero_grad()

        # forward pass
        y_pred_log = model(X)

        # Calc losses
        loss = criterion(y_pred_log, y)

        # Backward pass
        loss.backward()

        # update weights
        optimizer.step()
    losses.append(float(loss.data.detach().numpy()))
losses

# %% Show losses over epochs
sns.lineplot(x=range(NUM_EPOCHS), y=losses)

# %% Test the Model
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_log = model(X_test_torch)
    y_test_pred = torch.max(y_test_log.data, 1)

print(y_test)
y_test_pred

# %% Accuracy
accuracy_score(y_test, y_test_pred.indices)

# %% Most common class
from collections import Counter
most_common_count = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier Accuracy: {most_common_count/len(y_test)*100}%")

# %% Save model state dict
torch.save(model.state_dict(), "model_iris.pt")
# %%

