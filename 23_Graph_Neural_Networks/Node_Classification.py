# %% Packages
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns

# inspired by Google Colab Notebook:
# https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing

# pip install torch_geometric , read docs

# %% Import Dataset
dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
print(f"Dataset: {dataset}")
print(f"# graphs: {len(dataset)}")
print(f"# features: {dataset.num_features}")
print(f"# classes: {dataset.num_classes}")

# %%
data = dataset[0]
print(data)

data.edge_index
data.y

# %%
from collections import Counter
Counter(data.y.cpu().numpy())

# %% Model Class
class GCN(torch.nn.Module):
    def __init__(self, num_hidden, num_features, num_classes) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2)
        x = self.conv2(x, edge_index)
        return x

model = GCN(num_hidden=16, num_features=dataset.num_features, num_classes=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# %%  Model Training
loss_lst = []
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(data.x, data.edge_index)
    y_true = data.y
    loss = criterion(y_pred[data.train_mask], y_true[data.train_mask])
    loss_lst.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss}")

# %% Train Loss
sns.lineplot(x = list(range(len(loss_lst))), y = loss_lst)

# %% Model Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(data.x, data.edge_index)
    # pick the index of highest class out of 3
    y_pred_cls = y_pred.argmax(dim=1)
    correct_pred = y_pred_cls[data.test_mask] == data.y[data.test_mask]
    test_acc = int(correct_pred.sum()) / int(data.test_mask.sum())
print(f"Test Accuracy: {test_acc:.4f}")

# %% Visualize Result
z = TSNE(n_components=2).fit_transform(y_pred[data.test_mask].detach().cpu().numpy())
sns.scatterplot(x=z[:, 0], y=z[: 1], hue=data.y[data.test_mask])