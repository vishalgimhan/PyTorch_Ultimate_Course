# %% Packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# %% Transform and Load Data
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# %% Model Architecture
CLASSES = ['artifact', 'extrahls', 'murmur', 'normal']
NUM_CLASSES = len(CLASSES)

class ImageMultiClassClassification(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.conv1(x) # out: (BS, 6, 100, 100)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 6, 50, 50)
        x = self.conv2(x) # out: (BS, 16, 50, 50)
        x = self.relu(x)
        x = self.pool(x) # out: (BS, 16, 25, 25)
        x = self.flatten(x) # out: (BS, 16 * 25 * 25)
        x = self.fc1(x) # out: (BS, 128)
        x = self.relu(x)
        x = self.fc2(x) # out: (BS, 64)
        x = self.relu(x)
        x = self.fc3(x) # out: (BS, NUM_CLASSES)
        x = self.softmax(x)
        return x
    
# input = torch.rand(1, 1, 100, 100) # BS=1, C=1, H=100, W=100
model = ImageMultiClassClassification()
# print(model(input).shape) # out: (1, NUM_CLASSES)

# %%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.otim.Adam(model.parameters())

# %% Training
losses_epoch_mean = []
NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    losses_epoch = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses_epoch.append(loss.item())
    losses_epoch_mean.append(np.mean(losses_epoch))
    print(f"Epoch {epoch}/{NUM_EPOCHS}, Loss: {np.mean(losses_epoch):.4f}")

# %% Plot Loss
sns.lineplot(x=list(range(len(losses_epoch_mean))), y=losses_epoch_mean)

# %% Testing
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()

    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %% Baseline Classifier
Counter(y_test)
# most dominant class: 12 obs
# length y_test = 24
# 12 / 24 = 50% accuracy

# %% Accuracy
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f"Accuracy: {acc*100:.2f}%")

# %% Confusion Matrix
cm = confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)