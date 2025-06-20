# %% Packages
from collections import OrderedDict
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# %% Dataset
# https://www.microsoft.com/en-us/download/details.aspx?id=54765

train_dir = 'train'
test_dir = 'test'

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
def imshow(image_torch):
    # flip image channels to RGB
    image_torch = image_torch.numpy().transpose(1, 2, 0)
    plt.figure()
    plt.imshow(image_torch)

X_train, y_train = next(iter(train_loader))

# Make a grid from batch
image_grid = torchvision.utils.make_grid(X_train[:16, :, :, :], scale_each=True, nrow=4)

imshow(image_grid)

# %% Download and Instantiate Pretrained Network
model = models.densenet121(pretrained=True)

# %% Freeze all layers
for params in model.parameters():
    params.requires_grad = False
    
 # %% Overwrite classifier of the model
model.classifier #existing
model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 1)),
    ('Output', nn.Sigmoid())
]))

# %% Optimizer and Loss Function
opt = optim.Adam(model.classifier.parameters())
loss_function = nn.BCELoss()
train_losses = []

# %% Training Loop
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_loss = 0
    test_loss = 0
    for bat, (img, label) in enumerate(train_loader):

        # Zero the gradients
        opt.zero_grad()

        # forward pass
        output = model(img)

        # losses
        loss = loss_function(output.squeeze(), label.float())

        # propagate losses
        loss.backward()

        # update weights
        opt.step()

        # update current train loss 
        train_loss += loss.item()

    train_losses.append(train_loss)
    print(f"Epoch: {epoch}. Train Loss: {train_loss}")
# %% show losses over epoch
sns.lineplot(x = range(len(train_losses)), y = train_losses)

# %%
fig = plt.figure(figsize=(10, 10)) 
class_labels = {0:'cat', 1:'dog'} 
X_test, y_test = next(iter(test_loader)) 
with torch.no_grad():
    y_pred = model(X_test) 
    y_pred = y_pred.round()
    y_pred = [p.item() for p in y_pred] 

# create subplots
for num, sample in enumerate(X_test): 
    plt.subplot(4,6,num+1) 
    plt.title(class_labels[y_pred[num]]) 
    plt.axis('off') 
    sample = sample.cpu().numpy() 
    plt.imshow(np.transpose(sample, (1,2,0))) 

# %% accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {np.round(acc * 100, 2)} %")
# %%
