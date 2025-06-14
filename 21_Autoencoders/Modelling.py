# %% Packages
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils

# %% Dataset and Dataloader
path_images = 'data/train'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(root=path_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# %% Model Class
LATENT_DIMS = 128

# Encoder Class
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # out: BS, 6, 62, 62
        self.conv2 = nn.Conv2d(6, 16, 3) # out: BS,, 16, 60, 60
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 60 * 60, LATENT_DIMS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Decoder Class
class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(LATENT_DIMS, 16 * 60 * 60)
        self.conv2 = nn.ConvTranspose2d(16, 6, 3) # out: BS, 6, 60, 60
        self.conv1 = nn.ConvTranspose2d(6, 1, 3) # out: BS, 6, 62, 62
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16, 60, 60) # from 2D (BS, feat) to 4D
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        return x

# Autoencoder Class
class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %% Test the model
model = Autoencoder()
input = torch.rand((1, 1, 64, 64))
model(input).shape

# %% Model Instance, Loss Function and Optimizer
model = Autoencoder()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 30

for epoch in range(NUM_EPOCHS):
    losses_epoch = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.view(-1, 1, 64, 64)
        optimizer.zero_grad()
        output = model(data)

        loss = F.mse_loss(output, data)
        losses_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch} \tLoss: {np.mean(losses_epoch)}')

# %% Visualize Original and Reconstructed images
def show_image(img):
    img = 0.5 * (img + 1) # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

images, labels = next(iter(dataloader))
print('Original')
plt.rcParams['figure.figsize'] = (20, 3)
show_image(torchvision.utils.make_grid(images))

# %% Latent Space
print('Latent Space')
latent_img = model.encoder(images)
latent_img = latent_img.view(-1, 1, 8, 16)
show_image(torchvision.utils.make_grid(latent_img))

# %%
print('Reconstructed')
show_image(torchvision.utils.make_grid(model(images)))

# %% Compression Rate
image_size = images.shape[2] * images.shape[3] * 1
compression_rate = (1 - LATENT_DIMS / image_size) * 100
print(f'Compression Rate: {compression_rate}%')
# %%
