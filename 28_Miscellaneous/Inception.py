# %% Packages
import torch
import torch.nn as nn

# %%
class ImageClassificationInception(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, kernel_size=1) -> None:
        super().__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )

        # max pool branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )

    def forward(self, x):
        out1x1 = self.branch1x1(x)
        out3x3 = self.branch3x3(x)
        out5x5 = self.branch5x5(x)
        out_pool = self.branch_pool(x)

        # Concatenate all branches
        out = torch.cat((out1x1, out3x3, out5x5, out_pool), dim=1)
        out = torch.flatten(out, 1) # reduce from 4dim to 1 dim
        out = nn.Linear(out.shape[1], 1)(out) # reduce to 1 neuron
        out = nn.Sigmoid()(out) # apply sigmoid activation for binary classification
        return out

# %% Test
input = torch.rand([4, 1, 32, 32]) #BS, Color Dim, Height, Width
model = ImageClassificationInception(in_channels=1, out_channels=128)
output = model(input)
output.shape
# %%
