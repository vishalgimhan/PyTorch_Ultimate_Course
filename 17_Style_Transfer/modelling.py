# %% Packages
import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
from torchvision import transforms
from torch.nn.functional import mse_loss
from torchvision import models

# %% Model
vgg = models.vgg19(pretrained=True).features

# %%
print(vgg)

# %% Image Transformations
preprocess_steps = transforms.Compose([
    transforms.Resize((200, 200)), # better (300, 300)
    transforms.ToTensor(),
])
content_img = Image.open('Hamburg.jpg').convert('RGB')
content_img = preprocess_steps(content_img)
# transform from C, H, W to H, W, C
# content_img = content_img.transpose(0, 2)
content_img = torch.unsqueeze(content_img, 0)
print(content_img.shape)

style_img = Image.open('The_Great_Wave_off_Kanagawa.jpg').convert('RGB')
style_img = preprocess_steps(style_img)
# style_img = style_img.transpose(0, 2)
style_img = torch.unsqueeze(style_img, 0)
print(style_img.shape)
# %% Feature Extraction
LOSS_LAYERS = { '0': 'conv1_1',
                '5': 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21': 'conv4_2',
                '28': 'conv5_1'}

def extract_features(x, model):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)

        if name in LOSS_LAYERS:
            features[LOSS_LAYERS[name]] = x
        
    return features
    
# %%
content_img_features = extract_features(content_img, vgg)
style_img_features = extract_features(style_img, vgg)

# %%
def calc_gram_matrix(tensor):
    _, C, H, W = tensor.size()
    tensor = tensor.view(C, H * W)
    gram_matrix = torch.mm(tensor, tensor.t())
    gram_matrix = gram_matrix.div(C * H * W) # normalization required
    return gram_matrix

style_features_gram_matrix = {layer: calc_gram_matrix(style_img_features[layer]) for layer in style_img_features}
style_features_gram_matrix

# %%
weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.6, 'conv4_1': 0.4, 'conv5_1': 0.2}

target = content_img.clone().requires_grad_()

optimizer = Adam([target], lr=0.01)

# %% Train the model
for i in range(1, 100):
    target_features = extract_features(target, vgg)

    content_loss = mse_loss(target_features['conv4_2'], content_img_features['conv4_2'])

    style_loss = 0
    for layer in weights:
        target_feature = target_features[layer]
        target_gram_matrix = calc_gram_matrix(target_feature)
        style_gram_matrix = style_features_gram_matrix[layer]

        layer_loss = mse_loss(target_gram_matrix, style_gram_matrix) * weights[layer]
        # layer_loss *=weights[layer]

        style_loss += layer_loss

    total_loss = 1000000 * style_loss + content_loss

    if i % 10 == 0:
        print(f"Epoch {i}:, Style loss: {style_loss}, Content Loss: {content_loss}")

    optimizer.zero_grad()

    total_loss.backward(retain_graph=True)

    optimizer.step()
    
# %% Get target image
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.25)

def tensor_to_image(tensor):
    image = tensor.clone().detach()
    image = image.cpu().nump().squeeze()

    image = image.transpose(1, 2, 0)

    image *= np.array(std) + np.array(mean)
    image = image.clip(0, 1)
    return image

import matplotlib.pyplot as plt
img = tensor_to_image(target)
fig = plt.figure()
fig.suptitle('Target Image')
plt.imshow(img)