# %% Packages
from datasets import load_dataset, list_datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import variable
from PIL import Image
import os

# %% Load Model
model = models.resnet18(pretrained=True)
# Remove the last layer (fully connected layer)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()  # Set the model to evaluation mode

# %% Preprocess image function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.456], std=[0.229,0.224,0.225])
])
# %% Load Image
image_path = '../14_Object_Detection/data/images/'
image_files = os.listdir(image_path)
img = Image.open(image_path + image_files[0]).convert('RGB')
img

# preprocess(img).shape
# preprocess(img).unsqueeze(0).shape

# %% Create embeddings for candidate images
embeddings = []
for i in range(100):
    img = Image.open(image_path + image_files[i]).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        embedding = model(img_tensor)
        embedding = embedding[0, :, 0, 0]

    embeddings.append(embedding)
# %%
print(len(embeddings))
# %% Compare embedding to target image
sample_image = Image.open(image_path + image_files[101]).convert('RGB')
sample_tensor = preprocess(sample_image).unsqueeze(0)
with torch.no_grad():
    sample_embedding = model(sample_tensor)
    sample_embedding = sample_embedding[0, :, 0, 0]

# %% Cosine similarity function
similarieties = []
for i in range(len(embeddings)):

    # cosine method
    similarity = torch.nn.functional.cosine_similarity(sample_embedding, embeddings[i], dim=0).tolist()

    # euclidean distance
    # similarity = torch.dist(sample_embedding, embeddings[i], p=2)

    similarieties.append(similarity)
# %%
idx_max_similarity = similarieties.index(max(similarieties))
image_files[idx_max_similarity]

# %%
image_files[101]