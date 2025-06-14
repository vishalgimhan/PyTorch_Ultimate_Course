# %% Packages
from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np
import torch

# %% Specify paths to data
path_images = 'data/images'
path_train_labels = 'data/train_labels'
path_test_labels = 'data/test_labels'

# %% Data Augmentation
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((50)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    utils.normalize_transform()
])
# %% Dataset and Dataloader
trained_labels = ['apple', 'banana']

train_dataset = core.Dataset(image_folder=path_images, label_data=path_train_labels, transform=custom_transforms)
test_dataset = core.Dataset(image_folder=path_images, label_data=path_test_labels, transform=custom_transforms) 

train_loader = core.DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = core.DataLoader(test_dataset, batch_size=2, shuffle=False)

# %% Initialize the Model
model = core.Model(trained_labels)

# %% Internal Model Structure
model.get_internal_model()

# %% Training
losses = model.fit(train_loader, test_loader, epochs=2, verbose=True)

# %% Show image with predictions
test_image_path = 'images/apple_77.jpg'
test_image = utils.read_image(test_image_path)
pred = model.predict(test_image)
labels, boxes, scores = pred

# %% Show image with predictions above threshold
threshold = 0.7
filtered_indices = np.where(scores > threshold)
filtered_scores = scores[filtered_indices]
filtered_boxes = boxes[filtered_indices]
num_list = filtered_boxes[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(test_image, filtered_boxes, filtered_labels)