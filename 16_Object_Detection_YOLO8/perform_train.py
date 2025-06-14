# %%
from ultralytics import YOLO

# %% Load the model
model = YOLO("yolov8n.pt")  # Load a pretrained model (YOLOv8n)

# %% Train the model
results = model.train(data="train_custom/masks.yaml", epochs=1, imgsz=512, batch=4, verbose=True, device='cpu')

# %% Export the model
model.export()

# %% 
import torch
torch.cuda.is_available()

# %%
