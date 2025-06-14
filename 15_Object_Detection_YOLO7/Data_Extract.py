# %% Download and Extract Dataset
import os
import zipfile
import kaggle
import shutil
import glob

# %% Define the dataset identifier and download directory
dataset_id = 'andrewmvd/face-mask-detection'
download_dir = 'data'
zip_file_path = os.path.join(download_dir, 'face-mask-detection.zip')

# Create download directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Download the dataset
kaggle.api.dataset_download_files(dataset_id, path=download_dir, unzip=False)

# Check if the zip file already exists, remove the old file if it does
if os.path.exists(zip_file_path):
    os.remove(zip_file_path)

# Download the latest dataset again
kaggle.api.dataset_download_files(dataset_id, path=download_dir, unzip=False)

# Extract the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

print(f'Dataset updated and extracted to {download_dir}')

# %%
