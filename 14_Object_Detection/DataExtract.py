# %% Download and Extract Dataset
import os
import zipfile
import kaggle
import shutil
import glob

# %% Define the dataset identifier and download directory
dataset_id = 'mbkinaci/fruit-images-for-object-detection'
download_dir = 'data'
zip_file_path = os.path.join(download_dir, 'fruit-images-for-object-detection.zip')

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


# %% Create Images Folder
# # Define source directories and target directory
source_dir1 = 'data/train_zip/train'
source_dir2 = 'data/test_zip/test'
target_dir = 'data/images'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Function to copy files from a source directory to the target directory
def copy_files(source_dir, target_dir):
    files = glob.glob(os.path.join(source_dir, '*.jpg'))  # Get all files in the source directory
    for file in files:
        shutil.copy(file, target_dir)  # Copy each file to the target directory

# Copy files from both source directories
copy_files(source_dir1, target_dir)
copy_files(source_dir2, target_dir)

print(f"Only .jpg files from '{source_dir1}' and '{source_dir2}' have been merged into '{target_dir}'.")

# %% Create Train and Test Labels Folder
# Define source directories and target directory
source_dir1 = 'data/train_zip/train'
source_dir2 = 'data/test_zip/test'
target_dir1 = 'data/train_labels'
target_dir2 = 'data/test_labels'

# Create the target directories if they don't exist
if not os.path.exists(target_dir1):
    os.makedirs(target_dir1)
if not os.path.exists(target_dir2):
    os.makedirs(target_dir2)

# Function to copy files from a source directory to the target directory
def copy_files(source_dir, target_dir):
    files = glob.glob(os.path.join(source_dir, '*.xml'))  # Get all files in the source directory
    for file in files:
        shutil.copy(file, target_dir)  # Copy each file to the target directory

# Copy files from both source directories
copy_files(source_dir1, target_dir1)
copy_files(source_dir2, target_dir2)

print(f"Only .xml files from '{source_dir1}' and '{source_dir2}' have been merged into '{target_dir1}' and '{target_dir2}'.")

# %%
