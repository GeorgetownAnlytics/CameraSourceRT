import os
import random
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from config import paths
from utils import read_json_as_dict

# Set a random seed for reproducibility
config_dict = read_json_as_dict(paths.CONFIG_FILE)
random.seed(config_dict["seed"])

# Specify the paths to the source and destination directories
src_data_dir = paths.RAW_DATA_DIR
dest_data_dir = paths.VISION_DATA_DIR
split_dir = paths.SPLIT_DATA_DIR

Path(split_dir).mkdir(parents=True, exist_ok=True)

# Split ratios for training, testing, and validation sets
train_split = 0.7
test_split = 0.2
# Validation split is the remaining percentage

# Initialize dictionaries to hold file lists for JSON
split_file_lists = {"train": [], "test": [], "validation": []}


# Function to create category subfolders in train, test, and validation directories
def create_category_folders(base_dir, class_name, subfolder_name):
    for folder_type in ["train", "test", "validation"]:
        folder_path = os.path.join(base_dir, folder_type, class_name, subfolder_name)
        Path(folder_path).mkdir(parents=True, exist_ok=True)


# Function to split data into train, test, and validation sets and copy the files
def split_data(class_name, subfolder_name, images):
    random.shuffle(images)
    train_end = int(len(images) * train_split)
    test_end = train_end + int(len(images) * test_split)

    train_images = images[:train_end]
    test_images = images[train_end:test_end]
    validation_images = images[test_end:]

    # Copy function
    def copy_images(image_list, target_dir, split_name):
        for image in tqdm(image_list, desc=f"Copying to {target_dir}"):
            dest_path = os.path.join(
                target_dir, class_name, subfolder_name, os.path.basename(image)
            )
            shutil.copy(image, dest_path)
            # Add the image path relative to dest_data_dir to the split's file list
            relative_path = os.path.relpath(dest_path, dest_data_dir)
            split_file_lists[split_name].append(relative_path)

    # Copy images to train, test, and validation directories
    print(f"Processing Training Set for {class_name}/{subfolder_name}")
    copy_images(train_images, os.path.join(dest_data_dir, "train"), "train")
    print(f"Processing Testing Set for {class_name}/{subfolder_name}")
    copy_images(test_images, os.path.join(dest_data_dir, "test"), "test")
    print(f"Processing Validation Set for {class_name}/{subfolder_name}")
    copy_images(
        validation_images, os.path.join(dest_data_dir, "validation"), "validation"
    )


# Main script to process and split images
for class_folder in os.listdir(src_data_dir):
    class_path = os.path.join(src_data_dir, class_folder, "images")
    if os.path.isdir(class_path):
        for subfolder_name in os.listdir(class_path):
            subfolder_path = os.path.join(class_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                images = [
                    os.path.join(subfolder_path, f)
                    for f in os.listdir(subfolder_path)
                    if f.lower().endswith(".jpg")
                ]
                create_category_folders(dest_data_dir, class_folder, subfolder_name)
                split_data(class_folder, subfolder_name, images)

# Write the split file lists to JSON files in the split directory
for split_name, file_list in split_file_lists.items():
    json_path = os.path.join(split_dir, f"{split_name}_files.json")
    with open(json_path, "w") as json_file:
        json.dump(file_list, json_file, indent=4)

print(
    f"Dataset split completed. Training, testing, and validation are located in {dest_data_dir}"
)
print(f"JSON files of the splits are saved in {split_dir}")
