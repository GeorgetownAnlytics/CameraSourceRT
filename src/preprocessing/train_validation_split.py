import os
from tqdm import tqdm
from pathlib import Path
from config import paths
from utils import move_file, list_paths
from sklearn.model_selection import train_test_split


def get_image_paths_and_labels(train_folder_path):
    image_paths = []
    image_labels = []

    class_dirs_paths = [i for i in list_paths(train_folder_path) if os.path.isdir(i)]
    class_dirs = [Path(i).name for i in class_dirs_paths]
    for label in class_dirs:
        path = os.path.join(train_folder_path, label)
        images_files_names = [
            i
            for i in os.listdir(path)
            if i.lower().endswith(".jpg") or i.lower().endswith(".jpeg")
        ]
        images_files_paths = sorted([os.path.join(path, i) for i in images_files_names])
        image_paths.extend(images_files_paths)
        image_labels += [label] * len(images_files_paths)

    return image_paths, image_labels


def split_and_move_validation_files(image_paths, image_labels, validation_size):
    _, X_valid, _, y_valid = train_test_split(
        image_paths, image_labels, test_size=validation_size, random_state=42
    )
    for file_path, label in tqdm(
        zip(X_valid, y_valid), desc="Moving validation files..."
    ):
        file_name = Path(file_path).name
        destination_path = os.path.join(paths.VALIDATION_DIR, label, file_name)
        move_file(source_path=file_path, destination_path=destination_path)
