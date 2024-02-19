import os
import json
import random
import torch as T
import time
import shutil
import tracemalloc
import numpy as np
from pathlib import Path
from typing import Dict, Union
from config import paths


def read_json_as_dict(input_path: str) -> Dict:
    """
    Reads a JSON file and returns its content as a dictionary.
    If input_path is a directory, the first JSON file in the directory is read.
    If input_path is a file, the file is read.

    Args:
        input_path (str): The path to the JSON file or directory containing a JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.

    Raises:
        ValueError: If the input_path is neither a file nor a directory,
                    or if input_path is a directory without any JSON files.
    """
    if os.path.isdir(input_path):
        # Get all the JSON files in the directory
        json_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ]

        # If there are no JSON files, raise a ValueError
        if not json_files:
            raise ValueError("No JSON files found in the directory")

        # Else, get the path of the first JSON file
        json_file_path = json_files[0]

    elif os.path.isfile(input_path):
        json_file_path = input_path
    else:
        raise ValueError("Input path is neither a file nor a directory")

    # Read the JSON file and return it as a dictionary
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data_as_dict = json.load(file)

    return json_data_as_dict


def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)
    else:
        raise ValueError(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def get_model_parameters(
    model_name: str,
    hyperparameters_file_path: str = paths.HYPERPARAMETERS_FILE,
    hyperparameter_tuning: bool = False,
) -> dict:
    """
    Read hyperparameters from hyperparameters file.

    Args:
        model_name (str): Name of the model for which hyperparameters are read.
        hyperparameters_file_path (str): File path for hyperparameters.
        hyperparameter_tuning (bool): Whether hyperparameter tuning is used or not.


    """
    hyperparameters_dict = read_json_as_dict(hyperparameters_file_path)
    model_parameters = hyperparameters_dict[model_name]

    if not hyperparameter_tuning:
        hyperparameters = {i["name"]: i["default"] for i in model_parameters}

    else:
        # TODO: Read hyperparameters in case of tuning.
        pass

    return hyperparameters


def is_segment_in_path(segment: str, path: str) -> bool:
    """
    Check if the specified segment is a part of the given path.

    Args:
    - segment (str): The directory segment to check.
    - path (str): The full path to check against.

    Returns:
     - bool: True if the segment is part of the path, False otherwise.
    """
    # Create a Path object for the path
    path_obj = Path(path)

    # Iterate through each part of the path to check if the segment exists
    return segment in path_obj.parts


def replace_segment_in_path(
    original_path: str, old_segment: str, new_segment: str
) -> str:
    """
    Replace an old segment with a new segment in the given path.

    Args:
    - original_path (str): The original path as a string.
    - old_segment (str): The path segment to be replaced.
    - new_segment (str): The new segment to replace the old one.

    Returns:
    - str: The modified path with the segment replaced.
    """
    # Break down the original path into its components
    parts = list(Path(original_path).parts)

    # Replace the old segment with the new segment where found
    modified_parts = [new_segment if part == old_segment else part for part in parts]

    # Reconstruct the path from the modified parts
    # Use Path() to handle different OS path separators automatically
    modified_path = Path(*modified_parts)

    # Convert the Path object back to a string representation
    return str(modified_path)


def move_file(source_path: str, destination_path: str) -> None:
    """
    Move a file from a source path to a destination path.

    This function moves a file from the specified source path to the specified destination path.
    If the directory structure for the destination path does not exist, it is created.

    Parameters:
    - source_path (str): The path to the file that needs to be moved. This should include the full
      path along with the file name.
    - destination_path (str): The target path where the file should be moved. This includes the full
      path along with the new file name at the destination.

    Returns:
    None

    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)


def contains_subdirectories(folder_path):
    """Check if the given folder contains any subdirectories using pathlib."""
    folder = Path(folder_path)
    return any(entry.is_dir() for entry in folder.iterdir())


def list_paths(root_dir):
    """lists all dirs/file paths in a given directory."""
    paths = []
    listdir = [i for i in os.listdir(root_dir)]
    for entry in listdir:
        full_path = os.path.join(root_dir, entry)
        paths.append(full_path)

    return paths


def get_peak_memory_usage() -> Union[float, None]:
    """
    Returns the peak memory usage by current cuda device if available
    """
    if not T.cuda.is_available():
        return None

    current_device = T.cuda.current_device()
    peak_memory = T.cuda.max_memory_allocated(current_device)
    return peak_memory / 1e6


class TimeAndMemoryTracker(object):
    """
    This class serves as a context manager to track time and
    memory allocated by code executed inside it.
    """

    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cuda_peak = get_peak_memory_usage()

        elapsed_time = self.end_time - self.start_time

        self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")
        self.logger.info(f"Memory allocated (peak): {peak / 1024**2:.2f} MB")

        if cuda_peak:
            self.logger.info(f"CUDA Memory allocated (peak): {cuda_peak:.2f} MB")
