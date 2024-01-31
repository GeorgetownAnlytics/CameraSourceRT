import os
import json
import random
import torch
import numpy as np
from typing import Dict
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
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
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
