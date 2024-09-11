import os

from dataset import Datasets
from llms import Model


def validate_input(dataset_type, model_name, n_tasks, n_samples, seed, result_path, result_folder):

    # check if dataset_type is valid.
    if dataset_type not in Datasets.list():  # Datasets is an Enum
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    # check if model_name is valid.
    if model_name not in Model.list():
        raise ValueError(f"Invalid model name: {model_name}")

    if not os.path.isdir(result_path):
        raise FileNotFoundError(f"Result path doesn't exist: {result_path}")

    result_path = os.path.join(result_path, result_folder)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        print(f"Result folder created: {result_path}")
    else:
        raise FileExistsError(f"Result folder already exists: {result_path}")

    return True
