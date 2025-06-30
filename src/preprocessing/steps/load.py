import os
import boto3
import pandas as pd

from src.utils import DEFAULT_BUCKET

def load_raw_data(raw_data_paths: dict) -> dict:
    """
    Downloads and loads raw data files from S3 to local paths.

    Args:
        raw_data_paths (dict): Dictionary of {name: s3_key} for raw files.

    Returns:
        dict: Dictionary of {name: {'df': dataframe, 'path': local_path}}
    """
    s3 = boto3.client("s3")
    data_dict = {}

    for name, s3_key in raw_data_paths.items():
        filename = os.path.basename(s3_key)
        local_path = os.path.join("data/raw", filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3.download_file(DEFAULT_BUCKET, s3_key, local_path)
        df = pd.read_csv(local_path)

        data_dict[name] = {"df": df, "path": local_path}

    return data_dict
