import os
import boto3
import mlflow
import pandas as pd
from datetime import datetime

from src.utils import DEFAULT_BUCKET

def load_raw_data(raw_data_paths: dict, experiment_name: str, run_name: str = None) -> dict:
    def detect_data_source(paths: dict) -> str:
        for path in paths.values():
            if "oot" in path.lower():
                return "oot"
        return "train"

    # Infer data source type for dynamic naming
    if run_name is None:
        data_source = detect_data_source(raw_data_paths)
        run_name = f"load_{data_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_tracking_uri("http://18.219.91.131:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        s3 = boto3.client("s3")
        data_dict = {}

        for name, s3_key in raw_data_paths.items():
            filename = os.path.basename(s3_key)
            local_path = os.path.join("data/raw", filename)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            s3.download_file(DEFAULT_BUCKET, s3_key, local_path)

            df = pd.read_csv(local_path)
            mlflow.log_input(
                mlflow.data.from_pandas(df, source=local_path),
                context=f"raw_{name}"
            )
            mlflow.log_metric(f"{name}_rows", df.shape[0])

            data_dict[name] = {"df": df, "path": local_path}

    return data_dict
