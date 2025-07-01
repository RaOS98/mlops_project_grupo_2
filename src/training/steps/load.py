import os
import re
import boto3
import pandas as pd
import mlflow
from datetime import datetime
from sagemaker.workflow.function_step import step
from src.utils import (
    DEFAULT_BUCKET,
    sagemaker_session,
    EXPERIMENT_NAME,
    TRACKING_SERVER_ARN,
    SAGEMAKER_ROLE
)

from sagemaker import image_uris

IMAGE_URI = image_uris.retrieve(
    framework="xgboost",
    region="us-east-2",
    version="1.3-1"
)

# SageMaker step config
INSTANCE_TYPE = "ml.m5.large"

@step(
    name="LoadCleanTrainData",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE
)
def load_train_data(experiment_name: str, run_name: str) -> tuple[str, str, str]:
    import subprocess
    subprocess.run(['pip', 'install', 'boto3', 'pandas', 'mlflow'])  # ensure environment has deps

    import boto3
    import pandas as pd
    import mlflow
    import os
    import re
    from datetime import datetime

    prefix = "data/processed/"
    local_dir = "data/processed"
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=DEFAULT_BUCKET, Prefix=prefix)
    objects = response.get("Contents", [])

    train_files = [
        obj["Key"]
        for obj in objects
        if re.match(rf"{prefix}clean_train_data_\d{{8}}_\d{{6}}\.csv", obj["Key"])
    ]

    if not train_files:
        raise FileNotFoundError("No clean_train_data CSV files found in processed/")

    def extract_timestamp(key):
        match = re.search(r"clean_train_data_(\d{8}_\d{6})\.csv", key)
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else datetime.min

    latest_file_key = sorted(train_files, key=extract_timestamp)[-1]
    local_path = os.path.join(local_dir, "latest_train.csv")
    s3.download_file(DEFAULT_BUCKET, latest_file_key, local_path)

    df = pd.read_csv(local_path)

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        with mlflow.start_run(run_name="LoadCleanTrainData", nested=True):
            mlflow.log_input(mlflow.data.from_pandas(df, source=local_path), context="train_data")
            mlflow.log_metric("train_rows", df.shape[0])

    return local_path, experiment_name, run_id
