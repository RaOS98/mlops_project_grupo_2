import os
import boto3
import pandas as pd
import mlflow
from sagemaker.workflow.function_step import step
from src.utils import (
    DEFAULT_BUCKET,
    TRACKING_SERVER_ARN,
    SAGEMAKER_ROLE
)
from src.training.config import (INSTANCE_TYPE, IMAGE_URI)

@step(
    name="LoadCleanTrainData",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE,
    dependencies="src/training/requirements.txt"
)
def load_train_data(experiment_name: str, run_name: str) -> tuple[str, str, str]:
    prefix = "data/processed/train"
    local_dir = "data/processed/train"
    os.makedirs(local_dir, exist_ok=True)

    s3 = boto3.client("s3")
    file_key = f"{prefix}clean_train_data.csv"
    local_path = os.path.join(local_dir, "clean_train_data.csv")

    # Check if file exists in S3
    response = s3.list_objects_v2(Bucket=DEFAULT_BUCKET, Prefix=file_key)
    if "Contents" not in response or len(response["Contents"]) == 0:
        raise FileNotFoundError(f"No file found in S3 at key {file_key}")

    # Download to local file
    s3.download_file(DEFAULT_BUCKET, file_key, local_path)

    # Load into pandas
    df = pd.read_csv(local_path)

    # Set MLflow config
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        with mlflow.start_run(run_name="LoadCleanTrainData", nested=True):
            mlflow.log_input(mlflow.data.from_pandas(df, source=local_path), context="train_data")
            mlflow.log_metric("train_rows", df.shape[0])

    return local_path, experiment_name, run_id
