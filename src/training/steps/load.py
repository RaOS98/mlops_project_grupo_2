import os
import boto3
import pandas as pd
import mlflow
from sagemaker.workflow.function_step import step
from src.utils import (
    DEFAULT_BUCKET,
    MLFLOW_URI,
    SAGEMAKER_ROLE
)
from src.training.config import INSTANCE_TYPE, IMAGE_URI

@step(
    name="LoadCleanTrainData",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE,
    dependencies="src/training/requirements.txt"
)
def load_train_data(experiment_name: str, run_name: str) -> tuple[str, str, str]:
    import io

    prefix = "data/processed/train"
    file_key = f"{prefix}/clean_train_data.csv"
    local_path = os.path.join(prefix, "clean_train_data.csv")
    os.makedirs(prefix, exist_ok=True)

    s3 = boto3.client("s3")

    # Check existence
    response = s3.list_objects_v2(Bucket=DEFAULT_BUCKET, Prefix=file_key)
    if "Contents" not in response or not any(obj["Key"] == file_key for obj in response["Contents"]):
        raise FileNotFoundError(f"File not found in S3 at key: {file_key}")

    # Download and inspect first few lines
    obj = s3.get_object(Bucket=DEFAULT_BUCKET, Key=file_key)
    content = obj["Body"].read()

    # Optional debug: print first few lines
    try:
        print("First few lines of S3 file:")
        print(content.decode("utf-8").splitlines()[:5])
    except Exception as e:
        print("Warning: Failed to decode file. Possibly not text:", e)

    # Try reading CSV
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Failed to load CSV from S3 content. Error: {e}")

    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="LoadCleanTrainData", nested=True):
            mlflow.log_input(mlflow.data.from_pandas(df, source=f"s3://{DEFAULT_BUCKET}/{file_key}"), context="train_data")
            mlflow.log_metric("train_rows", df.shape[0])

    return f"s3://{DEFAULT_BUCKET}/{file_key}", experiment_name, run_id
