import os
import pandas as pd
import mlflow
from sagemaker.workflow.function_step import step
from src.utils import (
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
    prefix = "data/processed/train"
    local_path = os.path.join(prefix, "clean_train_data.csv")
    os.makedirs(prefix, exist_ok=True)

    # Check local file existence
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found at path: {local_path}")

    # Load CSV from local path
    try:
        df = pd.read_csv(local_path)
        print("First few lines of local file:")
        print(df.head())
    except Exception as e:
        raise ValueError(f"Failed to load CSV from local path. Error: {e}")

    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="LoadCleanTrainData", nested=True):
            mlflow.log_input(
                mlflow.data.from_pandas(df, source=local_path),
                context="train_data"
            )
            mlflow.log_metric("train_rows", df.shape[0])

    return local_path, experiment_name, run_id