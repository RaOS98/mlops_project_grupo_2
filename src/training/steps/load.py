import os
import pandas as pd
import mlflow
import awswrangler as wr
from sagemaker.workflow.function_step import step

from utils import MLFLOW_URI, SAGEMAKER_ROLE, DEFAULT_PATH
from training.config import IMAGE_URI, INSTANCE_TYPE
from preprocessing.steps.load import load_raw_data
from preprocessing.steps.merge import merge_raw_data
from preprocessing.steps.transform import PreprocessData

@step(
    name="LoadAndPreprocessTrainData",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE,
    dependencies="requirements.txt"
)
def load_train_data(experiment_name: str, run_name: str) -> tuple[str, str, str]:
    """
    Loads, merges, and preprocesses training data, saves locally and to S3,
    logs to MLflow, and returns the output path and experiment metadata.
    """
    # Step 1: Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    raw_data_paths = {
        "clientes": "data/raw/train_clientes_sample.csv",
        "requerimientos": "data/raw/train_requerimientos_sample.csv"
    }

    # Step 2: Load and merge raw data
    print("Loading and merging raw training data...")
    raw_data = load_raw_data(raw_data_paths, experiment_name)
    merged_df = merge_raw_data(raw_data["clientes"]["df"], raw_data["requerimientos"]["df"])

    temp_path = "data/temp/temp_train_merged.csv"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    merged_df.to_csv(temp_path, index=False)

    # Step 3: Preprocess the data
    print("Running preprocessing...")
    preprocessor = PreprocessData(ref_path=temp_path)  # Or ref path to cleaned base if available
    processed_df, _ = preprocessor.run(temp_path, is_train=True)

    # Step 4: Save to S3 and locally
    s3_key = "train/clean_train_data.csv"
    s3_path = f"s3://{DEFAULT_PATH}/{s3_key}"

    local_output_path = "data/processed/train/clean_train_data.csv"
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    processed_df.to_csv(local_output_path, index=False)
    wr.s3.to_csv(df=processed_df, path=s3_path, index=False)

    # Step 5: Log to MLflow
    print("Logging training data to MLflow...")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="LoadAndPreprocessTrainData", nested=True):
            mlflow.log_input(
                mlflow.data.from_pandas(processed_df, source=s3_path),
                context="train_data"
            )
            mlflow.log_metric("train_rows", processed_df.shape[0])

    print("Training data preprocessing and logging complete.")
    return s3_path, experiment_name, run_id
