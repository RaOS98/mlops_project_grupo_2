from batch_inference_utils import MLFLOW_URI, DEFAULT_PATH, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step

# Required for SageMaker
instance_type = "ml.m5.large"
image_uri = "885854791233.dkr.ecr.us-east-1.amazonaws.com/sagemaker-distribution-prod@sha256:92cfd41f9293e3cfbf58f3bf728348fbb298bca0eeea44464968f08622d78ed0"

@step(
    name="DataPull",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE,
)
def load_and_preprocess_oot(experiment_name: str, run_name: str) -> tuple[str, str, str]:
    """
    Loads, merges, and preprocesses the OOT datasets, uploads to S3, logs to MLflow.
    Returns (s3_path, experiment_name, run_id)
    """
    import os
    import mlflow
    import awswrangler as wr
    from src.preprocessing.steps.load import load_raw_data
    from src.preprocessing.steps.merge import merge_raw_data
    from src.preprocessing.steps.transform import PreprocessData

    # Step 1: Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    raw_data_paths = {
        "clientes": "data/raw/oot_clientes_sample.csv",
        "requerimientos": "data/raw/oot_requerimientos_sample.csv"
    }

    # Step 2: Load and merge raw data
    print("Loading raw data...")
    raw_data = load_raw_data(raw_data_paths=raw_data_paths, experiment_name=experiment_name)
    merged_df = merge_raw_data(raw_data["clientes"]["df"], raw_data["requerimientos"]["df"])

    temp_path = "data/temp/temp_oot_merged.csv"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    merged_df.to_csv(temp_path, index=False)

    # Step 3: Preprocess
    print("Running preprocessing...")
    preprocessor = PreprocessData(ref_path="data/processed/train_clean.csv")  # use training reference
    processed_df, _ = preprocessor.run(temp_path, is_train=False)

    # Step 4: Save to S3 and locally
    s3_key = "oot/clean_oot_data.csv"
    s3_path = f"s3://{DEFAULT_PATH}/{s3_key}"

    print(f"Saving preprocessed data to S3: {s3_path}")
    wr.s3.to_csv(df=processed_df, path=s3_path, index=False)

    local_output_path = "data/processed/oot/clean_oot_data.csv"
    os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
    processed_df.to_csv(local_output_path, index=False)

    # Step 5: Log to MLflow
    print("Logging to MLflow...")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="DataPull", nested=True):
            mlflow.log_input(
                mlflow.data.from_pandas(processed_df, s3_path),
                context="DataPull"
            )

    print("OOT load and preprocessing completed.")
    return s3_path, experiment_name, run_id
