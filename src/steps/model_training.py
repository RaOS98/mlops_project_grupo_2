from steps.utils import ...
from sagemaker.workflow.function_step import step

@step(
    name="ModelTraining",
    instance_type="ml.m5.large",
    image_uri="686410906112.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    role=SAGEMAKER_ROLE,
)
def train(train_s3_path: str, experiment_name: str, run_id: str) -> tuple[str, str, str, str]:
    import subprocess
    subprocess.run(["pip", "install", "mlflow==2.13.2", "xgboost", "scikit-learn", "pandas"])

    import os
    import pandas as pd
    import mlflow
    from sklearn.model_selection import train_test_split
    from train_model import ModelTrainingPipeline

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    # Load full training data
    df = pd.read_csv(train_s3_path)
    TARGET_COL = "ATTRITION"
    SEED = 42
    TRAIN_SPLIT = 0.7

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=SEED)

    # Save test set to S3 for evaluation step
    test_df = pd.concat([X_test, y_test], axis=1)
    test_local_path = "/tmp/test_clean.csv"
    test_df.to_csv(test_local_path, index=False)

    test_s3_path = f"{DEFAULT_PATH}/test_data/test_clean.csv"

    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=test_local_path,
        Bucket=DEFAULT_PATH.replace("s3://", "").split("/")[0],
        Key="/".join(DEFAULT_PATH.replace("s3://", "").split("/")[1:] + ["test_data/test_clean.csv"])
    )

    # Save train set to local path for compatibility with existing pipeline
    train_local_path = "/tmp/train_clean.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_local_path, index=False)

    # Set up and run training pipeline
    pipeline = ModelTrainingPipeline(
        train_path=train_local_path,
        test_path=test_local_path,
        target_col=TARGET_COL
    )

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelTraining", nested=True) as training_run:
            training_run_id = training_run.info.run_id
            pipeline.run()

    return test_s3_path, experiment_name, run_id, training_run_id