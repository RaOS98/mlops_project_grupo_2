import os
import pandas as pd
import mlflow
from sagemaker.workflow.function_step import step
from src.utils import (
    SAGEMAKER_ROLE,
    TRACKING_SERVER_ARN,
    DEFAULT_BUCKET
)

# You can retrieve the SageMaker-provided XGBoost image dynamically
from sagemaker import image_uris
IMAGE_URI = image_uris.retrieve(
    framework="xgboost",
    region="us-east-2",
    version="1.3-1"
)

INSTANCE_TYPE = "ml.m5.large"

@step(
    name="TrainModel",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE
)
def train_model(train_s3_path: str, experiment_name: str, run_id: str) -> tuple[str, str, str, str]:
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'sagemaker-mlflow==0.1.0'])

    import pandas as pd
    import mlflow
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    TARGET_COL = "ATTRITION"
    SEED = 42
    TRAIN_SPLIT = 0.7

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(train_s3_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SPLIT, random_state=SEED
    )

    test_s3_path = f"s3://{DEFAULT_BUCKET}/test_data/test.csv"
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test.to_csv(test_s3_path, index=False)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelTraining", nested=True) as training_run:
            training_run_id = training_run.info.run_id

            mlflow.log_input(
                mlflow.data.from_pandas(df_test, test_s3_path, targets=TARGET_COL),
                context="ModelTraining"
            )

            mlflow.xgboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_datasets=True,
                model_format="xgb",
            )

            xgb = XGBClassifier(
                objective="binary:logistic",
                max_depth=5,
                eta=0.2,
                gamma=4,
                min_child_weight=6,
                subsample=0.7,
                tree_method="hist",  # or "gpu_hist" if GPU used
                n_estimators=50
            )
            xgb.fit(X_train, y_train)

    return test_s3_path, experiment_name, run_id, training_run_id
