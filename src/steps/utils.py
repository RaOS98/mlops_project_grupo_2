import os

# ---- USER & ENVIRONMENT CONFIG ----
USERNAME = "diegovega9121"
ENV_CODE = "dev"  # Options: dev, stg, prod
DEFAULT_BUCKET = "my-batch-inference-data"

# ---- S3 PATH CONFIG ----
PROJECT_NAME = "bank-attrition"
default_prefix = f"sagemaker/{PROJECT_NAME}/{USERNAME}"
DEFAULT_PATH = f"s3://{DEFAULT_BUCKET}/{default_prefix}"

# ---- SAGEMAKER ROLE ----
SAGEMAKER_ROLE = "arn:aws:iam::686410906112:role/service-role/SageMaker-MLOpsEngineer"

# ---- MLflow CONFIG ----
TRACKING_SERVER_ARN = f"arn:aws:iam::686410906112:role/service-role/AmazonSageMaker-ExecutionRole-20250625T205217"

# ---- PIPELINE & MODEL NAMES ----
PIPELINE_NAME = f"pipeline-train-{ENV_CODE}-{USERNAME}"
MODEL_NAME = f"{PROJECT_NAME}-{USERNAME}"