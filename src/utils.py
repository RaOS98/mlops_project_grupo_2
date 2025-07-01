import os
import sagemaker

DEFAULT_BUCKET = "utec-bank-project"
ENV_CODE = "prod"
USERNAME = os.getenv("GITHUB_ACTOR", "local-user").lower()

TRACKING_SERVER_ARN = 'arn:aws:sagemaker:us-east-1:654654589924:mlflow-tracking-server/mlops-utec-mlflow-server'

# SageMaker role and session
SAGEMAKER_ROLE = "arn:aws:iam::686410906112:role/service-role/AmazonSageMaker-ExecutionRole-20250625T205217"
default_prefix = f"sagemaker/{DEFAULT_BUCKET}/{USERNAME}"
DEFAULT_PATH = f"{DEFAULT_BUCKET}/{default_prefix}"

sagemaker_session = sagemaker.Session(
    default_bucket=DEFAULT_BUCKET,
    default_bucket_prefix=default_prefix,
)

# Resource naming
# PIPELINE_NAME = f"pipeline-preprocessing-{ENV_CODE}-{USERNAME}"
PIPELINE_NAME_TRAINING = f"pipeline-training-{ENV_CODE}-{USERNAME}"
MODEL_NAME = f"{DEFAULT_BUCKET}-{USERNAME}"

EXPERIMENT_NAME = "utec-bank"
