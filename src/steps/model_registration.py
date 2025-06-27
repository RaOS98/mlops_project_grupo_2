from steps.utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step

# Global config
instance_type = "ml.m5.large"
image_uri = "686410906112.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"  # Region-specific ECR image

@step(
    name="ModelRegistration",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE,
)
def register(
    model_name: str,
    experiment_name: str,
    run_id: str,
    training_run_id: str,
):
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'sagemaker-mlflow==0.1.0'])

    import mlflow
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelRegistration", nested=True):
            try:
                result = mlflow.register_model(
                    f"runs:/{training_run_id}/model",
                    model_name
                )
                print(f"Model registered successfully: {model_name}, version {result.version}")
            except MlflowException as e:
                print(f"Model registration failed: {str(e)}")