from sagemaker.workflow.function_step import step
from src.utils import TRACKING_SERVER_ARN, SAGEMAKER_ROLE
from src.training.config import (INSTANCE_TYPE, IMAGE_URI)

@step(
    name="ModelRegistration",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE
)
def register_model(
    model_name: str,
    experiment_name: str,
    run_id: str,
    training_run_id: str,
):
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'sagemaker-mlflow==0.1.0'])

    import mlflow

    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelRegistration", nested=True):
            result = mlflow.register_model(
                model_uri=f"runs:/{training_run_id}/model",
                name=model_name
            )
            print(f"Registered model: {result.name} version {result.version}")
