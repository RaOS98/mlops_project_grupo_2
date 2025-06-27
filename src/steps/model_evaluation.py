from steps.utils import ...
from sagemaker.workflow.function_step import step

# Global configuration
instance_type = "ml.m5.large"
image_uri = "686410906112.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

@step(
    name="ModelEvaluation",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE,
)
def evaluate(
    test_s3_path: str,
    experiment_name: str,
    run_id: str,
    training_run_id: str,
) -> dict:
    import subprocess
    subprocess.run(['pip', 'install', 'mlflow==2.13.2', 'sagemaker-mlflow==0.1.0'])

    import mlflow
    import pandas as pd

    TARGET_COL = "ATTRITION"  # updated for your bank attrition target

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelEvaluation", nested=True):
            test_df = pd.read_csv(test_s3_path)
            model = mlflow.pyfunc.load_model(f"runs:/{training_run_id}/model")

            results = mlflow.evaluate(
                model=model,
                data=test_df,
                targets=TARGET_COL,
                model_type="classifier",
                evaluators=["default"],
            )

            return {"f1_score": results.metrics["f1_score"]}