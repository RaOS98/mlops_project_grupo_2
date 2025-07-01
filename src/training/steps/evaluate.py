from src.utils import MLFLOW_URI, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step
from src.training.config import (INSTANCE_TYPE, IMAGE_URI)

@step(
    name="ModelEvaluation",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE
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

    TARGET_COL = "ATTRITION"

    mlflow.set_tracking_uri(MLFLOW_URI)
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

            return {
                "f1_score": results.metrics.get("f1_score", None),
                "accuracy": results.metrics.get("accuracy_score", None),
                "roc_auc": results.metrics.get("roc_auc_score", None)
            }
