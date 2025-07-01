from sagemaker.workflow.function_step import step
from src.utils import IMAGE_URI, INSTANCE_TYPE, MLFLOW_URI, DEFAULT_PATH, SAGEMAKER_ROLE
from src.batch_inference.config import (MODEL_NAME, MODEL_VERSION)


@step(
    name="ModelInference",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE
)
def model_inference(inf_raw_s3_path: str, experiment_name: str, run_id: str) -> tuple[str, str, str]:
    import pandas as pd
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    TARGET_COL = "ATTRITION"

    # Load input data from S3
    print(f"Loading input data from {inf_raw_s3_path}...")
    df = pd.read_csv(inf_raw_s3_path)

    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df.copy()  # fallback if label isn't present

    # Load model from MLflow
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # Run inference
    print("Running inference...")
    df["prob"] = model.predict_proba(X)[:, 1]

    # Save results to S3
    inf_proc_s3_path = f"s3://{DEFAULT_PATH}/oot/inference_result.csv"
    df.to_csv(inf_proc_s3_path, index=False)

    # Log to MLflow
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelInference", nested=True):
            mlflow.log_input(
                mlflow.data.from_pandas(df, inf_proc_s3_path),
                context="ModelInference"
            )

    print("Inference completed.")
    return inf_proc_s3_path, experiment_name, run_id
