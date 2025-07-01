from batch_inference_utils import (
    TRACKING_SERVER_ARN,
    DEFAULT_PATH,
    SAGEMAKER_ROLE,
    MODEL_NAME,
    MODEL_VERSION
)
from sagemaker.workflow.function_step import step

# Global variables
instance_type = "ml.m5.large"
image_uri = "885854791233.dkr.ecr.us-east-1.amazonaws.com/sagemaker-distribution-prod@sha256:92cfd41f9293e3cfbf728348fbb298bca0eeea44464968f08622d78ed0"

@step(
    name="ModelInference",
    instance_type=instance_type,
    image_uri=image_uri,
    role=SAGEMAKER_ROLE
)
def model_inference(inf_raw_s3_path: str, experiment_name: str, run_id: str) -> tuple[str, str, str]:
    import pandas as pd
    import mlflow

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    # Define features used in training
    FEATURES = [
        "CODMES",
        "CODMES_ANT",
        "DIF_CODMES",
        "EDAD",
        "SEXO",
        "SEGMENTO_2",
        "NRO_DEPEN",
        "RENTA",
        "PRODUCTO",
        "LINEA",
        "DIAS_MORA",
        "MORA_MAX",
        "TIPO_CLIENTE",
        "PROB_DEFAULT",
        "CODMES_INGRESO"
    ]

    # Load input data from S3
    print(f"Loading input data from {inf_raw_s3_path}...")
    df = pd.read_csv(inf_raw_s3_path)
    X = df[FEATURES]

    # Load model from MLflow
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
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

    print("✅ Inference completed.")
    return inf_proc_s3_path, experiment_name, run_id