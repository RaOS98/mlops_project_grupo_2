from src.preprocessing.steps.load import load_raw_data
from src.preprocessing.steps.merge import merge_raw_data
from src.preprocessing.steps.transform import PreprocessData
from src.utils import EXPERIMENT_NAME, MLFLOW_URI
import mlflow

def load_and_preprocess_oot(cod_month: int) -> tuple[str, str, str]:
    # Setup
    experiment_name = EXPERIMENT_NAME
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)
    
    raw_data_paths = {
        "clientes": f"data/oot/clientes_{cod_month}.csv",
        "requerimientos": f"data/oot/requerimientos_{cod_month}.csv"
    }

    # Load and preprocess
    raw_data = load_raw_data(raw_data_paths=raw_data_paths, experiment_name=experiment_name)
    merged_df = merge_raw_data(raw_data["clientes"]["df"], raw_data["requerimientos"]["df"])
    temp_path = f"data/processed/temp_merged_{cod_month}.csv"
    merged_df.to_csv(temp_path, index=False)

    preprocessor = PreprocessData(ref_path="data/processed/train_clean.csv")
    processed_df, _ = preprocessor.run(temp_path, is_train=False)

    # Save to S3
    s3_path = f"s3://{DEFAULT_PATH}/inf-raw-data/{cod_month}.csv"
    wr.s3.to_csv(df=processed_df, path=s3_path, index=False)

    # Log to MLflow
    run_name = f"pipeline-inference-{cod_month}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="DataPull", nested=True):
            mlflow.log_input(
                mlflow.data.from_pandas(processed_df, s3_path),
                context="DataPull"
            )

    return s3_path, experiment_name, run_id
