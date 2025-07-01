import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import mlflow
import mlflow.data
from src.preprocessing.steps.load import load_raw_data
from src.preprocessing.steps.merge import merge_raw_data
from src.preprocessing.steps.transform import PreprocessData
from src.preprocessing.steps.save import save_processed_data
from src.utils import EXPERIMENT_NAME
from src.preprocessing.config import RAW_PATHS


def detect_data_source(raw_paths: dict) -> str:
    for path in raw_paths.values():
        if "oot" in path.lower():
            return "oot"
    return "train"

def main():
    # Setup MLflow
    mlflow.set_tracking_uri("http://18.219.91.131:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_source = detect_data_source(RAW_PATHS)
    run_name = f"preprocessing_{data_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        print("STEP 1: Loading raw data...")
        raw_data = load_raw_data(RAW_PATHS)
        clientes_df = raw_data["clientes"]["df"]
        requerimientos_df = raw_data["requerimientos"]["df"]

        # Log inputs
        mlflow.log_input(
            mlflow.data.from_pandas(clientes_df, source=raw_data["clientes"]["path"]),
            context="raw_clientes"
        )
        mlflow.log_input(
            mlflow.data.from_pandas(requerimientos_df, source=raw_data["requerimientos"]["path"]),
            context="raw_requerimientos"
        )

        mlflow.log_metric("clientes_rows", clientes_df.shape[0])
        mlflow.log_metric("requerimientos_rows", requerimientos_df.shape[0])

        print("STEP 2: Merging datasets...")
        merged_df = merge_raw_data(clientes_df, requerimientos_df)

        # Save intermediate merged version (must happen BEFORE transform step)
        temp_path = "data/temp/temp_merged.csv"
        merged_df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path, artifact_path="intermediate")

        print("STEP 3: Running preprocessing...")
        preprocessor = PreprocessData(ref_path="data/processed/train_clean.csv")
        processed_df, is_train = preprocessor.run(temp_path, is_train=None)

        print("STEP 4: Saving processed data...")
        save_path = save_processed_data(processed_df, is_train)
        mlflow.log_artifact(save_path, artifact_path="processed")

        mlflow.log_metric("processed_rows", processed_df.shape[0])
        mlflow.set_tag("is_train", is_train)

        print("Preprocessing pipeline complete.")

if __name__ == "__main__":
    main()
