import os
from src.batch_inference.steps.load import load_and_preprocess_oot
from src.utils import EXPERIMENT_NAME
from datetime import datetime

def test_inference_load():
    # Generate a unique run name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"load_oot_{timestamp}"

    try:
        s3_path, experiment_name, run_id = load_and_preprocess_oot(
            experiment_name=EXPERIMENT_NAME,
            run_name=run_name
        )
        print("\n✅ Inference load test completed successfully!")
        print(f"S3 Path: {s3_path}")
        print(f"Experiment: {experiment_name}")
        print(f"MLflow Run ID: {run_id}")
    except Exception as e:
        print("\n❌ Inference load test failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_inference_load()