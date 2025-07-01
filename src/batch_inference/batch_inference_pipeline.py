import os
from batch_inference.load import load_and_preprocess_oot

def test_inference_load():
    # Set this to an actual test month you have data for
    test_cod_month = 202406

    try:
        s3_path, experiment_name, run_id = load_and_preprocess_oot(cod_month=test_cod_month)
        print("\n✅ Inference load test completed successfully!")
        print(f"S3 Path: {s3_path}")
        print(f"Experiment: {experiment_name}")
        print(f"MLflow Run ID: {run_id}")
    except Exception as e:
        print("\n❌ Inference load test failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_inference_load()