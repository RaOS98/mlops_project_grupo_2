import os
import argparse
import mlflow
from src.training.train_pipeline import ModelTrainingPipeline

def main():
    # Parse any SageMaker environment variables or custom arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mlflow_tracking_uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--train_path", type=str, default="data/processed/train_clean.csv")
    parser.add_argument("--test_path", type=str, default="data/processed/test_clean.csv")
    
    args = parser.parse_args()

    # Set MLflow tracking URI from env var (passed in SageMaker training job)
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    print(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")
    print(f"Training with train_path: {args.train_path}, test_path: {args.test_path}")

    # Run the pipeline
    pipeline = ModelTrainingPipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        target_col="ATTRITION"
    )
    pipeline.run()

if __name__ == "__main__":
    main()
