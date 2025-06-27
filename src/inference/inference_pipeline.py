import os
import pandas as pd
import mlflow
import mlflow.pyfunc
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import awswrangler as wr
import boto3

from src.models.train_pipeline import ModelTrainingPipeline


class ModelInferencePipeline:
    def __init__(
        self,
        output_path: str = "data/output/predictions.csv",
        s3_output_path: str = "s3://my-batch-inference-data/predictions.csv",
        target_metric: str = "roc_auc",
        s3_clientes_path: str = "s3://my-batch-inference-data/oot_clientes_sample.csv",
        s3_reqs_path: str = "s3://my-batch-inference-data/oot_requerimientos_sample.csv"
    ):
        self.output_path = output_path
        self.s3_output_path = s3_output_path
        self.target_metric = target_metric
        self.s3_clientes_path = s3_clientes_path
        self.s3_reqs_path = s3_reqs_path

        self.training_pipeline = ModelTrainingPipeline()
        self.model_name, self.model_version = self._get_best_model()
        self.model = self._load_model_from_registry()

    def _get_best_model(self):
        print("Searching for best model in MLflow...")
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("bank-attrition")
        if experiment is None:
            raise Exception("Experiment 'bank-attrition' not found.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=[f"metrics.{self.target_metric} DESC"]
        )

        if not runs:
            raise Exception("No valid model runs found in MLflow.")

        best_run = runs[0]
        model_name = best_run.data.tags.get("mlflow.runName") + "_model"
        model_version = client.get_latest_versions(model_name, stages=["None"])[-1].version

        print(f"Selected model: {model_name} (version {model_version}) with {self.target_metric} = {best_run.data.metrics.get(self.target_metric)}")
        return model_name, model_version

    def _load_model_from_registry(self):
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        print(f"Loading model: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    def run(self):
        print("Downloading OOT data from S3...")
        clientes = wr.s3.read_csv(self.s3_clientes_path)
        reqs = wr.s3.read_csv(self.s3_reqs_path)

        df_merged = clientes.merge(reqs, on="ID_CORRELATIVO", how="left")
        df_processed, y_true = self.training_pipeline.preprocess_oot_dataframe(df_merged)

        print("Generating predictions...")
        predictions = self.model.predict(df_processed)

        if y_true is not None:
            print("\nEvaluation Metrics:")
            print(" - Accuracy:", accuracy_score(y_true, predictions))
            print(" - F1 Score:", f1_score(y_true, predictions))
            print(" - ROC AUC:", roc_auc_score(y_true, predictions))

        output = df_merged[["ID_CORRELATIVO"]].copy()
        output["prediction"] = predictions

        # Save locally
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        output.to_csv(self.output_path, index=False)
        print(f"\nPredictions saved locally to: {self.output_path}")

        # Save to S3
        wr.s3.to_csv(df=output, path=self.s3_output_path, index=False)
        print(f"Predictions uploaded to S3 at: {self.s3_output_path}")


if __name__ == "__main__":
    pipeline = ModelInferencePipeline(
        s3_clientes_path="s3://my-batch-inference-data/oot_clientes_sample.csv",
        s3_reqs_path="s3://my-batch-inference-data/oot_requerimientos_sample.csv",
        s3_output_path="s3://my-batch-inference-data/predictions.csv"
    )
    pipeline.run()
