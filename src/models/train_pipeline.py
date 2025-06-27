import os
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.features.preprocessing_pipeline import PreprocessingPipeline


class ModelTrainingPipeline:
    def __init__(
        self,
        train_path: str = "data/processed/train_clean.csv",
        clientes_oot_path: str = "data/raw/oot_clientes_sample.csv",
        reqs_oot_path: str = "data/raw/oot_requerimientos_sample.csv",
        target_col: str = "ATTRITION"
    ):
        self.train_path = train_path
        self.clientes_oot_path = clientes_oot_path
        self.reqs_oot_path = reqs_oot_path
        self.target_col = target_col
        self.models: Dict[str, object] = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "xgboost": XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
        }
        self.preprocessor = PreprocessingPipeline(target_col=target_col)

    def load_train_data(self):
        print("Loading training data...")
        df = pd.read_csv(self.train_path)
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y

    def load_and_preprocess_oot_data(self):
        print("Loading and preprocessing OOT data...")
        clientes = pd.read_csv(self.clientes_oot_path)
        reqs = pd.read_csv(self.reqs_oot_path)
        df = clientes.merge(reqs, on="ID_CORRELATIVO", how="left")
        y = df[self.target_col] if self.target_col in df.columns else None
        df_processed = self.preprocessor.run(df, is_train=False)
        if self.target_col in df_processed.columns:
            df_processed = df_processed.drop(columns=[self.target_col])
        return df_processed, y

    def evaluate_model(self, y_true, y_pred, y_probs):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_probs)
        }

    def train_and_log_model(
        self, name: str, model, 
        X_train: pd.DataFrame, y_train: pd.Series, 
        X_test: pd.DataFrame, y_test: Optional[pd.Series]
    ):
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            mlflow.log_params(model.get_params())

            if y_test is not None:
                y_pred = model.predict(X_test)
                y_probs = model.predict_proba(X_test)[:, 1]
                metrics = self.evaluate_model(y_test, y_pred, y_probs)
                mlflow.log_metrics(metrics)
                print(f"Logged {name} with metrics: {metrics}")
            else:
                print(f"Logged {name} without evaluation — no OOT target available.")

            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=f"{name}_model")

    def run(self):
        mlflow.set_experiment("bank-attrition")

        X_train, y_train = self.load_train_data()
        X_oot, y_oot = self.load_and_preprocess_oot_data()

        for name, model in self.models.items():
            self.train_and_log_model(name, model, X_train, y_train, X_oot, y_oot)


if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.run()
