import os
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlflow.exceptions import MlflowException

from src.features.preprocessing_pipeline import PreprocessingPipeline


class ModelTrainingPipeline:
    def __init__(
        self,
        train_path: str = "data/processed/train_clean.csv",
        test_path: str = "data/processed/test_clean.csv",
        target_col: str = "ATTRITION"
    ):
        self.train_path = train_path
        self.test_path = test_path
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

    def load_test_data(self):
        print("Loading test data...")
        df = pd.read_csv(self.test_path)
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y

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
                print(f"Logged {name} without evaluation — no test target available.")

            try:
                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=f"{name}_model")
            except MlflowException as e:
                if "already exists" in str(e):
                    print(f"Model '{name}_model' already exists. Logging only as artifact.")
                    mlflow.sklearn.log_model(model, artifact_path="model")
                else:
                    raise

    def run(self):
        mlflow.set_experiment("bank-attrition")

        X_train, y_train = self.load_train_data()
        X_test, y_test = self.load_test_data()

        for name, model in self.models.items():
            self.train_and_log_model(name, model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.run()