import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Constants
TRAIN_PATH = "data/processed/train_clean.csv"
TEST_PATH = "data/processed/test_clean.csv"
TARGET = "ATTRITION"

# Model dictionary
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
}

def evaluate(y_true, y_pred, y_probs):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "f1": f1_score(y_true, y_pred)
    }

def train_and_log_model(name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        print(f"Training model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # Log metrics
        metrics = evaluate(y_test, y_pred, y_probs)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=f"{name}_model")

        print(f"Metrics for {name}: {metrics}")

def main():
    print("Loading data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train = df_train.drop(TARGET, axis=1)
    y_train = df_train[TARGET]
    X_test = df_test.drop(TARGET, axis=1)
    y_test = df_test[TARGET]

    mlflow.set_experiment("bank-attrition")

    for name, model in MODELS.items():
        train_and_log_model(name, model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()