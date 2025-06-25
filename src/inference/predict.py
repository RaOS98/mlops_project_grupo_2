import os
import pandas as pd
import mlflow.pyfunc
from src.features.client_features import preprocess_clientes_dataframe


class ModelPredictor:
    def __init__(self, model_name: str, model_version: int):
        self.model_name = model_name
        self.model_version = model_version
        self.model = self.load_model_from_registry()

    def load_model_from_registry(self):
        print(f"Loading model '{self.model_name}' version '{self.model_version}'...")
        model_uri = f"models:/{self.model_name}/{self.model_version}"
        return mlflow.pyfunc.load_model(model_uri)

    def load_and_merge_data(self, clientes_path: str, reqs_path: str) -> pd.DataFrame:
        print("Loading OOT data...")
        clientes = pd.read_csv(clientes_path)
        reqs = pd.read_csv(reqs_path)
        merged = clientes.merge(reqs, on="ID_CORRELATIVO", how="left")
        return merged

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Preprocessing data for inference...")
        return preprocess_clientes_dataframe(df, is_train=False)

    def predict(self, df_processed: pd.DataFrame) -> pd.Series:
        print("Making predictions...")
        return self.model.predict(df_processed)

    def save_predictions(self, original_df: pd.DataFrame, predictions, output_path: str):
        print("Saving predictions...")
        output = original_df[["ID_CORRELATIVO"]].copy()
        output["prediction"] = predictions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")


def main():
    CLIENTES_FILE = "data/raw/oot_clientes_sample.csv"
    REQS_FILE = "data/raw/oot_requerimientos_sample.csv"
    OUTPUT_FILE = "data/output/predictions.csv"

    predictor = ModelPredictor(model_name="xgboost_model", model_version=1)

    # Step 1: Load merged raw data
    df_oot = predictor.load_and_merge_data(CLIENTES_FILE, REQS_FILE)

    # Step 2: Extract target if available
    y_true = None
    if "ATTRITION" in df_oot.columns:
        y_true = df_oot["ATTRITION"].copy()

    # Step 3: Preprocess for inference (drops ATTRITION)
    df_processed = predictor.preprocess(df_oot)

    # Force drop ATTRITION if still present
    if "ATTRITION" in df_processed.columns:
        df_processed = df_processed.drop(columns=["ATTRITION"])

    # Step 4: Predict
    preds = predictor.predict(df_processed)

    # Step 5: Optionally evaluate
    if y_true is not None:
        print("Evaluation Metrics:")
        print(" - Accuracy:", accuracy_score(y_true, preds))
        print(" - F1 Score:", f1_score(y_true, preds))
        print(" - ROC AUC:", roc_auc_score(y_true, preds))

    # Step 6: Save predictions
    predictor.save_predictions(df_oot, preds, OUTPUT_FILE)

if __name__ == "__main__":
    main()
