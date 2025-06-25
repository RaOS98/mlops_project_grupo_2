import os
import pandas as pd
from src.features.client_features import preprocess_clientes_dataframe

INPUT_PATH = "data/processed/test.csv"
OUTPUT_PATH = "data/processed/test_clean.csv"

def main():
    print("Loading test dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Preprocessing test data...")
    df_clean = preprocess_clientes_dataframe(df, is_train=False)

    print(f"Saving cleaned test dataset to: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)

    print("Done.")

if __name__ == "__main__":
    main()