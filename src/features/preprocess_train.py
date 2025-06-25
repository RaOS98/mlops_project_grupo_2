import os
import pandas as pd
from src.features.client_features import preprocess_clientes_dataframe

INPUT_PATH = "data/processed/train.csv"
OUTPUT_PATH = "data/processed/train_clean.csv"

def main():
    print("Loading training dataset...")
    df = pd.read_csv(INPUT_PATH)
    
    print("Preprocessing training data...")
    df_clean = preprocess_clientes_dataframe(df, is_train=True)

    print(f"Saving cleaned train dataset to: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)

    print("Done.")

if __name__ == "__main__":
    main()