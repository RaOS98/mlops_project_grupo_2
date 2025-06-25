import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed/"

def load_datasets(clientes_file: str, reqs_file: str) -> pd.DataFrame:
    print("Loading raw datasets...")
    clientes = pd.read_csv(os.path.join(RAW_DATA_PATH, clientes_file))
    reqs = pd.read_csv(os.path.join(RAW_DATA_PATH, reqs_file))

    print(f"Clientes shape: {clientes.shape}")
    print(f"Requerimientos shape: {reqs.shape}")

    return clientes, reqs

def merge_datasets(clientes: pd.DataFrame, reqs: pd.DataFrame) -> pd.DataFrame:
    print("Merging datasets on ID_CORRELATIVO...")
    merged = clientes.merge(reqs, on="ID_CORRELATIVO", how="left")
    print(f"Merged shape: {merged.shape}")
    return merged

def split_and_save(df: pd.DataFrame, test_size=0.3, random_state=42):
    print("Splitting into train/test...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "test.csv"), index=False)

    print("Saved train/test to data/processed/")

def main():
    clientes, reqs = load_datasets("train_clientes_sample.csv", "train_requerimientos_sample.csv")
    merged_df = merge_datasets(clientes, reqs)
    split_and_save(merged_df)

if __name__ == "__main__":
    main()