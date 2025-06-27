import os
import pandas as pd
from sklearn.model_selection import train_test_split

class ChurnDataPipeline:
    def __init__(self, 
                 raw_data_path="data/raw", 
                 processed_data_path="data/processed/",
                 clientes_file="train_clientes_sample.csv",
                 reqs_file="train_requerimientos_sample.csv"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.clientes_file = clientes_file
        self.reqs_file = reqs_file

    def load_datasets(self, clientes_file: str, reqs_file: str) -> pd.DataFrame:
        print("Loading raw datasets...")
        clientes = pd.read_csv(os.path.join(self.raw_data_path, clientes_file))
        reqs = pd.read_csv(os.path.join(self.raw_data_path, reqs_file))

        print(f"Clientes shape: {clientes.shape}")
        print(f"Requerimientos shape: {reqs.shape}")

        return clientes, reqs

    def merge_datasets(self, clientes: pd.DataFrame, reqs: pd.DataFrame) -> pd.DataFrame:
        print("Merging datasets on ID_CORRELATIVO with aggregation...")

        # One-hot encode categorical columns
        categorical_cols = ["TIPO_REQUERIMIENTO2", "DICTAMEN", "PRODUCTO_SERVICIO_2", "SUBMOTIVO_2"]
        reqs_encoded = pd.get_dummies(reqs, columns=categorical_cols)

        # Group by ID_CORRELATIVO and sum one-hot encoded columns
        reqs_agg = reqs_encoded.groupby('ID_CORRELATIVO').sum()
        reqs_agg.reset_index(inplace=True)

        merged = clientes.merge(reqs_agg, on='ID_CORRELATIVO', how='left')
        merged = merged.fillna(0)

        print(f"Merged shape: {merged.shape}")
        return merged

    def split_and_save(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        print("Splitting into train/test...")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        os.makedirs(self.processed_data_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.processed_data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_data_path, "test.csv"), index=False)

        print("Saved train/test to data/processed/")

    def run(self):
        clientes, reqs = self.load_datasets(self.clientes_file, self.reqs_file)
        merged_df = self.merge_datasets(clientes, reqs)
        self.split_and_save(merged_df)

if __name__ == "__main__":
    pipeline = ChurnDataPipeline()
    pipeline.run()