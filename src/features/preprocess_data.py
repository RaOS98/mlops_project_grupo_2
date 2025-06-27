import os
import pandas as pd
from preprocessing_pipeline import PreprocessingPipeline

def preprocess_and_save(input_path: str, output_path: str, is_train: bool):
    stage = "training" if is_train else "test"
    print(f"Loading {stage} dataset...")
    df = pd.read_csv(input_path)

    print(f"Preprocessing {stage} data...")
    pipeline = PreprocessingPipeline()
    df_clean = pipeline.run(df, is_train=is_train)

    print(f"Saving cleaned {stage} dataset to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print("Done.")

if __name__ == "__main__":
    preprocess_and_save("data/processed/train.csv", "data/processed/train_clean.csv", is_train=True)
    preprocess_and_save("data/processed/test.csv", "data/processed/test_clean.csv", is_train=False)
