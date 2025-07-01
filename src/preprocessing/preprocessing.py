import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.preprocessing.steps.load import load_raw_data
from src.preprocessing.steps.merge import merge_raw_data
from src.preprocessing.steps.transform import PreprocessData
from src.preprocessing.steps.save import save_processed_data
from src.utils import EXPERIMENT_NAME
from src.preprocessing.config import RAW_PATHS


def main():
    print("STEP 1: Loading raw data...")
    raw_data = load_raw_data(raw_data_paths=RAW_PATHS, experiment_name=EXPERIMENT_NAME)

    clientes_df = raw_data["clientes"]["df"]
    requerimientos_df = raw_data["requerimientos"]["df"]

    print("STEP 2: Merging datasets...")
    merged_df = merge_raw_data(clientes_df, requerimientos_df)

    # Save intermediate merged version (must happen BEFORE transform step)
    temp_path = "data/temp/temp_merged.csv"
    merged_df.to_csv(temp_path, index=False)

    print("STEP 3: Running preprocessing...")
    preprocessor = PreprocessData(ref_path="data/processed/train_clean.csv")
    processed_df, is_train = preprocessor.run(temp_path, is_train=None)

    print("STEP 4: Saving processed data...")
    save_processed_data(processed_df, is_train)

    print("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()
