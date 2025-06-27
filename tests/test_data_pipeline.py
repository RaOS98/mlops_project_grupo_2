import os
import pandas as pd
from src.data.data_pipeline import ChurnDataPipeline

def test_split_and_merge_creates_expected_files(tmp_path):
    # Set up test pipeline with temp output dir
    pipeline = ChurnDataPipeline(
        raw_data_path="data/raw",
        processed_data_path=tmp_path,
        clientes_file="train_clientes_sample.csv",
        reqs_file="train_requerimientos_sample.csv"
    )
    
    pipeline.run()
    
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    
    assert train_file.exists(), "train.csv was not created"
    assert test_file.exists(), "test.csv was not created"
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    assert not df_train.empty, "train.csv is empty"
    assert not df_test.empty, "test.csv is empty"