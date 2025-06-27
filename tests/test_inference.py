import pytest
import pandas as pd
from src.inference.predict import ModelPredictor

TEST_CLEAN_PATH = "data/processed/test_clean.csv"
TRAIN_CLEAN_PATH = "data/processed/train_clean.csv"

@pytest.fixture
def predictor():
    return ModelPredictor(model_name="xgboost_model", model_version=1)

def test_data_integrity():
    df_test = pd.read_csv(TEST_CLEAN_PATH)
    df_train = pd.read_csv(TRAIN_CLEAN_PATH)
    assert not df_test.empty, "Test data should not be empty"
    assert not df_train.empty, "Train data should not be empty"
    assert "ID_CORRELATIVO" in df_test.columns, "'ID_CORRELATIVO' should be present in test data"

def test_process_alignment():
    df_test = pd.read_csv(TEST_CLEAN_PATH)
    df_train = pd.read_csv(TRAIN_CLEAN_PATH)
    test_cols = set(df_test.columns)
    train_cols = set(df_train.columns)
    assert test_cols == train_cols, "Mismatch between test and train columns"

def test_predict_output_shape(predictor):
    df_test = pd.read_csv(TEST_CLEAN_PATH)
    preds = predictor.predict(df_test)
    assert len(preds) == len(df_test), "Prediction output length must match input rows"
