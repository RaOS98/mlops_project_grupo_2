import pytest
import pandas as pd
from src.models.train import ModelTrainingPipeline

@pytest.fixture
def pipeline():
    return ModelTrainingPipeline()

def test_load_data_structure(pipeline):
    train_df, test_df = pipeline.load_data()
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert "ATTRITION" in train_df.columns
    assert "ATTRITION" in test_df.columns

def test_data_split_dimensions(pipeline):
    train_df, test_df = pipeline.load_data()
    X_train, y_train, X_test, y_test = pipeline.split_features_target(train_df, test_df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "ATTRITION" not in X_train.columns
    assert "ATTRITION" not in X_test.columns

def test_model_training_runs(pipeline):
    train_df, test_df = pipeline.load_data()
    X_train, y_train, X_test, y_test = pipeline.split_features_target(train_df, test_df)
    model = pipeline.models["xgboost"]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
