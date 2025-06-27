from src.data.data_pipeline import ChurnDataPipeline
from src.features.preprocessing_pipeline import PreprocessingPipeline
from training.train_model import ModelTrainingPipeline
from batch_inference.batch_inference import ModelInferencePipeline
import pandas as pd

def main():
    print("\n[1] Running data pipeline...")
    data_pipeline = ChurnDataPipeline()
    data_pipeline.run()

    print("\n[2] Preprocessing train and test sets...")
    preprocessor = PreprocessingPipeline()
    
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_clean = preprocessor.run(train_df, is_train=True)
    test_clean = preprocessor.run(test_df, is_train=False)

    train_clean.to_csv("data/processed/train_clean.csv", index=False)
    test_clean.to_csv("data/processed/test_clean.csv", index=False)

    print("\n[3] Training models and evaluating on OOT...")
    trainer = ModelTrainingPipeline()
    trainer.run()

    print("\n[4] Running inference pipeline...")
    inference = ModelInferencePipeline()
    inference.run()

if __name__ == "__main__":
    main()