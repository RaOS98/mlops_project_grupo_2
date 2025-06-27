from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString

from sagemaker.session import Session
from sagemaker import get_execution_role

# Parameters
train_path_param = ParameterString(name="TrainPath", default_value="s3://my-bucket/train.csv")
test_path_param = ParameterString(name="TestPath", default_value="s3://my-bucket/test.csv")

# Estimator
estimator = SKLearn(
    entry_point="train.py",
    source_dir="src/training",
    role=get_execution_role(),
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    environment={
        "MLFLOW_TRACKING_URI": "<your-tracking-uri>",
        "PYTHONPATH": "/opt/ml/code/.."
    }
)

# Training step
train_step = TrainingStep(
    name="ModelTraining",
    estimator=estimator,
    inputs={
        "train": train_path_param,
        "test": test_path_param
    }
)

# Pipeline
pipeline = Pipeline(
    name="MyTrainingPipeline",
    parameters=[train_path_param, test_path_param],
    steps=[train_step]
)
