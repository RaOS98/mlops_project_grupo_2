from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session
from sagemaker import get_execution_role

# Setup
sagemaker_session = Session()
role = get_execution_role()

# Parameters
train_path_param = ParameterString(name="TrainPath", default_value="s3://my-bucket/train.csv")
test_path_param = ParameterString(name="TestPath", default_value="s3://my-bucket/test.csv")

# Estimator
estimator = SKLearn(
    entry_point="train.py",
    source_dir="src/training",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    environment={
        "MLFLOW_TRACKING_URI": "s3://your-s3-bucket-or-tracking-url",
        "PYTHONPATH": "/opt/ml/code/.."
    },
    sagemaker_session=sagemaker_session
)

# Training Step
train_step = TrainingStep(
    name="ModelTraining",
    estimator=estimator,
    inputs={
        "train": train_path_param,
        "test": test_path_param
    }
)

# Pipeline Definition
pipeline = Pipeline(
    name="MyTrainingPipeline",
    parameters=[train_path_param, test_path_param],
    steps=[train_step],
    sagemaker_session=sagemaker_session
)

# (Optional) Create/Update Pipeline
pipeline.upsert(role_arn=role)

# (Optional) Execute Pipeline Run
# pipeline.start()