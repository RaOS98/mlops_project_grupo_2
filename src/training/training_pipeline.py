from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.execution_variables import ExecutionVariables

from src.training.steps.load import load_train_data
from src.utils import (
    SAGEMAKER_ROLE,
    PIPELINE_NAME_TRAINING,
    EXPERIMENT_NAME,
)

# Define the step
load_step = load_train_data(
    experiment_name=EXPERIMENT_NAME,
    run_name=ExecutionVariables.PIPELINE_EXECUTION_ID
)

print(f"Using pipeline name: {PIPELINE_NAME_TRAINING}")

pipeline = Pipeline(
    name=PIPELINE_NAME_TRAINING,
    parameters=[],
    steps=[load_step]
)

response = pipeline.upsert(role_arn=SAGEMAKER_ROLE)
print("✅ Pipeline upsert response:", response)

execution = pipeline.start()
print("✅ Pipeline execution started!")
print("Execution ARN:", execution.arn)
