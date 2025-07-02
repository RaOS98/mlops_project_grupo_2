from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, FunctionStep

from src.batch_inference.steps.load import load_and_preprocess_oot
from src.batch_inference.steps.inference import model_inference
from src.batch_inference.steps.push import data_push
from src.utils import PIPELINE_NAME, SAGEMAKER_SESSION, SAGEMAKER_ROLE


experiment_name = ParameterString(name="ExperimentName", default_value="batch-inference-experiment")

# Step 1: Load + Preprocess OOT data
data_pull_step = load_and_preprocess_oot(
    experiment_name=experiment_name,
        run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
)

# Step 2: Run model inference
inference_step = FunctionStep(
    name="ModelInference",
    step_func=model_inference,
    inputs={
        "inf_raw_s3_path": load_step.properties.Outputs["s3_path"],
        "experiment_name": load_step.properties.Outputs["experiment_name"],
        "run_id": load_step.properties.Outputs["run_id"]
    }
)

# Step 3: Post-process + save results
push_step = FunctionStep(
    name="DataPush",
    step_func=data_push,
    inputs={
        "inf_proc_s3_path": inference_step.properties.Outputs["s3_path"],
        "experiment_name": inference_step.properties.Outputs["experiment_name"],
        "run_id": inference_step.properties.Outputs["run_id"]
    }
)

# Build pipeline
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[experiment_name, run_name,
    steps=[load_step, inference_step, push_step],
    sagemaker_session=SAGEMAKER_SESSION,
    role=SAGEMAKER_ROLE
)

pipeline.upsert(role_arn=SAGEMAKER_ROLE)