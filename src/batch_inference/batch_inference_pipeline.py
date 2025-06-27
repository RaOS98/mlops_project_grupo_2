from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
import os

from batch_inference_utils import (  # create this file separately with constants
    PIPELINE_NAME,
    SAGEMAKER_ROLE,
    IMAGE_URI,
    MODEL_NAME,
    INSTANCE_TYPE,
    INSTANCE_COUNT,
    BUCKET,
    REGION
)

pipeline_session = PipelineSession()

# ---- Pipeline Parameter ----
cod_month = ParameterInteger(name="PeriodoCarga", default_value=202406)  # e.g., June 2024

# ---- Step 1: Pull OOT data ----
data_pull_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=SAGEMAKER_ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    base_job_name="data-pull",
    sagemaker_session=pipeline_session
)

data_pull_step = ProcessingStep(
    name="DataPullStep",
    processor=data_pull_processor,
    code="src/steps/data_pull.py",
    job_arguments=["--month", cod_month],
    outputs=[
        {
            "OutputName": "inference_input",
            "S3Output": Join(on="/", values=["s3:/", BUCKET, "inference", "input", ExecutionVariables.PIPELINE_EXECUTION_ID])
        }
    ]
)

# ---- Step 2: Run inference ----
model_transformer = Transformer(
    model_name=MODEL_NAME,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    output_path=Join(on="/", values=["s3:/", BUCKET, "inference", "output", ExecutionVariables.PIPELINE_EXECUTION_ID]),
    strategy="SingleRecord",
    assemble_with="Line"
)

model_inference_step = TransformStep(
    name="ModelInferenceStep",
    transformer=model_transformer,
    inputs=data_pull_step.properties.ProcessingOutputConfig.Outputs["inference_input"].S3Output.S3Uri
)

# ---- Step 3: Push results ----
data_push_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=SAGEMAKER_ROLE,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    base_job_name="data-push",
    sagemaker_session=pipeline_session
)

data_push_step = ProcessingStep(
    name="DataPushStep",
    processor=data_push_processor,
    code="src/steps/data_push.py",
    job_arguments=["--month", cod_month],
    inputs=[model_inference_step.output]
)

# ---- Pipeline Assembly ----
pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[cod_month],
    steps=[data_pull_step, model_inference_step, data_push_step],
    sagemaker_session=pipeline_session
)