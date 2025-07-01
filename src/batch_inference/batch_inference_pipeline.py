from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, FunctionStep

from src.batch_inference.steps.load import load_and_preprocess_oot
from src.batch_inference.steps.inference import model_inference
from src.batch_inference.steps.push import data_push
from src.utils import PIPELINE_NAME, SAGEMAKER_SESSION, SAGEMAKER_ROLE

def get_batch_inference_pipeline() -> Pipeline:
    # Parameters
    experiment_name = ParameterString(name="ExperimentName", default_value="batch-inference-experiment")
    run_name = ParameterString(name="RunName", default_value="oot-run")
    codmes = ParameterInteger(name="PeriodoCarga", default_value=202403)

    # Step 1: Load + Preprocess OOT data
    load_step = FunctionStep(
        name="DataPull",
        step_func=load_and_preprocess_oot,
        inputs={
            "experiment_name": experiment_name,
            "run_name": run_name
        }
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
            "run_id": inference_step.properties.Outputs["run_id"],
            "codmes": codmes
        }
    )

    # Build pipeline
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[experiment_name, run_name, codmes],
        steps=[load_step, inference_step, push_step],
        sagemaker_session=SAGEMAKER_SESSION,
        role=SAGEMAKER_ROLE
    )

    return pipeline
