from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep

from steps.data_pull import data_pull
from steps.model_training import train
from steps.model_evaluation import evaluate
from steps.model_registration import register
from steps.utils import PIPELINE_NAME, MODEL_NAME, USERNAME, ENV_CODE, SAGEMAKER_ROLE

# MLflow experiment name
experiment_name = f"pipeline-train-{ENV_CODE}-{USERNAME}"

# Parameter definitions
cod_month_start = ParameterInteger(name="PeriodoCargaInicio", default_value=202101)
cod_month_end = ParameterInteger(name="PeriodoCargaFin", default_value=202112)

# Data Pull Step
data_pull_step = data_pull(
    experiment_name=experiment_name,
    run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    cod_month_start=cod_month_start,
    cod_month_end=cod_month_end
)

# Model Training Step
model_training_step = train(
    train_s3_path=data_pull_step[0],
    experiment_name=data_pull_step[1],
    run_id=data_pull_step[2]
)

# Conditional Model Evaluation and Registration Step
conditional_register_step = ConditionStep(
    name="ConditionalRegister",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=evaluate(
                test_s3_path=model_training_step[0],
                experiment_name=model_training_step[1],
                run_id=model_training_step[2],
                training_run_id=model_training_step[3],
            )["f1_score"],
            right=0.6
        )
    ],
    if_steps=[
        register(
            model_name=MODEL_NAME,
            experiment_name=model_training_step[1],
            run_id=model_training_step[2],
            training_run_id=model_training_step[3],
        )
    ],
    else_steps=[
        FailStep(
            name="Fail",
            error_message="Model performance is not good enough to register."
        )
    ]
)

# Final Pipeline Definition
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[
        data_pull_step,
        model_training_step,
        conditional_register_step
    ],
    parameters=[
        cod_month_start,
        cod_month_end
    ]
)

# Register pipeline to SageMaker
if __name__ == "__main__":
    pipeline.upsert(role_arn=SAGEMAKER_ROLE)
    execution = pipeline.start()
    print("Pipeline execution started...")
    execution.wait()
    print("Pipeline execution completed.")