from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep

from src.training.steps.load import load_train_data
from src.training.steps.train import train_model
from src.training.steps.evaluate import evaluate
from src.training.steps.register import register_model
from src.utils import MODEL_NAME, PIPELINE_NAME_TRAINING, SAGEMAKER_ROLE, ENV_CODE, USERNAME

# MLflow experiment name
experiment_name = f"pipeline-train-{ENV_CODE}-{USERNAME}"

# Pipeline parameters (e.g., for filtering raw data, if needed)
# cod_month_start = ParameterInteger(name="PeriodoCargaInicio")
# cod_month_end = ParameterInteger(name="PeriodoCargaFin")

# Step 1: Load training data
data_pull_step = load_train_data(
    experiment_name=experiment_name,
    run_name=ExecutionVariables.PIPELINE_EXECUTION_ID#,
    # cod_month_start=cod_month_start,
    # cod_month_end=cod_month_end,
)

# Step 2: Train model
model_training_step = train_model(
    train_s3_path=data_pull_step[0],
    experiment_name=data_pull_step[1],
    run_id=data_pull_step[2],
)

# Step 3: Conditional evaluation and registration
model_evaluation_step = evaluate(
    test_s3_path=model_training_step[0],
    experiment_name=model_training_step[1],
    run_id=model_training_step[2],
    training_run_id=model_training_step[3],
)

conditional_register_step = ConditionStep(
    name="ConditionalRegister",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=model_evaluation_step["f1_score"],
            right=0.6,
        )
    ],
    if_steps=[
        register_model(
            model_name=MODEL_NAME,
            experiment_name=model_training_step[1],
            run_id=model_training_step[2],
            training_run_id=model_training_step[3],
        )
    ],
    else_steps=[
        FailStep(
            name="Fail",
            error_message="Model performance is not good enough."
        )
    ],
)

# Final pipeline object
pipeline = Pipeline(
    name=PIPELINE_NAME_TRAINING,
    steps=[data_pull_step, model_training_step, conditional_register_step]#,
    # parameters=[cod_month_start, cod_month_end],
)

pipeline.upsert(role_arn=SAGEMAKER_ROLE)
