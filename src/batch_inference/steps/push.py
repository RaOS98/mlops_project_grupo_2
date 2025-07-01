from sagemaker.workflow.function_step import step
from src.utils import INSTANCE_TYPE, IMAGE_URI, SAGEMAKER_ROLE, MLFLOW_URI, DEFAULT_PATH
from src.batch_inference.config import MODEL_NAME

@step(
    name="DataPush",
    instance_type=INSTANCE_TYPE,
    image_uri=IMAGE_URI,
    role=SAGEMAKER_ROLE
)
def data_push(inf_proc_s3_path: str, experiment_name: str, run_id: str, codmes: int):
    import subprocess
    subprocess.run(['pip', 'install', 'awswrangler==3.12.0'])

    import pandas as pd
    import numpy as np
    import mlflow
    from datetime import datetime
    import pytz
    import awswrangler as wr

    # Constants
    ID_COL = "transaction_id"   # change if your ID column differs
    TIME_COL = "codmes"         # matches the codmes parameter
    PRED_COL = "prob"
    PARTITION_COL = TIME_COL

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    # Load inference output
    print(f"Reading inference results from {inf_proc_s3_path}")
    df = pd.read_csv(inf_proc_s3_path)

    # Risk bucketing
    df['fraud_profile'] = np.where(df[PRED_COL] >= 0.415, 'High risk',
                            np.where(df[PRED_COL] >= 0.285, 'Medium risk', 'Low risk'))

    # Add metadata
    df['model'] = MODEL_NAME
    timezone = pytz.timezone("America/Lima")
    df['load_date'] = datetime.now(timezone).strftime("%Y%m%d")
    df['order'] = df[PRED_COL].rank(method='first', ascending=False).astype(int)
    df[TIME_COL] = codmes

    # Select and reorder final columns
    df = df[[ID_COL, PRED_COL, 'model', 'fraud_profile', 'load_date', 'order', TIME_COL]]

    # Write to S3 as Parquet with partitioning
    output_base = f"s3://{DEFAULT_PATH}/inf-posproc-data"
    output_path = f"{output_base}/{PARTITION_COL}={codmes}/output.parquet"

    print(f"Saving post-processed data to {output_path}")
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')

    # Optional: create external table in Athena
    database = "mlops"
    table_name = f"{database}.utec-bank-project"

    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (
        {ID_COL} string,
        {PRED_COL} double,
        model string,
        fraud_profile string,
        load_date string,
        order int
    )
    PARTITIONED BY ({PARTITION_COL} int)
    STORED AS parquet
    LOCATION '{output_base}'
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """

    print(f"Creating Athena table: {table_name}")
    query_exec_id = wr.athena.start_query_execution(sql=ddl, database=database)
    wr.athena.wait_query(query_execution_id=query_exec_id)

    # Register partition
    print(f"Repairing Athena table partition: {table_name}")
    dml = f"MSCK REPAIR TABLE {table_name}"
    query_exec_id = wr.athena.start_query_execution(sql=dml, database=database)
    wr.athena.wait_query(query_execution_id=query_exec_id)

    # Log to MLflow
    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="DataPush", nested=True):
            mlflow.log_input(
                mlflow.data.from_pandas(df, output_path),
                context="DataPush"
            )

    print("Data push completed successfully.")
