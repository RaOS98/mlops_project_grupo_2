from steps.utils import TRACKING_SERVER_ARN, DEFAULT_PATH, SAGEMAKER_ROLE
from sagemaker.workflow.function_step import step

@step(
    name="DataPull",
    instance_type="ml.m5.large",
    image_uri="686410906112.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    role=SAGEMAKER_ROLE,
)
def data_pull(experiment_name: str, run_name: str, cod_month_start: int, cod_month_end: int) -> tuple[str, str, str]:
    import subprocess
    subprocess.run(['pip', 'install', 'awswrangler==3.9.1', 'mlflow', 'pandas'])

    import awswrangler as wr
    import pandas as pd
    import mlflow
    import os

    # ---- Preprocessing class ----
    class PreprocessingPipeline:
        def __init__(self, target_col: str = "ATTRITION", ref_path: str = "data/processed/train_clean.csv"):
            self.target_col = target_col
            self.ref_path = ref_path
            self.binary_flags = [
                "FLG_BANCARIZADO", "FLG_SEGURO", "FLG_NOMINA", "FLG_SDO_OTSSFF"
            ]
            self.categorical_columns = [
                "RANG_INGRESO", "RANG_SDO_PASIVO_MENOS0", "RANG_NRO_PRODUCTOS_MENOS0",
                "TIPO_REQUERIMIENTO2", "DICTAMEN", "PRODUCTO_SERVICIO_2", "SUBMOTIVO_2"
            ]

        def run(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
            df = df.copy()
            df = self._drop_unnecessary_columns(df)
            df = self._fill_binary_flags(df)
            df = self._map_lima_provincia(df)
            df = self._aggregate_time_series(df)
            df = self._encode_categoricals(df)
            df = df.fillna(0)
            if is_train:
                df = self._handle_target_column(df)
            else:
                df = self._align_with_train_columns(df)
            return df

        def _drop_unnecessary_columns(self, df): ...
        def _fill_binary_flags(self, df): ...
        def _map_lima_provincia(self, df): ...
        def _aggregate_time_series(self, df): ...
        def _encode_categoricals(self, df): ...
        def _handle_target_column(self, df): ...
        def _align_with_train_columns(self, df): ...

        # Implement all internal methods like before (omitted here for brevity)

    # ---- Pipeline logic ----
    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)
    TARGET_COL = "ATTRITION"

    clientes = wr.s3.read_csv("s3://my-batch-inference-data/oot_clientes_sample.csv")
    requerimientos = wr.s3.read_csv("s3://my-batch-inference-data/oot_requerimientos_sample.csv")

    df = clientes.merge(requerimientos, on="ID_CORRELATIVO", how="left")

    pipeline = PreprocessingPipeline(target_col=TARGET_COL)
    df_clean = pipeline.run(df, is_train=True)

    train_s3_path = f"{DEFAULT_PATH}/train_data/train.csv"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        with mlflow.start_run(run_name="DataPull", nested=True):
            df_clean.to_csv(train_s3_path, index=False)
            mlflow.log_input(
                mlflow.data.from_pandas(df_clean, train_s3_path, targets=TARGET_COL),
                context="DataPull"
            )

    return train_s3_path, experiment_name, run_id