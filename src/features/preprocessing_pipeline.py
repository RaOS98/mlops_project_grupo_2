import os
import pandas as pd

class PreprocessingPipeline:
    def __init__(self, target_col: str = "ATTRITION", ref_path: str = "data/processed/train_clean.csv"):
        self.target_col = target_col
        self.ref_path = ref_path

        self.binary_flags = [
            "FLG_BANCARIZADO", "FLG_SEGURO", "FLG_NOMINA", "FLG_SDO_OTSSFF"
        ]

        self.categorical_columns = [
            "RANG_INGRESO",
            "RANG_SDO_PASIVO_MENOS0",
            "RANG_NRO_PRODUCTOS_MENOS0",
            "TIPO_REQUERIMIENTO2",
            "DICTAMEN",
            "PRODUCTO_SERVICIO_2",
            "SUBMOTIVO_2"
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

    def _drop_unnecessary_columns(self, df):
        df.drop(columns=[col for col in ["ID_CORRELATIVO", "CODMES"] if col in df.columns], inplace=True)
        return df

    def _fill_binary_flags(self, df):
        for col in self.binary_flags:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        return df

    def _map_lima_provincia(self, df):
        if "FLAG_LIMA_PROVINCIA" in df.columns:
            df["FLAG_LIMA_PROVINCIA"] = df["FLAG_LIMA_PROVINCIA"].map({"Lima": 1, "Provincia": 0}).fillna(0).astype(int)
        return df

    def _aggregate_time_series(self, df):
        sdo_cols = [col for col in df.columns if "SDO_ACTIVO" in col]
        canal_cols = [col for col in df.columns if "NRO_ACCES_CANAL" in col]
        ssff_cols = [col for col in df.columns if "NRO_ENTID_SSFF" in col]

        if sdo_cols:
            df["SDO_ACTIVO_PROM"] = df[sdo_cols].mean(axis=1)
            df.drop(columns=sdo_cols, inplace=True)
        if canal_cols:
            df["TOTAL_ACCESOS"] = df[canal_cols].sum(axis=1)
            df.drop(columns=canal_cols, inplace=True)
        if ssff_cols:
            df["NRO_ENTID_SSFF_PROM"] = df[ssff_cols].mean(axis=1)
            df.drop(columns=ssff_cols, inplace=True)

        return df

    def _encode_categoricals(self, df):
        existing_cat_cols = [col for col in self.categorical_columns if col in df.columns]
        return pd.get_dummies(df, columns=existing_cat_cols)

    def _handle_target_column(self, df):
        if self.target_col in df.columns:
            df[self.target_col] = df[self.target_col].astype(int)
            target = df.pop(self.target_col)
            df[self.target_col] = target
        return df

    def _align_with_train_columns(self, df):
        if os.path.exists(self.ref_path):
            ref_cols = pd.read_csv(self.ref_path, nrows=1).columns
            for col in ref_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[ref_cols]
        return df
