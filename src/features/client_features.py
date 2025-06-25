import os
import pandas as pd
import numpy as np

def preprocess_clientes_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans and transforms the clientes dataset.
    Parameters:
        df (pd.DataFrame): input dataframe after merging clientes + requerimientos
        is_train (bool): whether this is training data (contains target)
    Returns:
        pd.DataFrame: cleaned and transformed
    """

    df = df.copy()  # avoid mutating original

    # Drop raw ID if not needed
    if "ID_CORRELATIVO" in df.columns:
        df.drop("ID_CORRELATIVO", axis=1, inplace=True)

    # Drop partition marker
    if "CODMES" in df.columns:
        df.drop("CODMES", axis=1, inplace=True)

    # Target column should be last
    target_col = "ATTRITION"

    # Fill NA for binary flags
    binary_flags = [
        "FLG_BANCARIZADO", "FLG_SEGURO", "FLG_NOMINA", "FLG_SDO_OTSSFF"
    ]
    for col in binary_flags:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Special mapping for "Lima" / "Provincia"
    if "FLAG_LIMA_PROVINCIA" in df.columns:
        df["FLAG_LIMA_PROVINCIA"] = df["FLAG_LIMA_PROVINCIA"].map({
            "Lima": 1, "Provincia": 0
        }).fillna(0).astype(int)

    # Handle time series variables like SDO_ACTIVO_menos0 to menos5
    sdo_cols = [col for col in df.columns if "SDO_ACTIVO" in col]
    canal_cols = [col for col in df.columns if "NRO_ACCES_CANAL" in col]
    ssff_cols = [col for col in df.columns if "NRO_ENTID_SSFF" in col]

    # Optional: aggregate those as mean or sum
    if sdo_cols:
        df["SDO_ACTIVO_PROM"] = df[sdo_cols].mean(axis=1)
        df.drop(columns=sdo_cols, inplace=True)
    if canal_cols:
        df["TOTAL_ACCESOS"] = df[canal_cols].sum(axis=1)
        df.drop(columns=canal_cols, inplace=True)
    if ssff_cols:
        df["NRO_ENTID_SSFF_PROM"] = df[ssff_cols].mean(axis=1)
        df.drop(columns=ssff_cols, inplace=True)

    # One-hot encode ranked categoricals
    cat_cols = [
    "RANG_INGRESO",
    "RANG_SDO_PASIVO_MENOS0",
    "RANG_NRO_PRODUCTOS_MENOS0",
    "TIPO_REQUERIMIENTO2",
    "DICTAMEN",
    "PRODUCTO_SERVICIO_2",
    "SUBMOTIVO_2"
    ]
    cat_cols = [col for col in cat_cols if col in df.columns]
    df = pd.get_dummies(df, columns=cat_cols)

    # Handle remaining nulls (safe default)
    df = df.fillna(0)

    # Separate target if in training mode
    if is_train and target_col in df.columns:
        # Make sure it's binary int
        df[target_col] = df[target_col].astype(int)
        # Move it to the end
        target = df.pop(target_col)
        df[target_col] = target

    # Optional: align columns for inference
    if not is_train:
        # Load train columns if they exist
        ref_path = "data/processed/train_clean.csv"
        if os.path.exists(ref_path):
            ref_cols = pd.read_csv(ref_path, nrows=1).columns
            for col in ref_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[ref_cols]

    return df
