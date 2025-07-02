import pandas as pd

def merge_raw_data(clientes_df: pd.DataFrame, requerimientos_df: pd.DataFrame) -> pd.DataFrame:
    print("Merging datasets on ID_CORRELATIVO with aggregation...")

    # One-hot encode categorical columns
    categorical_cols = ["TIPO_REQUERIMIENTO2", "DICTAMEN", "PRODUCTO_SERVICIO_2", "SUBMOTIVO_2"]
    reqs_encoded = pd.get_dummies(requerimientos_df, columns=categorical_cols)

    # Group by ID_CORRELATIVO and sum one-hot encoded columns
    reqs_agg = reqs_encoded.groupby("ID_CORRELATIVO").sum().reset_index()

    # Merge and fill NAs with 0
    merged = clientes_df.merge(reqs_agg, on="ID_CORRELATIVO", how="left").fillna(0)

    print(f"Merged shape: {merged.shape}")
    return merged
