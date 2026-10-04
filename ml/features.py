"""Construcción de la matriz de features a partir del dataset limpio."""
import pandas as pd

NUMERIC_FEATURES = ["ffmc", "dmc", "dc", "isi", "temp_c", "rh_pct", "wind_kmh", "rain_mm"]
CATEGORICAL_FEATURES = ["month", "day_of_week"]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X_numeric = df[NUMERIC_FEATURES].reset_index(drop=True)
    X_categorical = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES).reset_index(drop=True)
    return pd.concat([X_numeric, X_categorical], axis=1)
