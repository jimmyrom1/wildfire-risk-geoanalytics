"""
Funciones de limpieza y transformación del dataset "Forest Fires".

La rejilla original X,Y (1-9, 1-9) anonimiza la posición real de cada
incendio dentro del Parque Natural de Montesinho. Para poder construir
un flujo PostGIS real y demostrable se proyecta esa rejilla, de forma
APROXIMADA, sobre el bounding box geográfico real del parque. Esto es
una simplificación documentada con fines educativos/demostrativos, no
una georreferenciación oficial.
"""
from __future__ import annotations

import pandas as pd

# Bounding box aproximado del Parque Natural de Montesinho (Portugal)
LAT_MIN, LAT_MAX = 41.78, 41.95
LON_MIN, LON_MAX = -7.05, -6.75

COLUMN_MAP = {
    "X": "x_grid",
    "Y": "y_grid",
    "month": "month",
    "day": "day_of_week",
    "FFMC": "ffmc",
    "DMC": "dmc",
    "DC": "dc",
    "ISI": "isi",
    "temp": "temp_c",
    "RH": "rh_pct",
    "wind": "wind_kmh",
    "rain": "rain_mm",
    "area": "burned_area_ha",
}


def grid_to_latlon(x_grid: pd.Series, y_grid: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Mapea la rejilla discreta (1-9, 1-9) al bbox real del parque."""
    lon = LON_MIN + (x_grid - 1) / 8 * (LON_MAX - LON_MIN)
    lat = LAT_MIN + (y_grid - 1) / 8 * (LAT_MAX - LAT_MIN)
    return lat, lon


def clean_fires_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Limpia y transforma el CSV crudo de UCI al esquema interno del proyecto."""
    df = raw.rename(columns=COLUMN_MAP).copy()

    expected_cols = set(COLUMN_MAP.values())
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el CSV de origen: {missing}")

    df = df.drop_duplicates()
    df = df.dropna(subset=list(expected_cols))

    for col in ["x_grid", "y_grid"]:
        df[col] = df[col].astype(int)

    numeric_cols = ["ffmc", "dmc", "dc", "isi", "temp_c", "rh_pct", "wind_kmh", "rain_mm", "burned_area_ha"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    df["month"] = df["month"].str.lower().str.strip()
    df["day_of_week"] = df["day_of_week"].str.lower().str.strip()

    df["lat"], df["lon"] = grid_to_latlon(df["x_grid"], df["y_grid"])

    return df.reset_index(drop=True)
