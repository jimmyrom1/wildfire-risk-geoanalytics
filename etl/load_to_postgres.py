"""
Carga el dataset limpio en PostgreSQL/PostGIS, generando la geometría de
punto (geom) a partir de la rejilla transformada a lat/lon.

Requiere que la base de datos ya exista con el esquema aplicado
(ver sql/schema.sql, aplicado automáticamente por docker-compose).

Uso:
    python -m etl.load_to_postgres
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from etl.db import get_engine
from etl.transform import clean_fires_df

RAW_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "forestfires.csv"

INSERT_SQL = text(
    """
    INSERT INTO fires (
        x_grid, y_grid, month, day_of_week, ffmc, dmc, dc, isi,
        temp_c, rh_pct, wind_kmh, rain_mm, burned_area_ha, geom
    ) VALUES (
        :x_grid, :y_grid, :month, :day_of_week, :ffmc, :dmc, :dc, :isi,
        :temp_c, :rh_pct, :wind_kmh, :rain_mm, :burned_area_ha,
        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
    )
    """
)


def load(csv_path: Path = RAW_CSV_PATH, truncate: bool = True) -> int:
    raw = pd.read_csv(csv_path)
    clean = clean_fires_df(raw)

    engine = get_engine()
    with engine.begin() as conn:
        if truncate:
            conn.execute(text("TRUNCATE TABLE fires RESTART IDENTITY"))
        conn.execute(INSERT_SQL, clean.to_dict(orient="records"))

    return len(clean)


if __name__ == "__main__":
    n_rows = load()
    print(f"Cargadas {n_rows} filas en la tabla 'fires'.")
