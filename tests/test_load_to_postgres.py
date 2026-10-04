"""
Test de integración real contra PostgreSQL/PostGIS.

Se omite automáticamente si no hay una base de datos disponible (p. ej.
en un entorno local sin Docker). En la CI de GitHub Actions sí se
ejecuta, contra un servicio postgis real, validando que el esquema y la
carga funcionan de extremo a extremo.
"""
import pytest
from sqlalchemy import text

from etl.db import get_engine
from etl.load_to_postgres import load


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="No hay base de datos PostgreSQL disponible")


def test_load_inserts_expected_row_count():
    n_rows = load()
    assert n_rows > 0

    engine = get_engine()
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM fires")).scalar_one()
        assert count == n_rows


def test_geometry_column_is_populated_and_valid():
    load()
    engine = get_engine()
    with engine.connect() as conn:
        invalid = conn.execute(
            text("SELECT COUNT(*) FROM fires WHERE geom IS NULL OR NOT ST_IsValid(geom)")
        ).scalar_one()
        assert invalid == 0


def test_risk_view_returns_expected_categories():
    load()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT risk_level FROM v_fires_risk")).all()
        categories = {r[0] for r in rows}
        assert categories.issubset({"SIN_DANO", "BAJO", "MODERADO", "ALTO", "EXTREMO"})
