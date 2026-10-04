"""Configuración de conexión a la base de datos, vía variables de entorno."""
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_DSN = "postgresql+psycopg2://wildfire:wildfire@localhost:5432/wildfire"


def get_engine() -> Engine:
    dsn = os.environ.get("WILDFIRE_DB_DSN", DEFAULT_DSN)
    return create_engine(dsn)
