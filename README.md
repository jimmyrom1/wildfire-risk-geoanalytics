# Wildfire Risk Geoanalytics · Python + PostGIS + Machine Learning

[![CI](https://github.com/jimmyrom1/wildfire-risk-geoanalytics/actions/workflows/ci.yml/badge.svg)](https://github.com/jimmyrom1/wildfire-risk-geoanalytics/actions/workflows/ci.yml)

Pipeline completo de analítica avanzada e IA sobre datos reales de incendios forestales: ingesta y limpieza con Python, modelado geoespacial en PostgreSQL/PostGIS, análisis exploratorio y un modelo de Machine Learning en dos etapas (ocurrencia + severidad) con validación rigurosa.

Dataset: ["Forest Fires"](https://archive.ics.uci.edu/dataset/162/forest+fires) (Cortez & Morais, 2007), incendios registrados en el Parque Natural de Montesinho, Portugal.

| Área | Figura |
| --- | --- |
| Correlación FWI/clima/área | ![Correlación](reports/figures/correlation_heatmap.png) |
| Localización aproximada | ![Mapa](reports/figures/geo_scatter.png) |
| Clasificador (matriz de confusión) | ![Confusion matrix](reports/figures/classifier_confusion_matrix.png) |
| Regresor: real vs. predicho | ![Real vs predicho](reports/figures/regressor_real_vs_predicted.png) |

## Stack

| Capa | Tecnología |
| --- | --- |
| Base de datos | PostgreSQL 16 + PostGIS 3.4 (geometrías, índices GIST, clustering espacial, consultas de proximidad) |
| ETL | Python 3.12, pandas, SQLAlchemy, psycopg2 |
| Análisis / EDA | pandas, matplotlib, seaborn |
| Machine Learning | scikit-learn (RandomForest), validación cruzada 5-fold, train/test split estratificado |
| Calidad | pytest (incluye tests de integración reales contra PostGIS) |
| Infraestructura | Docker Compose (PostgreSQL/PostGIS), GitHub Actions CI |

## Arrancar en un minuto

```bash
docker compose up -d                      # Levanta PostgreSQL + PostGIS y aplica sql/schema.sql
pip install -r requirements.txt

python -m etl.download_data               # Descarga el dataset de UCI
python -m etl.load_to_postgres            # Limpia, geocodifica y carga en PostGIS

python -m analysis.eda                    # Genera reports/figures/*.png y reports/metrics/eda_summary.json
python -m ml.train_model                  # Entrena y valida los modelos, genera métricas y figuras

psql postgresql://wildfire:wildfire@localhost:5432/wildfire -f sql/analysis_queries.sql
```

## Decisiones técnicas

- **Proyección aproximada de la rejilla a coordenadas reales.** El dataset original anonimiza la posición de cada incendio en una rejilla discreta `X,Y` (1-9, 1-9) dentro del mapa del parque. Para poder demostrar un flujo PostGIS real (índices GIST, clustering espacial, proximidad), esa rejilla se proyecta linealmente sobre el bounding box geográfico real del Parque Natural de Montesinho (`etl/transform.py::grid_to_latlon`). Esto se documenta explícitamente como una aproximación con fines demostrativos, no como georreferenciación oficial — una distinción importante en cualquier proyecto de I+D que use datos espaciales derivados.
- **Modelo en dos etapas (clasificación + regresión) en lugar de una regresión directa.** ~48% de los registros no presentan área quemada > 0. Entrenar una única regresión sobre todo el dataset está dominada por los ceros. Se separa el problema en: (1) `RandomForestClassifier` para predecir si habrá daño, y (2) `RandomForestRegressor` sobre `log1p(área)` solo para los casos con daño real. Es el enfoque estándar en la literatura de riesgo de incendios.
- **Esquema con restricciones e índices espaciales en la propia base de datos.** `sql/schema.sql` define `geometry(Point, 4326)` con índice `GIST`, además de vistas (`v_fires_risk`, `v_fires_by_season`) que encapsulan la lógica de negocio directamente en SQL, reutilizable tanto desde `psql` como desde Python.
- **Tests de integración reales, no solo mocks.** `tests/test_load_to_postgres.py` se ejecuta contra un PostGIS real en la CI (GitHub Actions, servicio `postgis/postgis:16-3.4`) y se omite automáticamente en local si no hay base de datos disponible, en vez de usar una base de datos simulada que podría ocultar errores reales de SQL/PostGIS.

## Hallazgos reales (investigación)

1. **El dataset tiene baja señal predictiva a partir solo del clima — y eso también es un resultado válido.** El regresor de severidad obtiene R² negativo en test (`reports/metrics/model_metrics.json`). Esto no es un fallo de implementación: es un resultado ya documentado en el paper original de Cortez & Morais (2007), que concluye que el área quemada depende fuertemente de factores no capturados por el clima (velocidad de intervención de bomberos, causa de ignición, recursos disponibles). Reportarlo honestamente, en vez de maquillar métricas, es parte del propio proceso de investigación.
2. **4 filas duplicadas exactas en el CSV de origen** (517 → 513 tras `drop_duplicates()`), un recordatorio de que ningún dataset "de libro" llega limpio.
3. **El índice GIST no siempre se usa** con solo 517 filas: el planificador de PostgreSQL prefiere `Seq Scan` porque la tabla cabe en una sola página. Documentado en `sql/analysis_queries.sql` junto con `SET enable_seqscan = off;` para forzar y comprobar el uso real del índice — un matiz de rendimiento que solo se descubre ejecutando `EXPLAIN ANALYZE`, no leyendo la documentación.

## Estructura

```
sql/
  schema.sql              # DDL: tabla fires, geometría, índices GIST, vistas de riesgo
  analysis_queries.sql    # Clustering espacial, proximidad, KNN, ranking, EXPLAIN ANALYZE
etl/
  download_data.py        # Descarga el CSV de UCI
  transform.py             # Limpieza + proyección de rejilla a lat/lon
  load_to_postgres.py      # Carga a PostGIS generando la geometría
analysis/
  eda.py                   # Análisis exploratorio, genera reports/figures/
ml/
  features.py              # Construcción de la matriz de features
  train_model.py           # Clasificador de ocurrencia + regresor de severidad
reports/
  figures/                 # Gráficas generadas (EDA + modelos)
  metrics/                 # eda_summary.json, model_metrics.json
tests/                     # pytest, incluye integración real contra PostGIS
```

## Tests

```bash
pytest -v
```

La CI ejecuta en cada push: aplica el esquema contra un PostGIS real, corre los tests (incluida la integración con base de datos), genera el EDA y entrena los modelos de extremo a extremo.
