-- =====================================================================
-- Esquema: wildfire-risk-geoanalytics
-- Fuente de datos: UCI Machine Learning Repository - "Forest Fires"
--                   (Cortez & Morais, 2007), Parque Natural de Montesinho,
--                   Trás-os-Montes, Portugal.
--
-- NOTA METODOLÓGICA (ver también docs/documento explicativo):
-- El dataset original anonimiza la ubicación real de cada incendio en una
-- rejilla discreta X,Y (1-9). Para poder demostrar un flujo de trabajo
-- geoespacial real con PostGIS, esa rejilla se proyecta de forma
-- aproximada sobre el bounding box real del Parque Natural de Montesinho
-- (lat ~41.78-41.95 N, lon ~-7.05 a -6.75 W). Las coordenadas resultantes
-- son una APROXIMACIÓN para fines demostrativos/educativos, no la
-- localización exacta de cada incendio histórico.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------
-- Tabla principal: registros de incendios con variables meteorológicas,
-- índices FWI (Fire Weather Index canadiense) y geometría de punto.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fires (
    id              SERIAL PRIMARY KEY,
    x_grid          SMALLINT NOT NULL CHECK (x_grid BETWEEN 1 AND 9),
    y_grid          SMALLINT NOT NULL CHECK (y_grid BETWEEN 1 AND 9),
    month           VARCHAR(3) NOT NULL,
    day_of_week     VARCHAR(3) NOT NULL,
    ffmc            NUMERIC(5,1) NOT NULL,  -- Fine Fuel Moisture Code
    dmc             NUMERIC(6,1) NOT NULL,  -- Duff Moisture Code
    dc              NUMERIC(6,1) NOT NULL,  -- Drought Code
    isi             NUMERIC(5,1) NOT NULL,  -- Initial Spread Index
    temp_c          NUMERIC(4,1) NOT NULL,
    rh_pct          NUMERIC(5,1) NOT NULL,
    wind_kmh        NUMERIC(4,1) NOT NULL,
    rain_mm         NUMERIC(5,1) NOT NULL,
    burned_area_ha  NUMERIC(8,2) NOT NULL,
    geom            geometry(Point, 4326) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fires_geom ON fires USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_fires_month ON fires (month);
CREATE INDEX IF NOT EXISTS idx_fires_area ON fires (burned_area_ha);

-- ---------------------------------------------------------------------
-- Puestos de vigilancia forestal (sintéticos, dentro del bbox del
-- parque) para demostrar consultas de proximidad (ST_DWithin / ST_Distance).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ranger_stations (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    geom        geometry(Point, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ranger_geom ON ranger_stations USING GIST (geom);

INSERT INTO ranger_stations (name, geom) VALUES
    ('Puesto Norte - Rio de Onor',   ST_SetSRID(ST_MakePoint(-6.90, 41.93), 4326)),
    ('Puesto Central - Montesinho',  ST_SetSRID(ST_MakePoint(-6.88, 41.86), 4326)),
    ('Puesto Sur - Gimonde',         ST_SetSRID(ST_MakePoint(-6.80, 41.80), 4326))
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------
-- Vista: categoría de riesgo derivada del área quemada histórica.
-- Umbrales basados en clasificación habitual de severidad de incendios
-- (0 ha = sin daño; <1 ha = conato; 1-10 ha = moderado; >10 ha = severo).
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_fires_risk AS
SELECT
    f.*,
    CASE
        WHEN burned_area_ha = 0        THEN 'SIN_DANO'
        WHEN burned_area_ha < 1        THEN 'BAJO'
        WHEN burned_area_ha < 10       THEN 'MODERADO'
        WHEN burned_area_ha < 100      THEN 'ALTO'
        ELSE 'EXTREMO'
    END AS risk_level
FROM fires f;

-- ---------------------------------------------------------------------
-- Vista: agregados por estación del año (derivada del mes) para
-- analítica de negocio rápida (usada también desde Python/EDA).
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_fires_by_season AS
SELECT
    CASE
        WHEN month IN ('dec','jan','feb') THEN 'invierno'
        WHEN month IN ('mar','apr','may') THEN 'primavera'
        WHEN month IN ('jun','jul','aug') THEN 'verano'
        ELSE 'otono'
    END AS season,
    COUNT(*)                         AS n_incendios,
    ROUND(AVG(burned_area_ha), 2)    AS area_media_ha,
    ROUND(SUM(burned_area_ha), 2)    AS area_total_ha,
    ROUND(AVG(temp_c), 1)            AS temp_media_c,
    ROUND(AVG(wind_kmh), 1)          AS viento_medio_kmh
FROM fires
GROUP BY 1
ORDER BY area_total_ha DESC;
