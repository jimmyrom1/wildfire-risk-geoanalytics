-- =====================================================================
-- Consultas de análisis espacial y analítico avanzado
-- Ejecutar contra la base ya cargada (ver etl/load_to_postgres.py)
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1) CLUSTERING ESPACIAL (ST_ClusterKMeans)
-- Agrupa los incendios en 5 zonas de alta incidencia según su posición
-- geográfica, para poder priorizar vigilancia por zona en vez de por
-- punto individual.
-- ---------------------------------------------------------------------
SELECT
    cluster_id,
    COUNT(*)                                       AS n_incendios,
    ROUND(AVG(burned_area_ha)::numeric, 2)         AS area_media_ha,
    ROUND(SUM(burned_area_ha)::numeric, 2)         AS area_total_ha,
    ROUND(ST_X(ST_Centroid(ST_Collect(geom)))::numeric, 4) AS centroide_lon,
    ROUND(ST_Y(ST_Centroid(ST_Collect(geom)))::numeric, 4) AS centroide_lat
FROM (
    SELECT
        *,
        ST_ClusterKMeans(geom, 5) OVER () AS cluster_id
    FROM fires
) clustered
GROUP BY cluster_id
ORDER BY area_total_ha DESC;

-- ---------------------------------------------------------------------
-- 2) PROXIMIDAD A PUESTOS DE VIGILANCIA (ST_DWithin)
-- Incendios ocurridos a menos de 5 km de cada puesto de vigilancia
-- forestal. Útil para evaluar cobertura real de los puestos existentes.
-- ---------------------------------------------------------------------
SELECT
    rs.name                                        AS puesto_vigilancia,
    COUNT(f.id)                                    AS incendios_en_radio_5km,
    ROUND(AVG(f.burned_area_ha)::numeric, 2)       AS area_media_ha
FROM ranger_stations rs
JOIN fires f
    ON ST_DWithin(rs.geom::geography, f.geom::geography, 5000)  -- metros
GROUP BY rs.name
ORDER BY incendios_en_radio_5km DESC;

-- ---------------------------------------------------------------------
-- 3) VECINO MÁS CERCANO (KNN con operador <->)
-- Para cada puesto de vigilancia, los 3 incendios históricos más
-- cercanos y su distancia real en metros (ST_Distance sobre geography).
-- ---------------------------------------------------------------------
SELECT
    rs.name AS puesto_vigilancia,
    f.id    AS fire_id,
    f.month,
    f.burned_area_ha,
    ROUND(ST_Distance(rs.geom::geography, f.geom::geography)::numeric, 1) AS distancia_m
FROM ranger_stations rs
CROSS JOIN LATERAL (
    SELECT *
    FROM fires f
    ORDER BY f.geom <-> rs.geom
    LIMIT 3
) f
ORDER BY rs.name, distancia_m;

-- ---------------------------------------------------------------------
-- 4) RANKING DE MESES MÁS DESTRUCTIVOS (window functions)
-- Ranking por área total quemada y porcentaje acumulado sobre el total,
-- típico de un análisis de concentración (similar a un análisis 80/20).
-- ---------------------------------------------------------------------
WITH por_mes AS (
    SELECT
        month,
        SUM(burned_area_ha) AS area_total_ha
    FROM fires
    GROUP BY month
)
SELECT
    month,
    area_total_ha,
    RANK() OVER (ORDER BY area_total_ha DESC)                                   AS ranking,
    ROUND(100 * SUM(area_total_ha) OVER (ORDER BY area_total_ha DESC)
          / SUM(area_total_ha) OVER (), 1)                                      AS pct_acumulado
FROM por_mes
ORDER BY ranking;

-- ---------------------------------------------------------------------
-- 5) INCENDIOS DENTRO DE UN RADIO DE RIESGO CRÍTICO (ST_Buffer + ST_Within)
-- Genera un buffer de 3 km alrededor del puesto central y cuenta cuántos
-- incendios de riesgo ALTO/EXTREMO caen dentro de esa zona de influencia.
-- ---------------------------------------------------------------------
SELECT
    rs.name,
    COUNT(*) AS incendios_alto_riesgo_cercanos
FROM ranger_stations rs
JOIN v_fires_risk f
    ON ST_Within(
        f.geom,
        ST_Buffer(rs.geom::geography, 3000)::geometry
    )
WHERE rs.name = 'Puesto Central - Montesinho'
  AND f.risk_level IN ('ALTO', 'EXTREMO')
GROUP BY rs.name;

-- ---------------------------------------------------------------------
-- 6) PLAN DE EJECUCIÓN CON ÍNDICE GIST (demostración de rendimiento)
-- Comprobar que el índice espacial se usa realmente en el filtro por
-- proximidad (debe aparecer "Index Scan using idx_fires_geom").
--
-- NOTA: con solo 517 filas el planner de PostgreSQL normalmente elige
-- Seq Scan porque es más barato que usar el índice (la tabla cabe en
-- una sola página). Para forzar y comprobar que el índice SÍ se usa
-- correctamente cuando el volumen de datos lo justifica, ejecutar antes:
--   SET enable_seqscan = off;
-- Este es precisamente el tipo de comportamiento real de PostgreSQL que
-- conviene documentar en el informe técnico en vez de dar por hecho que
-- "tener un índice" implica "usarlo siempre".
-- ---------------------------------------------------------------------
EXPLAIN ANALYZE
SELECT *
FROM fires
WHERE ST_DWithin(
    geom::geography,
    ST_SetSRID(ST_MakePoint(-6.88, 41.86), 4326)::geography,
    5000
);
