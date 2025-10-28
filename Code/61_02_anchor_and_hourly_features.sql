/* ===== session ===== */
SET client_encoding = 'UTF8';
SET client_min_messages TO WARNING;

/* ===== guards ===== */
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname='cdscm') THEN
    RAISE EXCEPTION 'schema cdscm missing';
  END IF;
  PERFORM 1 FROM cdscm.a_map_stg;  IF NOT FOUND THEN RAISE EXCEPTION 'missing a_map_stg'; END IF;
  PERFORM 1 FROM cdscm.b_vaso_stg; IF NOT FOUND THEN RAISE EXCEPTION 'missing b_vaso_stg'; END IF;
  PERFORM 1 FROM cdscm.c_urine_stg;IF NOT FOUND THEN RAISE EXCEPTION 'missing c_urine_stg'; END IF;
  PERFORM 1 FROM cdscm.d_creat_stg;IF NOT FOUND THEN RAISE EXCEPTION 'missing d_creat_stg'; END IF;
  PERFORM 1 FROM cdscm.e_rrt_stg;  IF NOT FOUND THEN RAISE EXCEPTION 'missing e_rrt_stg'; END IF;
END $$;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_anchor_t0'; END $$;
CREATE OR REPLACE VIEW cdscm.v_anchor_t0 AS
WITH cfg AS (
  SELECT t0_map_low_threshold_mmhg AS thr,
         t0_map_low_min_duration_min AS min_dur_min
  FROM cdscm.cfg_params LIMIT 1
),
raw AS (
  SELECT s.stay_id, s.start_time AS ts, s.value_raw AS map_mmhg
  FROM cdscm.a_map_stg s
),
flagged AS (
  SELECT r.*, (r.map_mmhg < (SELECT thr FROM cfg)) AS is_low
  FROM raw r
),
seg AS (
  SELECT f.*,
         SUM(CASE WHEN is_low THEN 0 ELSE 1 END)
           OVER (PARTITION BY stay_id ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS grp
  FROM flagged f
),
low_runs AS (
  SELECT stay_id, MIN(ts) AS run_start, MAX(ts) AS run_end,
         EXTRACT(EPOCH FROM (MAX(ts)-MIN(ts)))/60.0 AS dur_min
  FROM seg
  WHERE is_low
  GROUP BY stay_id, grp
),
qualified AS (
  SELECT stay_id, run_start AS t0_ts
  FROM low_runs, cfg
  WHERE dur_min >= cfg.min_dur_min
),
picked AS (
  SELECT DISTINCT ON (stay_id) stay_id, t0_ts
  FROM qualified
  ORDER BY stay_id, t0_ts
)
SELECT * FROM picked;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_hour_grid_t0'; END $$;
CREATE OR REPLACE VIEW cdscm.v_hour_grid_t0 AS
WITH cfg AS (SELECT obs_window_hours AS H FROM cdscm.cfg_params LIMIT 1)
SELECT
  a.stay_id,
  gs.hour_from_t0,
  (a.t0_ts + (gs.hour_from_t0    )*INTERVAL '1 hour') AS bin_start,
  (a.t0_ts + (gs.hour_from_t0 + 1)*INTERVAL '1 hour') AS bin_end
FROM cdscm.v_anchor_t0 a
JOIN LATERAL (SELECT generate_series(0,(SELECT H FROM cfg)-1) AS hour_from_t0) gs ON TRUE;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_a_map_hourly'; END $$;
CREATE OR REPLACE VIEW cdscm.v_a_map_hourly AS
SELECT
  h.stay_id, h.hour_from_t0, h.bin_start, h.bin_end,
  a.itemid AS map_itemid, a.map_mmhg, 'mmhg'::text AS uom
FROM cdscm.v_hour_grid_t0 h
LEFT JOIN LATERAL (
  SELECT s.itemid, s.start_time, s.value_raw::numeric AS map_mmhg
  FROM cdscm.a_map_stg s
  WHERE s.stay_id = h.stay_id
    AND s.start_time >= (SELECT t0_ts FROM cdscm.v_anchor_t0 t WHERE t.stay_id=h.stay_id)
    AND s.start_time <  h.bin_end
  ORDER BY s.start_time DESC
  LIMIT 1
) a ON TRUE;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_b_vaso_hourly'; END $$;
CREATE OR REPLACE VIEW cdscm.v_b_vaso_hourly AS
WITH base AS (SELECT * FROM cdscm.v_hour_grid_t0),
cate AS (
  SELECT
    g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end,
    SUM( GREATEST(0, EXTRACT(EPOCH FROM (LEAST(b.end_time,g.bin_end) - GREATEST(b.start_time,g.bin_start)))) * b.value_raw )/3600.0
      AS vaso_rate_mcgkgmin
  FROM base g
  JOIN cdscm.b_vaso_stg b
    ON b.stay_id=g.stay_id
   AND b.unit_raw='mcg/kg/min'
   AND tstzrange(b.start_time,b.end_time,'[)') && tstzrange(g.bin_start,g.bin_end,'[)')
  GROUP BY g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end
),
vp AS (
  SELECT
    g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end,
    SUM( GREATEST(0, EXTRACT(EPOCH FROM (LEAST(b.end_time,g.bin_end) - GREATEST(b.start_time,g.bin_start)))) * b.value_raw )/3600.0
      AS vaso_rate_unitshour
  FROM base g
  JOIN cdscm.b_vaso_stg b
    ON b.stay_id=g.stay_id
   AND b.unit_raw='units/hour'
   AND tstzrange(b.start_time,b.end_time,'[)') && tstzrange(g.bin_start,g.bin_end,'[)')
  GROUP BY g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end
)
SELECT
  g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end,
  cate.vaso_rate_mcgkgmin, vp.vaso_rate_unitshour
FROM base g
LEFT JOIN cate ON cate.stay_id=g.stay_id AND cate.hour_from_t0=g.hour_from_t0
LEFT JOIN vp   ON vp.stay_id=g.stay_id   AND vp.hour_from_t0=g.hour_from_t0;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_c_urine_hourly'; END $$;
CREATE OR REPLACE VIEW cdscm.v_c_urine_hourly AS
SELECT
  h.stay_id, h.hour_from_t0, h.bin_start, h.bin_end,
  SUM(u.value_raw)::numeric AS urine_ml, 'ml'::text AS uom
FROM cdscm.v_hour_grid_t0 h
LEFT JOIN cdscm.c_urine_stg u
  ON u.stay_id=h.stay_id
 AND u.start_time>=h.bin_start
 AND u.start_time< h.bin_end
GROUP BY h.stay_id, h.hour_from_t0, h.bin_start, h.bin_end;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_d_creat_hourly'; END $$;
CREATE OR REPLACE VIEW cdscm.v_d_creat_hourly AS
SELECT
  h.stay_id, h.hour_from_t0, h.bin_start, h.bin_end,
  d.itemid AS creat_itemid, d.creat_mgdl, 'mg/dl'::text AS uom
FROM cdscm.v_hour_grid_t0 h
LEFT JOIN LATERAL (
  SELECT s.itemid, s.start_time, s.value_raw::numeric AS creat_mgdl
  FROM cdscm.d_creat_stg s
  WHERE s.stay_id=h.stay_id
    AND s.start_time >= (SELECT t0_ts FROM cdscm.v_anchor_t0 t WHERE t.stay_id=h.stay_id)
    AND s.start_time <  h.bin_end
  ORDER BY s.start_time DESC
  LIMIT 1
) d ON TRUE;

DO $$ BEGIN RAISE NOTICE '[61_02] build v_e_rrt_hourly'; END $$;
CREATE OR REPLACE VIEW cdscm.v_e_rrt_hourly AS
WITH hit AS (
  SELECT g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end, 1 AS rrt_on
  FROM cdscm.v_hour_grid_t0 g
  JOIN cdscm.e_rrt_stg e
    ON e.stay_id=g.stay_id
   AND tstzrange(e.start_time,e.end_time,'[)') && tstzrange(g.bin_start,g.bin_end,'[)')
  GROUP BY g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end
)
SELECT g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end,
       COALESCE(h.rrt_on,0) AS rrt_on
FROM cdscm.v_hour_grid_t0 g
LEFT JOIN hit h USING (stay_id, hour_from_t0, bin_start, bin_end);

DO $$ BEGIN RAISE NOTICE '[61_02] build v_weight_baseline_t0'; END $$;
CREATE OR REPLACE VIEW cdscm.v_weight_baseline_t0 AS
WITH cand AS (
  SELECT
    w.stay_id, t.t0_ts,
    w.start_time AS weight_time,
    w.weight_kg
  FROM cdscm.weight_stg w
  JOIN cdscm.v_anchor_t0 t USING (stay_id)
  WHERE w.weight_kg IS NOT NULL
    AND w.start_time BETWEEN t.t0_ts - INTERVAL '24 hours' AND t.t0_ts + INTERVAL '24 hours'
),
scored AS (
  SELECT *,
         ABS(EXTRACT(EPOCH FROM (weight_time - t0_ts))) AS abs_sec,
         (weight_time <= t0_ts) AS is_prior
  FROM cand
),
picked AS (
  SELECT DISTINCT ON (stay_id)
         stay_id, t0_ts, weight_time, weight_kg AS weight_kg_baseline
  FROM scored
  ORDER BY stay_id, abs_sec ASC, is_prior DESC, weight_time DESC
)
SELECT * FROM picked;

DO $$ BEGIN RAISE NOTICE '[OK] 61_02 ready'; END $$;
