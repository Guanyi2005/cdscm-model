
/* ===== session ===== */
SET client_encoding = 'UTF8';
SET client_min_messages TO WARNING;
SET work_mem = '512MB';
SET maintenance_work_mem = '1GB';
SET max_parallel_workers_per_gather = 4;
SET parallel_leader_participation = on;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname='cdscm') THEN
    RAISE EXCEPTION 'schema cdscm missing';
  END IF;
  PERFORM 1 FROM cdscm.v_hour_grid_t0; IF NOT FOUND THEN RAISE EXCEPTION 'run 61_02 first'; END IF;
END $$;

DO $$
DECLARE src_a TEXT; src_b TEXT; src_c TEXT; src_d TEXT; src_e TEXT; src_g TEXT;
BEGIN
  src_g := 'cdscm.v_hour_grid_t0';
  src_a := 'cdscm.v_a_map_hourly';
  src_b := 'cdscm.v_b_vaso_hourly';
  src_c := 'cdscm.v_c_urine_hourly';
  src_d := 'cdscm.v_d_creat_hourly';
  src_e := 'cdscm.v_e_rrt_hourly';

  EXECUTE format($sql$
    CREATE OR REPLACE VIEW cdscm._src_hour_grid AS SELECT * FROM %s;
    CREATE OR REPLACE VIEW cdscm._src_a_map      AS SELECT * FROM %s;
    CREATE OR REPLACE VIEW cdscm._src_b_vaso     AS SELECT * FROM %s;
    CREATE OR REPLACE VIEW cdscm._src_c_urine    AS SELECT * FROM %s;
    CREATE OR REPLACE VIEW cdscm._src_d_creat    AS SELECT * FROM %s;
    CREATE OR REPLACE VIEW cdscm._src_e_rrt      AS SELECT * FROM %s;
  $sql$, src_g, src_a, src_b, src_c, src_d, src_e);
END $$;

DO $$ BEGIN RAISE NOTICE '[61_03] materialize mv_hourly_wide'; END $$;
DROP MATERIALIZED VIEW IF EXISTS cdscm.mv_hourly_wide;
CREATE MATERIALIZED VIEW cdscm.mv_hourly_wide AS
SELECT
  g.stay_id, g.hour_from_t0, g.bin_start, g.bin_end,
  a.map_itemid, a.map_mmhg,
  b.vaso_rate_mcgkgmin, b.vaso_rate_unitshour,
  c.urine_ml,
  d.creat_itemid, d.creat_mgdl,
  e.rrt_on
FROM cdscm._src_hour_grid g
LEFT JOIN cdscm._src_a_map   a USING (stay_id, hour_from_t0, bin_start, bin_end)
LEFT JOIN cdscm._src_b_vaso  b USING (stay_id, hour_from_t0, bin_start, bin_end)
LEFT JOIN cdscm._src_c_urine c USING (stay_id, hour_from_t0, bin_start, bin_end)
LEFT JOIN cdscm._src_d_creat d USING (stay_id, hour_from_t0, bin_start, bin_end)
LEFT JOIN cdscm._src_e_rrt   e USING (stay_id, hour_from_t0, bin_start, bin_end);

CREATE INDEX IF NOT EXISTS mv_hourly_wide_stay_hour_idx ON cdscm.mv_hourly_wide(stay_id, hour_from_t0);
CREATE INDEX IF NOT EXISTS mv_hourly_wide_bin_end_idx   ON cdscm.mv_hourly_wide(bin_end);
CREATE INDEX IF NOT EXISTS mv_hourly_wide_map_idx       ON cdscm.mv_hourly_wide(map_mmhg) WHERE map_mmhg IS NOT NULL;
CREATE INDEX IF NOT EXISTS mv_hourly_wide_cate_idx      ON cdscm.mv_hourly_wide(vaso_rate_mcgkgmin) WHERE vaso_rate_mcgkgmin IS NOT NULL;
CREATE INDEX IF NOT EXISTS mv_hourly_wide_vp_idx        ON cdscm.mv_hourly_wide(vaso_rate_unitshour) WHERE vaso_rate_unitshour IS NOT NULL;
CREATE INDEX IF NOT EXISTS mv_hourly_wide_urine_idx     ON cdscm.mv_hourly_wide(urine_ml) WHERE urine_ml IS NOT NULL;
CREATE INDEX IF NOT EXISTS mv_hourly_wide_creat_idx     ON cdscm.mv_hourly_wide(creat_mgdl) WHERE creat_mgdl IS NOT NULL;
ANALYZE cdscm.mv_hourly_wide;

DO $$ BEGIN RAISE NOTICE '[61_03] recompute winsor cutoffs (map/vaso/urine_ml/creat)'; END $$;
DROP TABLE IF EXISTS cdscm.winsor_cutoffs;
CREATE TABLE cdscm.winsor_cutoffs (var_name TEXT PRIMARY KEY, p_low NUMERIC, p_high NUMERIC);
WITH cfg AS (SELECT winsor_low AS pl, winsor_high AS ph FROM cdscm.cfg_params LIMIT 1)
INSERT INTO cdscm.winsor_cutoffs
SELECT * FROM (
  SELECT 'map_mmhg'::text,
         PERCENTILE_CONT((SELECT pl FROM cfg)) WITHIN GROUP (ORDER BY map_mmhg),
         PERCENTILE_CONT((SELECT ph FROM cfg)) WITHIN GROUP (ORDER BY map_mmhg)
  FROM cdscm.mv_hourly_wide WHERE map_mmhg IS NOT NULL
  UNION ALL
  SELECT 'vaso_rate_mcgkgmin',
         PERCENTILE_CONT((SELECT pl FROM cfg)) WITHIN GROUP (ORDER BY vaso_rate_mcgkgmin),
         PERCENTILE_CONT((SELECT ph FROM cfg)) WITHIN GROUP (ORDER BY vaso_rate_mcgkgmin)
  FROM cdscm.mv_hourly_wide WHERE vaso_rate_mcgkgmin IS NOT NULL
  UNION ALL
  SELECT 'vaso_rate_unitshour',
         PERCENTILE_CONT((SELECT pl FROM cfg)) WITHIN GROUP (ORDER BY vaso_rate_unitshour),
         PERCENTILE_CONT((SELECT ph FROM cfg)) WITHIN GROUP (ORDER BY vaso_rate_unitshour)
  FROM cdscm.mv_hourly_wide WHERE vaso_rate_unitshour IS NOT NULL
  UNION ALL
  SELECT 'urine_ml',
         PERCENTILE_CONT((SELECT pl FROM cfg)) WITHIN GROUP (ORDER BY urine_ml),
         PERCENTILE_CONT((SELECT ph FROM cfg)) WITHIN GROUP (ORDER BY urine_ml)
  FROM cdscm.mv_hourly_wide WHERE urine_ml IS NOT NULL
  UNION ALL
  SELECT 'creat_mgdl',
         PERCENTILE_CONT((SELECT pl FROM cfg)) WITHIN GROUP (ORDER BY creat_mgdl),
         PERCENTILE_CONT((SELECT ph FROM cfg)) WITHIN GROUP (ORDER BY creat_mgdl)
  FROM cdscm.mv_hourly_wide WHERE creat_mgdl IS NOT NULL
) q;
ANALYZE cdscm.winsor_cutoffs;

DO $$ BEGIN RAISE NOTICE '[61_03] rebuild v_hourly_clean + ml/kg/h'; END $$;
CREATE OR REPLACE VIEW cdscm.v_hourly_clean AS
WITH wc AS (
  SELECT
    (SELECT p_low  FROM cdscm.winsor_cutoffs WHERE var_name='map_mmhg')            AS map_lo,
    (SELECT p_high FROM cdscm.winsor_cutoffs WHERE var_name='map_mmhg')            AS map_hi,
    (SELECT p_low  FROM cdscm.winsor_cutoffs WHERE var_name='vaso_rate_mcgkgmin')  AS cate_lo,
    (SELECT p_high FROM cdscm.winsor_cutoffs WHERE var_name='vaso_rate_mcgkgmin')  AS cate_hi,
    (SELECT p_low  FROM cdscm.winsor_cutoffs WHERE var_name='vaso_rate_unitshour') AS vp_lo,
    (SELECT p_high FROM cdscm.winsor_cutoffs WHERE var_name='vaso_rate_unitshour') AS vp_hi,
    (SELECT p_low  FROM cdscm.winsor_cutoffs WHERE var_name='urine_ml')            AS urine_lo,
    (SELECT p_high FROM cdscm.winsor_cutoffs WHERE var_name='urine_ml')            AS urine_hi,
    (SELECT p_low  FROM cdscm.winsor_cutoffs WHERE var_name='creat_mgdl')          AS creat_lo,
    (SELECT p_high FROM cdscm.winsor_cutoffs WHERE var_name='creat_mgdl')          AS creat_hi
),
joined AS (
  SELECT x.*, wt.weight_kg_baseline
  FROM cdscm.mv_hourly_wide x
  LEFT JOIN cdscm.v_weight_baseline_t0 wt USING (stay_id)
),
wins AS (
  SELECT
    stay_id, hour_from_t0, bin_start, bin_end,
    map_itemid,
    CASE WHEN map_mmhg IS NULL THEN NULL
         ELSE LEAST((SELECT map_hi FROM wc),  GREATEST((SELECT map_lo FROM wc),  map_mmhg)) END AS map_mmhg,
    CASE WHEN vaso_rate_mcgkgmin  IS NULL OR vaso_rate_mcgkgmin  < 0 THEN NULL
         ELSE LEAST((SELECT cate_hi FROM wc), GREATEST((SELECT cate_lo FROM wc), vaso_rate_mcgkgmin )) END AS vaso_rate_mcgkgmin,
    CASE WHEN vaso_rate_unitshour IS NULL OR vaso_rate_unitshour < 0 THEN NULL
         ELSE LEAST((SELECT vp_hi   FROM wc), GREATEST((SELECT vp_lo   FROM wc), vaso_rate_unitshour)) END AS vaso_rate_unitshour,
    CASE WHEN urine_ml IS NULL OR urine_ml < 0 THEN NULL
         ELSE LEAST((SELECT urine_hi FROM wc), GREATEST((SELECT urine_lo FROM wc), urine_ml)) END AS urine_ml,
    creat_itemid,
    CASE WHEN creat_mgdl IS NULL THEN NULL
         ELSE LEAST((SELECT creat_hi FROM wc), GREATEST((SELECT creat_lo FROM wc), creat_mgdl)) END AS creat_mgdl,
    rrt_on,
    weight_kg_baseline
  FROM joined
),
with_mlkgh AS (
  SELECT *,
         CASE
           WHEN urine_ml IS NULL OR weight_kg_baseline IS NULL OR weight_kg_baseline<=0 THEN NULL
           ELSE urine_ml / weight_kg_baseline
         END AS urine_mlkgh_raw
  FROM wins
),
mlkgh_cut AS (
  SELECT
    PERCENTILE_CONT((SELECT winsor_low  FROM cdscm.cfg_params LIMIT 1)) WITHIN GROUP (ORDER BY urine_mlkgh_raw) AS lo,
    PERCENTILE_CONT((SELECT winsor_high FROM cdscm.cfg_params LIMIT 1)) WITHIN GROUP (ORDER BY urine_mlkgh_raw) AS hi
  FROM with_mlkgh
  WHERE urine_mlkgh_raw IS NOT NULL
),
final AS (
  SELECT w.*,
         CASE
           WHEN w.urine_mlkgh_raw IS NULL THEN NULL
           ELSE LEAST((SELECT hi FROM mlkgh_cut), GREATEST((SELECT lo FROM mlkgh_cut), w.urine_mlkgh_raw))
         END AS urine_mlkgh
  FROM with_mlkgh w
)
SELECT
  stay_id, hour_from_t0, bin_start, bin_end,
  map_itemid, map_mmhg,
  vaso_rate_mcgkgmin, vaso_rate_unitshour,
  urine_ml, urine_mlkgh,
  creat_itemid, creat_mgdl,
  rrt_on,
  weight_kg_baseline
FROM final;

DO $$ BEGIN RAISE NOTICE '[61_03] refresh mv_hourly_clean_0_71'; END $$;
DROP MATERIALIZED VIEW IF EXISTS cdscm.mv_hourly_clean_0_71;
CREATE MATERIALIZED VIEW cdscm.mv_hourly_clean_0_71 AS
SELECT *
FROM cdscm.v_hourly_clean
WHERE hour_from_t0 BETWEEN 0 AND (SELECT obs_window_hours-1 FROM cdscm.cfg_params LIMIT 1);

CREATE INDEX IF NOT EXISTS mv_hourly_clean_stay_hour_idx ON cdscm.mv_hourly_clean_0_71(stay_id, hour_from_t0);
CREATE INDEX IF NOT EXISTS mv_hourly_clean_bin_end_idx   ON cdscm.mv_hourly_clean_0_71(bin_end);
ANALYZE cdscm.mv_hourly_clean_0_71;

DO $$ BEGIN RAISE NOTICE '[OK] 61_03 ready: mv_hourly_wide + v_hourly_clean + mv_hourly_clean_0_71'; END $$;
