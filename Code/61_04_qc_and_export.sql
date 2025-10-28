/* ===== session ===== */
SET client_encoding='UTF8';
SET client_min_messages TO NOTICE;

/* ===== paths & files ===== */
\! cmd /c if not exist outputs\\61_data mkdir outputs\\61_data

/* ===== guards ===== */
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname='cdscm') THEN
    RAISE EXCEPTION 'schema cdscm not found';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_matviews
    WHERE schemaname='cdscm' AND matviewname='mv_hourly_clean_0_71'
  ) THEN
    RAISE EXCEPTION 'missing cdscm.mv_hourly_clean_0_71 (run 61_03 first)';
  END IF;
END $$;

/* ===== alias view (A..E; C uses ml/kg/h) ===== */
DO $$ BEGIN RAISE NOTICE '[61_04] create view cdscm.v_hourly_ABCD_for_exp'; END $$;
CREATE OR REPLACE VIEW cdscm.v_hourly_ABCD_for_exp AS
SELECT
  stay_id,
  hour_from_t0,
  bin_start,
  bin_end,
  map_mmhg            AS A_map_mmhg,          -- A (mmHg)
  vaso_rate_mcgkgmin  AS B_cate_mcgkgmin,     -- B1 (mcg/kg/min)
  vaso_rate_unitshour AS B_vp_unitshour,      -- B2 (units/hour)
  urine_mlkgh         AS C_urine_mlkgh,       -- C (mL/kg/h)
  creat_mgdl          AS D_creat_mgdl,        -- D (mg/dL)
  rrt_on              AS E_rrt_on             -- E (0/1)
FROM cdscm.mv_hourly_clean_0_71;

DO $$ BEGIN RAISE NOTICE '[61_04] create table cdscm.node_mapping'; END $$;
DROP TABLE IF EXISTS cdscm.node_mapping;
CREATE TABLE cdscm.node_mapping (
  node_letter    CHAR(1)  NOT NULL,
  alias_column   TEXT     NOT NULL,
  source_column  TEXT     NOT NULL,
  unit           TEXT     NOT NULL,
  source_itemids INT[]    NULL,
  PRIMARY KEY (node_letter, alias_column, source_column)
);

INSERT INTO cdscm.node_mapping (node_letter, alias_column, source_column, unit, source_itemids) VALUES
('A','A_map_mmhg',        'map_mmhg',            'mmHg',       ARRAY[220052,225312]),
('B','B_cate_mcgkgmin',   'vaso_rate_mcgkgmin',  'mcg/kg/min', NULL),
('B','B_vp_unitshour',    'vaso_rate_unitshour', 'units/hour', NULL),
('C','C_urine_mlkgh',     'urine_mlkgh',         'mL/kg/h',    ARRAY[226559,226560,226561,226563,226567]),
('D','D_creat_mgdl',      'creat_mgdl',          'mg/dL',      ARRAY[50912,52024,52546]),
('E','E_rrt_on',          'rrt_on',              '0/1',        ARRAY[224270,225441,225802,225805,225436]);

SELECT '[ROWS]' AS tag, COUNT(*) AS n FROM cdscm.v_hourly_ABCD_for_exp;
SELECT '[C_nonnull_rate]' AS tag,
       ROUND(100.0*SUM((C_urine_mlkgh IS NOT NULL)::int)/NULLIF(COUNT(*),0),2) AS pct
FROM cdscm.v_hourly_ABCD_for_exp;

DO $$ BEGIN RAISE NOTICE '[61_04] export CSVs -> outputs/61_data'; END $$;

\copy (SELECT stay_id, hour_from_t0, bin_start, bin_end, A_map_mmhg, B_cate_mcgkgmin, B_vp_unitshour, C_urine_mlkgh, D_creat_mgdl, E_rrt_on FROM cdscm.v_hourly_ABCD_for_exp ORDER BY stay_id, hour_from_t0) TO 'outputs/61_data/hourly_ABCD_exp.csv' CSV HEADER

\copy (SELECT node_letter, alias_column, source_column, unit, source_itemids FROM cdscm.node_mapping ORDER BY node_letter, alias_column) TO 'outputs/61_data/node_mapping.csv' CSV HEADER


/* ===== done ===== */
DO $$ BEGIN RAISE NOTICE '[OK] 61_04 exports done'; END $$;
