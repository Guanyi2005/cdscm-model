
/* ===== session ===== */
SET client_encoding = 'UTF8';
SET client_min_messages TO WARNING;

/* ===== paths & files ===== */


/* ===== schema ===== */
CREATE SCHEMA IF NOT EXISTS cdscm;

/* ===== params ===== */

DROP VIEW IF EXISTS cdscm._icu_window;

DROP TABLE IF EXISTS cdscm.cfg_params;
CREATE TABLE cdscm.cfg_params (
  obs_window_hours            INTEGER  NOT NULL,   -- grid length (hours)
  t0_map_low_threshold_mmhg   NUMERIC  NOT NULL,   -- anchor low-MAP threshold
  t0_map_low_min_duration_min INTEGER  NOT NULL,   -- contiguous minutes for t0
  map_range_low               NUMERIC  NOT NULL,   -- MAP sanity
  map_range_high              NUMERIC  NOT NULL,
  urine_event_max_ml          NUMERIC  NOT NULL,   -- per-event cap
  creat_range_low             NUMERIC  NOT NULL,   -- creat range (mg/dL)
  creat_range_high            NUMERIC  NOT NULL,
  winsor_low                  NUMERIC  NOT NULL,   -- percentile for winsor
  winsor_high                 NUMERIC  NOT NULL
);
INSERT INTO cdscm.cfg_params VALUES
(72, 65, 10, 20, 200, 3000, 0.2, 20, 0.01, 0.99);
ANALYZE cdscm.cfg_params;

/* ===== whitelists ===== */
DROP TABLE IF EXISTS cdscm.itemid_whitelists;
CREATE TABLE cdscm.itemid_whitelists (node TEXT NOT NULL, itemid INTEGER NOT NULL, label TEXT NOT NULL);

INSERT INTO cdscm.itemid_whitelists VALUES
 ('A_MAP',220052,'mean arterial pressure'),
 ('A_MAP',225312,'art mean');

INSERT INTO cdscm.itemid_whitelists VALUES
 ('B_VASO',221289,'epinephrine'),
 ('B_VASO',221653,'dobutamine'),
 ('B_VASO',221662,'dopamine'),
 ('B_VASO',221749,'phenylephrine'),
 ('B_VASO',221906,'norepinephrine'),
 ('B_VASO',222315,'vasopressin'),
 ('B_VASO',229617,'epinephrine.'),
 ('B_VASO',229630,'phenylephrine (50/250)'),
 ('B_VASO',229631,'phenylephrine (200/250)_old_1'),
 ('B_VASO',229632,'phenylephrine (200/250)'),
 ('B_VASO',229789,'phenylephrine (intubation)');


INSERT INTO cdscm.itemid_whitelists VALUES
 ('C_URINE',226559,'Urine Out Foley'),
 ('C_URINE',226560,'Urine Out Void'),
 ('C_URINE',226561,'Urine Out Condom Cath'),
 ('C_URINE',226563,'Urine Out Other'),
 ('C_URINE',226567,'Urine Out I&O');

INSERT INTO cdscm.itemid_whitelists VALUES
 ('D_CREAT',50912,'creatinine (chemistry blood)'),
 ('D_CREAT',52024,'creatinine, whole blood (blood gas)'),
 ('D_CREAT',52546,'creatininine (chemistry blood)');

INSERT INTO cdscm.itemid_whitelists VALUES
 ('E_RRT',224270,'dialysis catheter'),
 ('E_RRT',225441,'hemodialysis'),
 ('E_RRT',225802,'dialysis - crrt'),
 ('E_RRT',225805,'peritoneal dialysis'),
 ('E_RRT',225436,'crrt filter change');

DROP TABLE IF EXISTS cdscm.weight_itemids;
CREATE TABLE cdscm.weight_itemids (itemid INT PRIMARY KEY, label TEXT, unit TEXT);
INSERT INTO cdscm.weight_itemids VALUES
  (224639,'Daily Weight','kg'),
  (226512,'Admission Weight (Kg)','kg');

ANALYZE cdscm.itemid_whitelists;
ANALYZE cdscm.weight_itemids;

/* ===== guards ===== */
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname='mimiciv_icu') THEN
    RAISE EXCEPTION 'missing schema mimiciv_icu';
  END IF;
END $$;

/* ===== common helpers ===== */
CREATE OR REPLACE VIEW cdscm._icu_window AS
SELECT
  i.stay_id,
  i.subject_id,
  i.hadm_id,
  i.intime,
  i.intime + (SELECT obs_window_hours FROM cdscm.cfg_params LIMIT 1)*INTERVAL '1 hour' AS cutoff
FROM mimiciv_icu.icustays i;

/* ===== drop previous staging ===== */
DROP TABLE IF EXISTS cdscm.a_map_stg;
DROP TABLE IF EXISTS cdscm.b_vaso_stg;
DROP TABLE IF EXISTS cdscm.c_urine_stg;
DROP TABLE IF EXISTS cdscm.d_creat_stg;
DROP TABLE IF EXISTS cdscm.e_rrt_stg;
DROP TABLE IF EXISTS cdscm.weight_stg;

DO $$ BEGIN RAISE NOTICE '[61_01] A_MAP staging'; END $$;
CREATE TABLE cdscm.a_map_stg AS
SELECT
  ce.stay_id, ce.itemid,
  ce.charttime AS start_time, ce.charttime AS end_time,
  ce.valuenum::NUMERIC AS value_raw,
  lower(coalesce(ce.valueuom,'')) AS unit_raw
FROM mimiciv_icu.chartevents ce
JOIN cdscm._icu_window ic USING (stay_id)
JOIN (SELECT itemid FROM cdscm.itemid_whitelists WHERE node='A_MAP') ids USING (itemid)
WHERE ce.valuenum BETWEEN (SELECT map_range_low  FROM cdscm.cfg_params LIMIT 1)
                      AND (SELECT map_range_high FROM cdscm.cfg_params LIMIT 1)
  AND ce.charttime >= ic.intime
  AND ce.charttime <  ic.cutoff
  AND lower(coalesce(ce.valueuom,''))='mmhg';
CREATE INDEX ON cdscm.a_map_stg (stay_id, start_time, end_time);
ANALYZE cdscm.a_map_stg;

DO $$ BEGIN RAISE NOTICE '[61_01] B_VASO staging'; END $$;
CREATE TABLE cdscm.b_vaso_stg AS
SELECT
  ie.stay_id, ie.itemid,
  ie.starttime AS start_time, ie.endtime AS end_time,
  ie.rate::NUMERIC AS value_raw,
  lower(coalesce(ie.rateuom,'')) AS unit_raw
FROM mimiciv_icu.inputevents ie
JOIN cdscm._icu_window ic USING (stay_id)
JOIN (SELECT itemid FROM cdscm.itemid_whitelists WHERE node='B_VASO') ids USING (itemid)
WHERE ie.rate IS NOT NULL
  AND ie.endtime > ie.starttime
  AND tstzrange(ie.starttime, ie.endtime, '[)') && tstzrange(ic.intime, ic.cutoff, '[)');
CREATE INDEX ON cdscm.b_vaso_stg (stay_id, start_time, end_time);
ANALYZE cdscm.b_vaso_stg;

DO $$ BEGIN RAISE NOTICE '[61_01] C_URINE staging'; END $$;
CREATE TABLE cdscm.c_urine_stg AS
SELECT
  oe.stay_id, oe.itemid,
  oe.charttime AS start_time, oe.charttime AS end_time,
  oe.value::NUMERIC AS value_raw,
  lower(coalesce(oe.valueuom,'')) AS unit_raw
FROM mimiciv_icu.outputevents oe
JOIN cdscm._icu_window ic USING (stay_id)
JOIN (SELECT itemid FROM cdscm.itemid_whitelists WHERE node='C_URINE') ids USING (itemid)
WHERE oe.value IS NOT NULL
  AND oe.value BETWEEN 0 AND (SELECT urine_event_max_ml FROM cdscm.cfg_params LIMIT 1)
  AND oe.charttime >= ic.intime
  AND oe.charttime <  ic.cutoff
  AND lower(coalesce(oe.valueuom,''))='ml';
CREATE INDEX ON cdscm.c_urine_stg (stay_id, start_time, end_time);
ANALYZE cdscm.c_urine_stg;

DO $$ BEGIN RAISE NOTICE '[61_01] D_CREAT staging'; END $$;
CREATE TABLE cdscm.d_creat_stg AS
SELECT
  ic.stay_id, l.itemid,
  l.charttime AS start_time, l.charttime AS end_time,
  l.valuenum::NUMERIC AS value_raw,
  lower(coalesce(l.valueuom,'')) AS unit_raw
FROM mimiciv_hosp.labevents l
JOIN (SELECT stay_id, subject_id, hadm_id, intime, cutoff FROM cdscm._icu_window) ic
  USING (subject_id, hadm_id)
JOIN (SELECT itemid FROM cdscm.itemid_whitelists WHERE node='D_CREAT') ids
  ON ids.itemid = l.itemid
WHERE l.valuenum IS NOT NULL
  AND l.charttime >= ic.intime
  AND l.charttime <  ic.cutoff
  AND lower(coalesce(l.valueuom,'')) = 'mg/dl'
  AND l.valuenum BETWEEN (SELECT creat_range_low  FROM cdscm.cfg_params LIMIT 1)
                       AND (SELECT creat_range_high FROM cdscm.cfg_params LIMIT 1);
CREATE INDEX ON cdscm.d_creat_stg (stay_id, start_time, end_time);
ANALYZE cdscm.d_creat_stg;

DO $$ BEGIN RAISE NOTICE '[61_01] E_RRT staging'; END $$;
CREATE TABLE cdscm.e_rrt_stg AS
SELECT
  p.stay_id, p.itemid,
  p.starttime AS start_time, p.endtime AS end_time,
  NULL::NUMERIC AS value_raw, NULL::TEXT AS unit_raw
FROM mimiciv_icu.procedureevents p
JOIN cdscm._icu_window ic USING (stay_id)
JOIN (SELECT itemid FROM cdscm.itemid_whitelists WHERE node='E_RRT') ids USING (itemid)
WHERE p.starttime IS NOT NULL
  AND p.endtime   IS NOT NULL
  AND p.endtime   >  p.starttime
  AND tstzrange(p.starttime, p.endtime, '[)') && tstzrange(ic.intime, ic.cutoff, '[)');
CREATE INDEX ON cdscm.e_rrt_stg (stay_id, start_time, end_time);
ANALYZE cdscm.e_rrt_stg;

DO $$ BEGIN RAISE NOTICE '[61_01] WEIGHT staging'; END $$;
CREATE TABLE cdscm.weight_stg AS
SELECT
  ce.stay_id,
  ce.charttime AS start_time,
  ce.charttime AS end_time,
  ce.valuenum::NUMERIC AS value_raw,
  lower(coalesce(ce.valueuom,'')) AS unit_raw,
  CASE
    WHEN lower(coalesce(ce.valueuom,''))='kg' THEN ce.valuenum::NUMERIC
    WHEN lower(coalesce(ce.valueuom,'')) IN ('lb','lbs','pound','pounds') THEN ce.valuenum::NUMERIC * 0.453592
    ELSE NULL::NUMERIC
  END AS weight_kg
FROM mimiciv_icu.chartevents ce
JOIN cdscm._icu_window ic USING (stay_id)
JOIN cdscm.weight_itemids wi ON wi.itemid = ce.itemid
WHERE ce.valuenum IS NOT NULL
  AND ce.charttime >= ic.intime
  AND ce.charttime <  ic.cutoff;
CREATE INDEX ON cdscm.weight_stg (stay_id, start_time, end_time);
ANALYZE cdscm.weight_stg;

/* ===== summary ===== */
DO $$ BEGIN RAISE NOTICE '[DONE] 61_01_config_and_raw_extract'; END $$;
