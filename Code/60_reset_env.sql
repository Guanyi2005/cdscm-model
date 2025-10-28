
\encoding UTF8
\set ON_ERROR_STOP on
\timing on


SELECT current_database() AS db, current_user AS usr;

DROP SCHEMA IF EXISTS cdscm_ro    CASCADE;
DROP SCHEMA IF EXISTS cdscm_meta  CASCADE;
DROP SCHEMA IF EXISTS cdscm       CASCADE;

CREATE SCHEMA cdscm;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname='public' AND c.relkind='r' AND c.relname='tmp_chaina_by_stay')
  THEN EXECUTE 'DROP TABLE public.tmp_chaina_by_stay'; END IF;

  IF EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname='public' AND c.relkind='r' AND c.relname='tmp_chaina_summary')
  THEN EXECUTE 'DROP TABLE public.tmp_chaina_summary'; END IF;

  IF EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname='public' AND c.relkind='r' AND c.relname='tmp_chainb_by_stay')
  THEN EXECUTE 'DROP TABLE public.tmp_chainb_by_stay'; END IF;

  IF EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname='public' AND c.relkind='r' AND c.relname='tmp_chainb_summary')
  THEN EXECUTE 'DROP TABLE public.tmp_chainb_summary'; END IF;

  IF EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname='public' AND c.relkind='r' AND c.relname='tmp_raw_hourly_curves')
  THEN EXECUTE 'DROP TABLE public.tmp_raw_hourly_curves'; END IF;
END $$;


SET search_path = cdscm, public;


SELECT n.nspname AS schema, c.relkind AS kind, c.relname AS name
FROM pg_class c
JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='cdscm'
ORDER BY kind, name;


SHOW search_path;
