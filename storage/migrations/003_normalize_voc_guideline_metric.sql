-- Make the VOC guideline records reachable, and bring the metric CHECK back in line
-- with 002_guideline_records.sql.
--
-- `guideline_store.get_thresholds_for_metrics` looks records up with
-- `WHERE metric = ANY(%s)`, and the canonical metric name everywhere in the app is
-- `voc` (executors/metric_registry.py maps the legacy alias `tvoc` -> `voc`). Three
-- rows were seeded under the legacy key `tvoc`, so they never matched: VOC looked
-- like a metric with no published threshold at all. Answers said so ("no VOC
-- thresholds were provided in the sources") or quietly dropped the metric — even
-- though RESET Air, WELL v2 and WHO 2010 records were sitting in the table with
-- thresholds of 500, 500 and 300 ug/m3.
--
-- The deployed table also drifted from the repo: its CHECK permits 'tvoc' and
-- forbids 'voc', while 002_guideline_records.sql (and storage/seed_guidelines.py)
-- use 'voc'. So the rows could not simply be updated — the constraint is replaced
-- first, which is also what stops the next seeder run from failing.
--
-- Re-running the seeder does not fix any of this on its own: it is
-- ON CONFLICT DO NOTHING and one of the three rows (WELL_V2_A04) already occupies
-- its source_key, which would leave duplicates.
--
-- Idempotent: the constraint is dropped-if-exists before being recreated, and the
-- updates match nothing once applied.

ALTER TABLE env_guideline_records
    DROP CONSTRAINT IF EXISTS env_guideline_records_metric_check;

UPDATE env_guideline_records
   SET metric     = 'voc',
       updated_at = NOW()
 WHERE metric = 'tvoc';

ALTER TABLE env_guideline_records
    ADD CONSTRAINT env_guideline_records_metric_check
    CHECK (metric IN (
        'co2', 'pm25', 'voc', 'temperature',
        'humidity', 'light', 'sound', 'ieq', 'general'
    ));

-- Align the two source keys that also drifted from the seed file, so a future
-- `python -m storage.seed_guidelines` is a no-op rather than an insert of duplicates.
UPDATE env_guideline_records
   SET source_key = 'RESET_AIR_V2_VOC',
       updated_at = NOW()
 WHERE source_key = 'RESET_AIR_V2_TVOC';

UPDATE env_guideline_records
   SET source_key = 'WHO_IAQ_VOC_2010',
       updated_at = NOW()
 WHERE source_key = 'WHO_IAQ_TVOC_2010';
