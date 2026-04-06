-- Sensor readings benchmark query pack
-- Target DB: campus_iot
-- Notes:
-- 1) Replace :metric_id with a valid metric id (example used in tests: 82).
-- 2) Replace :lab_name with a valid lab/space (example: smart_lab).
-- 3) For device_id-array variants, resolve device ids first from device(space).

-- Q0. Resolve devices for a lab/space
SELECT device_id, device_name, space
FROM public.device
WHERE space = :lab_name
ORDER BY device_id;

-- Q1. Latest value by metric only
SELECT ts, state
FROM public.sensor_readings
WHERE metric_id = :metric_id
ORDER BY ts DESC
LIMIT 1;

-- Q2. Latest value by lab + metric (JOIN shape)
SELECT sr.ts, sr.state
FROM public.sensor_readings sr
JOIN public.device d ON d.device_id = sr.device_id
WHERE d.space = :lab_name
  AND sr.metric_id = :metric_id
ORDER BY sr.ts DESC
LIMIT 1;

-- Q3. Latest value by metric + resolved device set (device_id filter shape)
-- Replace :device_id_array with ARRAY[...], for example ARRAY[1,2,3]
SELECT ts, state
FROM public.sensor_readings
WHERE metric_id = :metric_id
  AND device_id = ANY(:device_id_array)
ORDER BY ts DESC
LIMIT 1;

-- Q4. 7-day hourly aggregate by lab + metric (JOIN shape)
SELECT date_trunc('hour', sr.ts) AS hour_bucket,
       avg(sr.state) AS avg_state
FROM public.sensor_readings sr
JOIN public.device d ON d.device_id = sr.device_id
WHERE d.space = :lab_name
  AND sr.metric_id = :metric_id
  AND sr.ts >= now() - interval '7 days'
GROUP BY hour_bucket
ORDER BY hour_bucket;

-- Q5. 24h comparison across two spaces
SELECT d.space,
       avg(sr.state) AS avg_state
FROM public.sensor_readings sr
JOIN public.device d ON d.device_id = sr.device_id
WHERE d.space = ANY(ARRAY['smart_lab', 'concrete_lab'])
  AND sr.metric_id = :metric_id
  AND sr.ts >= now() - interval '24 hours'
GROUP BY d.space
ORDER BY d.space;

-- Q6. Count rows in one lab (JOIN shape)
SELECT count(*)
FROM public.sensor_readings sr
JOIN public.device d ON d.device_id = sr.device_id
WHERE d.space = 'concrete_lab';

-- Q7. Count rows with resolved device list (device_id filter shape)
-- Replace :device_id_array with ARRAY[...]
SELECT count(*)
FROM public.sensor_readings
WHERE device_id = ANY(:device_id_array);

-- Optional: benchmark with EXPLAIN ANALYZE (example)
EXPLAIN (ANALYZE, BUFFERS)
SELECT sr.ts, sr.state
FROM public.sensor_readings sr
JOIN public.device d ON d.device_id = sr.device_id
WHERE d.space = 'smart_lab'
  AND sr.metric_id = 82
ORDER BY sr.ts DESC
LIMIT 1;
