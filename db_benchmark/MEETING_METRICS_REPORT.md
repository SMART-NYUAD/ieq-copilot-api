# Meeting Metrics Report

Generated at: 2026-04-03T20:07:28.996065+00:00
Lab: smart_lab

## Discovery (metric + device tables)
- devices in lab: 8
- CO2 metrics in lab: 1
- PM2.5 metrics in lab: 3

### Devices
- device_id=2, device_name=all_sensor_02, space=smart_lab
- device_id=21, device_name=all_sensor_65, space=smart_lab
- device_id=22, device_name=all_sensor_66, space=smart_lab
- device_id=23, device_name=all_sensor_67, space=smart_lab
- device_id=24, device_name=all_sensor_68, space=smart_lab
- device_id=25, device_name=all_sensor_69, space=smart_lab
- device_id=27, device_name=all_sensor_71, space=smart_lab
- device_id=29, device_name=all_sensor_73, space=smart_lab

### CO2 Metric Rows
- metric_id=215, metric_name=all_sensor_73_co2, domain=co2, device_id=29

### PM2.5 Metric Rows
- metric_id=6, metric_name=all_sensor_02_pm_2_5, domain=pm2_5, device_id=2
- metric_id=196, metric_name=all_sensor_71_pm_2_5, domain=pm2_5, device_id=27
- metric_id=211, metric_name=all_sensor_73_pm_2_5, domain=pm2_5, device_id=29

## Query Results (sensor_readings)

| Query | Window | Time (ms) | Key Output |
|---|---|---:|---|
| co2_latest | latest | 15.710 | ts=2026-04-04T00:07:13+00:00, state=414.000, metric_id=215, metric_name=all_sensor_73_co2 |
| pm25_latest | latest | 13.880 | ts=2026-04-04T00:07:13+00:00, state=3.800, metric_id=211, metric_name=all_sensor_73_pm_2_5 |
| co2_window | last_hour | 4.069 | window_start=2026-04-03T19:08:13+00:00, window_end=2026-04-04T00:07:13+00:00, points=300, avg_value=416.220 |
| pm25_window | last_hour | 3.383 | window_start=2026-04-03T19:07:31+00:00, window_end=2026-04-04T00:07:13+00:00, points=900, avg_value=4.593 |
| multimetric_window | last_hour | 20.104 | window_start=2026-04-03T19:07:30+00:00, window_end=2026-04-04T00:07:13+00:00, points=5696, co2_avg=416.220 |
| co2_window | last_12h | 11.108 | window_start=2026-04-03T08:08:13+00:00, window_end=2026-04-04T00:07:13+00:00, points=960, avg_value=415.479 |
| pm25_window | last_12h | 11.778 | window_start=2026-04-03T08:07:31+00:00, window_end=2026-04-04T00:07:13+00:00, points=2863, avg_value=4.117 |
| multimetric_window | last_12h | 38.827 | window_start=2026-04-03T08:07:31+00:00, window_end=2026-04-04T00:07:13+00:00, points=18175, co2_avg=415.479 |
| co2_window | last_7d | 167.392 | window_start=2026-03-27T20:07:52+00:00, window_end=2026-04-04T00:07:13+00:00, points=10321, avg_value=416.296 |
| pm25_window | last_7d | 157.648 | window_start=2026-03-27T20:07:36+00:00, window_end=2026-04-04T00:07:13+00:00, points=29239, avg_value=2.171 |
| multimetric_window | last_7d | 239.893 | window_start=2026-03-27T20:07:36+00:00, window_end=2026-04-04T00:07:13+00:00, points=189147, co2_avg=416.296 |
| co2_window | last_14d | 314.032 | window_start=2026-03-20T20:07:25+00:00, window_end=2026-04-04T00:07:13+00:00, points=18804, avg_value=419.671 |
| pm25_window | last_14d | 322.986 | window_start=2026-03-20T20:07:25+00:00, window_end=2026-04-04T00:07:13+00:00, points=54609, avg_value=2.117 |
| multimetric_window | last_14d | 365.115 | window_start=2026-03-20T20:07:44+00:00, window_end=2026-04-04T00:07:13+00:00, points=349694, co2_avg=419.671 |
| co2_window | last_1month | 612.595 | window_start=2026-03-03T20:07:58+00:00, window_end=2026-04-04T00:07:13+00:00, points=43285, avg_value=418.751 |
| pm25_window | last_1month | 614.631 | window_start=2026-03-03T20:07:48+00:00, window_end=2026-04-04T00:07:13+00:00, points=126582, avg_value=4.425 |
| multimetric_window | last_1month | 797.122 | window_start=2026-03-03T20:07:42+00:00, window_end=2026-04-04T00:07:13+00:00, points=800842, co2_avg=418.751 |

## SQL Used

### co2_latest (latest)

```sql
SELECT sr.ts, sr.state, m.metric_id, m.metric_name, d.device_id, d.device_name FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'co2' ORDER BY sr.ts DESC LIMIT 1
```

### pm25_latest (latest)

```sql
SELECT sr.ts, sr.state, m.metric_id, m.metric_name, d.device_id, d.device_name FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'pm2_5' ORDER BY sr.ts DESC LIMIT 1
```

### co2_window (last_hour)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'co2' AND sr.ts >= now() - interval %s
```

### pm25_window (last_hour)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'pm2_5' AND sr.ts >= now() - interval %s
```

### multimetric_window (last_hour)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) FILTER (WHERE m.domain = 'co2') AS co2_avg, AVG(sr.state) FILTER (WHERE m.domain = 'pm2_5') AS pm25_avg, AVG(sr.state) FILTER (WHERE m.domain = 'temperature') AS temperature_avg, AVG(sr.state) FILTER (WHERE m.domain = 'humidity') AS humidity_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS voc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'tvoc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS tvoc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc_index' AND sr.state IS NOT NULL AND sr.state >= 0 AND sr.state < 1000000 ) AS voc_index_avg, COUNT(*) FILTER ( WHERE m.domain = 'voc_index' AND sr.state >= 1000000 ) AS voc_index_outlier_points FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain IN ('co2','pm2_5','temperature','humidity','voc','tvoc','voc_index') AND sr.ts >= now() - interval %s
```

### co2_window (last_12h)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'co2' AND sr.ts >= now() - interval %s
```

### pm25_window (last_12h)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'pm2_5' AND sr.ts >= now() - interval %s
```

### multimetric_window (last_12h)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) FILTER (WHERE m.domain = 'co2') AS co2_avg, AVG(sr.state) FILTER (WHERE m.domain = 'pm2_5') AS pm25_avg, AVG(sr.state) FILTER (WHERE m.domain = 'temperature') AS temperature_avg, AVG(sr.state) FILTER (WHERE m.domain = 'humidity') AS humidity_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS voc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'tvoc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS tvoc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc_index' AND sr.state IS NOT NULL AND sr.state >= 0 AND sr.state < 1000000 ) AS voc_index_avg, COUNT(*) FILTER ( WHERE m.domain = 'voc_index' AND sr.state >= 1000000 ) AS voc_index_outlier_points FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain IN ('co2','pm2_5','temperature','humidity','voc','tvoc','voc_index') AND sr.ts >= now() - interval %s
```

### co2_window (last_7d)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'co2' AND sr.ts >= now() - interval %s
```

### pm25_window (last_7d)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'pm2_5' AND sr.ts >= now() - interval %s
```

### multimetric_window (last_7d)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) FILTER (WHERE m.domain = 'co2') AS co2_avg, AVG(sr.state) FILTER (WHERE m.domain = 'pm2_5') AS pm25_avg, AVG(sr.state) FILTER (WHERE m.domain = 'temperature') AS temperature_avg, AVG(sr.state) FILTER (WHERE m.domain = 'humidity') AS humidity_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS voc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'tvoc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS tvoc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc_index' AND sr.state IS NOT NULL AND sr.state >= 0 AND sr.state < 1000000 ) AS voc_index_avg, COUNT(*) FILTER ( WHERE m.domain = 'voc_index' AND sr.state >= 1000000 ) AS voc_index_outlier_points FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain IN ('co2','pm2_5','temperature','humidity','voc','tvoc','voc_index') AND sr.ts >= now() - interval %s
```

### co2_window (last_14d)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'co2' AND sr.ts >= now() - interval %s
```

### pm25_window (last_14d)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'pm2_5' AND sr.ts >= now() - interval %s
```

### multimetric_window (last_14d)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) FILTER (WHERE m.domain = 'co2') AS co2_avg, AVG(sr.state) FILTER (WHERE m.domain = 'pm2_5') AS pm25_avg, AVG(sr.state) FILTER (WHERE m.domain = 'temperature') AS temperature_avg, AVG(sr.state) FILTER (WHERE m.domain = 'humidity') AS humidity_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS voc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'tvoc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS tvoc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc_index' AND sr.state IS NOT NULL AND sr.state >= 0 AND sr.state < 1000000 ) AS voc_index_avg, COUNT(*) FILTER ( WHERE m.domain = 'voc_index' AND sr.state >= 1000000 ) AS voc_index_outlier_points FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain IN ('co2','pm2_5','temperature','humidity','voc','tvoc','voc_index') AND sr.ts >= now() - interval %s
```

### co2_window (last_1month)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'co2' AND sr.ts >= now() - interval %s
```

### pm25_window (last_1month)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) AS avg_value, MIN(sr.state) AS min_value, MAX(sr.state) AS max_value FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain = 'pm2_5' AND sr.ts >= now() - interval %s
```

### multimetric_window (last_1month)

```sql
SELECT MIN(sr.ts) AS window_start, MAX(sr.ts) AS window_end, COUNT(*) AS points, AVG(sr.state) FILTER (WHERE m.domain = 'co2') AS co2_avg, AVG(sr.state) FILTER (WHERE m.domain = 'pm2_5') AS pm25_avg, AVG(sr.state) FILTER (WHERE m.domain = 'temperature') AS temperature_avg, AVG(sr.state) FILTER (WHERE m.domain = 'humidity') AS humidity_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS voc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'tvoc' AND sr.state IS NOT NULL AND sr.state >= 0 ) AS tvoc_avg, AVG(sr.state) FILTER ( WHERE m.domain = 'voc_index' AND sr.state IS NOT NULL AND sr.state >= 0 AND sr.state < 1000000 ) AS voc_index_avg, COUNT(*) FILTER ( WHERE m.domain = 'voc_index' AND sr.state >= 1000000 ) AS voc_index_outlier_points FROM public.sensor_readings sr JOIN public.metric m ON m.metric_id = sr.metric_id JOIN public.device d ON d.device_id = sr.device_id WHERE d.space = %s AND m.domain IN ('co2','pm2_5','temperature','humidity','voc','tvoc','voc_index') AND sr.ts >= now() - interval %s
```

