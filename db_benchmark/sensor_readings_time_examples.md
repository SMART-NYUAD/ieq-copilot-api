# Sensor Readings Benchmark: Query + Output + Time Examples

Generated from live runs on campus_iot.

## Run Command

```bash
cd /home/smart/RAG_API_SERVER
python db_benchmark/run_sensor_readings_benchmark.py
```

Latest JSON snapshot:

- db_benchmark/sensor_readings_benchmark_latest.json

## Example Query Outputs and Times (latest run)

Fixture used by runner:

- lab_name: smart_lab
- metric_id: 1 (auto-selected as top metric for smart_lab)
- smart_lab_device_count: 8
- concrete_lab_device_count: 1

| Query | Example Output (first row) | Execution Time (ms) |
|---|---|---:|
| metric_only_latest | ts=2026-04-03 23:30:32+00:00, state=54.0 | 18.138 |
| lab_metric_latest_join | ts=2026-04-03 23:30:32+00:00, state=54.0 | 18.272 |
| lab_metric_latest_device_filter | ts=2026-04-03 23:30:32+00:00, state=54.0 | 20.087 |
| lab_metric_7d_agg_join | h=2026-04-03 23:00:00+00:00, v=54.0 | 33.254 |
| smart_vs_concrete_24h_compare | space=smart_lab, avg_state=54.0 | 2.417 |
| concrete_count_join | count=3338097 | 2922.691 |
| concrete_count_device_filter | count=3338097 | 304.465 |

## Stress-Case Timing Example (metric-specific shape)

This pattern has been consistently expensive in prior runs when metric distribution and lab coverage force deep scans.

| Query Shape | Time (ms) |
|---|---:|
| lab_metric_latest_join (metric_id=82, smart_lab) | 1394.111 |
| concrete_count_join | 2865.935 |
| concrete_count_device_filter | 295.621 |

## Interpretation

- device_id filter shapes are generally more stable for lab-scoped access.
- join count over concrete_lab remains the major bottleneck.
- indexes now present:
  - ix_device_space on device(space)
  - ix_sr_device_metric_ts on sensor_readings(device_id, metric_id, ts desc)
