import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras


def dsn() -> str:
    return os.getenv("DATABASE_URL", "postgresql://admin:secret123@192.168.50.12:5432/campus_iot")


def exec_time_ms(cur: Any, sql: str, params: tuple[Any, ...]) -> float | None:
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql, params)
    plan = "\n".join([row["QUERY PLAN"] for row in cur.fetchall()])
    m = re.search(r"Execution Time: ([0-9.]+) ms", plan)
    return float(m.group(1)) if m else None


def run_query(cur: Any, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    cur.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def compact(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, datetime):
        return v.isoformat()
    if v is None:
        return "null"
    return str(v)


def rows_to_bullets(rows: list[dict[str, Any]], max_rows: int = 3) -> list[str]:
    if not rows:
        return ["- no rows"]
    out: list[str] = []
    for row in rows[:max_rows]:
        parts = [f"{k}={compact(v)}" for k, v in row.items()]
        out.append("- " + ", ".join(parts))
    if len(rows) > max_rows:
        out.append(f"- ... {len(rows) - max_rows} more rows")
    return out


def main() -> None:
    started = time.time()
    lab_name = os.getenv("BENCH_LAB", "smart_lab")

    conn = psycopg2.connect(dsn())
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Discovery from device/metric tables.
    devices_sql = """
        SELECT device_id, device_name, space
        FROM public.device
        WHERE space = %s
        ORDER BY device_id
    """
    devices = run_query(cur, devices_sql, (lab_name,))

    co2_metrics_sql = """
        SELECT m.metric_id, m.metric_name, m.domain, m.device_id
        FROM public.metric m
        JOIN public.device d ON d.device_id = m.device_id
        WHERE d.space = %s AND (m.domain = 'co2' OR lower(m.metric_name) LIKE '%%co2%%')
        ORDER BY m.metric_id
    """
    co2_metrics = run_query(cur, co2_metrics_sql, (lab_name,))

    pm25_metrics_sql = """
        SELECT m.metric_id, m.metric_name, m.domain, m.device_id
        FROM public.metric m
        JOIN public.device d ON d.device_id = m.device_id
        WHERE d.space = %s AND (m.domain = 'pm2_5' OR lower(m.metric_name) LIKE '%%pm_2_5%%')
        ORDER BY m.metric_id
    """
    pm25_metrics = run_query(cur, pm25_metrics_sql, (lab_name,))

    latest_templates = {
        "co2_latest": """
            SELECT sr.ts, sr.state, m.metric_id, m.metric_name, d.device_id, d.device_name
            FROM public.sensor_readings sr
            JOIN public.metric m ON m.metric_id = sr.metric_id
            JOIN public.device d ON d.device_id = sr.device_id
            WHERE d.space = %s AND m.domain = 'co2'
            ORDER BY sr.ts DESC
            LIMIT 1
        """,
        "pm25_latest": """
            SELECT sr.ts, sr.state, m.metric_id, m.metric_name, d.device_id, d.device_name
            FROM public.sensor_readings sr
            JOIN public.metric m ON m.metric_id = sr.metric_id
            JOIN public.device d ON d.device_id = sr.device_id
            WHERE d.space = %s AND m.domain = 'pm2_5'
            ORDER BY sr.ts DESC
            LIMIT 1
        """,
    }

    window_map = {
        "last_hour": "1 hour",
        "last_12h": "12 hours",
        "last_7d": "7 days",
        "last_14d": "14 days",
        "last_1month": "1 month",
    }

    agg_templates = {
        "co2_window": """
            SELECT
              MIN(sr.ts) AS window_start,
              MAX(sr.ts) AS window_end,
              COUNT(*) AS points,
              AVG(sr.state) AS avg_value,
              MIN(sr.state) AS min_value,
              MAX(sr.state) AS max_value
            FROM public.sensor_readings sr
            JOIN public.metric m ON m.metric_id = sr.metric_id
            JOIN public.device d ON d.device_id = sr.device_id
            WHERE d.space = %s
              AND m.domain = 'co2'
              AND sr.ts >= now() - interval %s
        """,
        "pm25_window": """
            SELECT
              MIN(sr.ts) AS window_start,
              MAX(sr.ts) AS window_end,
              COUNT(*) AS points,
              AVG(sr.state) AS avg_value,
              MIN(sr.state) AS min_value,
              MAX(sr.state) AS max_value
            FROM public.sensor_readings sr
            JOIN public.metric m ON m.metric_id = sr.metric_id
            JOIN public.device d ON d.device_id = sr.device_id
            WHERE d.space = %s
              AND m.domain = 'pm2_5'
              AND sr.ts >= now() - interval %s
        """,
        "multimetric_window": """
            SELECT
              MIN(sr.ts) AS window_start,
              MAX(sr.ts) AS window_end,
              COUNT(*) AS points,
              AVG(sr.state) FILTER (WHERE m.domain = 'co2') AS co2_avg,
              AVG(sr.state) FILTER (WHERE m.domain = 'pm2_5') AS pm25_avg,
              AVG(sr.state) FILTER (WHERE m.domain = 'temperature') AS temperature_avg,
              AVG(sr.state) FILTER (WHERE m.domain = 'humidity') AS humidity_avg,
                            AVG(sr.state) FILTER (
                                WHERE m.domain = 'voc'
                                    AND sr.state IS NOT NULL
                                    AND sr.state >= 0
                            ) AS voc_avg,
                            AVG(sr.state) FILTER (
                                WHERE m.domain = 'tvoc'
                                    AND sr.state IS NOT NULL
                                    AND sr.state >= 0
                            ) AS tvoc_avg,
                            AVG(sr.state) FILTER (
                                WHERE m.domain = 'voc_index'
                                    AND sr.state IS NOT NULL
                                    AND sr.state >= 0
                                    AND sr.state < 1000000
                            ) AS voc_index_avg,
                            COUNT(*) FILTER (
                                WHERE m.domain = 'voc_index'
                                    AND sr.state >= 1000000
                            ) AS voc_index_outlier_points
            FROM public.sensor_readings sr
            JOIN public.metric m ON m.metric_id = sr.metric_id
            JOIN public.device d ON d.device_id = sr.device_id
            WHERE d.space = %s
                            AND m.domain IN ('co2','pm2_5','temperature','humidity','voc','tvoc','voc_index')
              AND sr.ts >= now() - interval %s
        """,
    }

    query_results: list[dict[str, Any]] = []

    # Latest snapshots.
    for name, sql in latest_templates.items():
        params = (lab_name,)
        t = exec_time_ms(cur, sql, params)
        rows = run_query(cur, sql, params)
        query_results.append({
            "query_name": name,
            "window": "latest",
            "time_ms": t,
            "rows": rows,
            "sql": " ".join(sql.split()),
        })

    # Windowed aggregates.
    for window_label, interval_expr in window_map.items():
        for base_name, sql in agg_templates.items():
            params = (lab_name, interval_expr)
            t = exec_time_ms(cur, sql, params)
            rows = run_query(cur, sql, params)
            query_results.append({
                "query_name": base_name,
                "window": window_label,
                "interval": interval_expr,
                "time_ms": t,
                "rows": rows,
                "sql": " ".join(sql.split()),
            })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "lab_name": lab_name,
        "devices_count": len(devices),
        "co2_metrics_count": len(co2_metrics),
        "pm25_metrics_count": len(pm25_metrics),
        "devices": devices,
        "co2_metrics": co2_metrics,
        "pm25_metrics": pm25_metrics,
        "results": query_results,
    }

    root = os.path.dirname(__file__)
    json_path = os.path.join(root, "meeting_metrics_results.json")
    md_path = os.path.join(root, "MEETING_METRICS_REPORT.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines: list[str] = []
    lines.append("# Meeting Metrics Report")
    lines.append("")
    lines.append(f"Generated at: {payload['generated_at']}")
    lines.append(f"Lab: {lab_name}")
    lines.append("")
    lines.append("## Discovery (metric + device tables)")
    lines.append(f"- devices in lab: {len(devices)}")
    lines.append(f"- CO2 metrics in lab: {len(co2_metrics)}")
    lines.append(f"- PM2.5 metrics in lab: {len(pm25_metrics)}")
    lines.append("")
    lines.append("### Devices")
    lines.extend(rows_to_bullets(devices, max_rows=20))
    lines.append("")
    lines.append("### CO2 Metric Rows")
    lines.extend(rows_to_bullets(co2_metrics, max_rows=20))
    lines.append("")
    lines.append("### PM2.5 Metric Rows")
    lines.extend(rows_to_bullets(pm25_metrics, max_rows=20))
    lines.append("")

    lines.append("## Query Results (sensor_readings)")
    lines.append("")
    lines.append("| Query | Window | Time (ms) | Key Output |")
    lines.append("|---|---|---:|---|")

    for item in query_results:
        row0 = item["rows"][0] if item["rows"] else {}
        key_out = ", ".join([f"{k}={compact(v)}" for k, v in list(row0.items())[:4]]) if row0 else "no rows"
        lines.append(
            f"| {item['query_name']} | {item.get('window','')} | {item.get('time_ms', 0) or 0:.3f} | {key_out} |"
        )

    lines.append("")
    lines.append("## SQL Used")
    lines.append("")
    for item in query_results:
        lines.append(f"### {item['query_name']} ({item.get('window','')})")
        lines.append("")
        lines.append("```sql")
        lines.append(item["sql"])
        lines.append("```")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({
        "json": json_path,
        "markdown": md_path,
        "queries": len(query_results),
        "duration_seconds": payload["duration_seconds"],
    }, indent=2))

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
