import json
import os
import re
import time
from typing import Any

import psycopg2
import psycopg2.extras


def _dsn() -> str:
    return os.getenv("DATABASE_URL", "postgresql://admin:secret123@192.168.50.12:5432/campus_iot")


def _exec_time_ms(cur: Any, sql: str, params: tuple[Any, ...] = ()) -> float | None:
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql, params)
    plan_lines = [row["QUERY PLAN"] for row in cur.fetchall()]
    plan_text = "\n".join(plan_lines)
    match = re.search(r"Execution Time: ([0-9.]+) ms", plan_text)
    if not match:
        return None
    return float(match.group(1))


def _fetch_sample(cur: Any, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    started = time.time()
    conn = psycopg2.connect(_dsn())
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Resolve benchmark fixture values.
    lab_name = "smart_lab"

    # Pick a metric id that actually has data for the target lab.
    cur.execute(
        """
        SELECT sr.metric_id
        FROM public.sensor_readings sr
        JOIN public.device d ON d.device_id = sr.device_id
        WHERE d.space = %s
        GROUP BY sr.metric_id
        ORDER BY count(*) DESC
        LIMIT 1
        """,
        (lab_name,),
    )
    metric_row = cur.fetchone()
    metric_id = int(metric_row["metric_id"]) if metric_row else 82

    cur.execute("SELECT device_id FROM public.device WHERE space=%s ORDER BY device_id", (lab_name,))
    smart_ids = [row["device_id"] for row in cur.fetchall()]

    cur.execute("SELECT device_id FROM public.device WHERE space='concrete_lab' ORDER BY device_id")
    concrete_ids = [row["device_id"] for row in cur.fetchall()]

    queries = [
        {
            "name": "metric_only_latest",
            "sql": """
                SELECT ts, state
                FROM public.sensor_readings
                WHERE metric_id=%s
                ORDER BY ts DESC
                LIMIT 1
            """,
            "params": (metric_id,),
            "sample_sql": """
                SELECT ts, state
                FROM public.sensor_readings
                WHERE metric_id=%s
                ORDER BY ts DESC
                LIMIT 1
            """,
            "sample_params": (metric_id,),
        },
        {
            "name": "lab_metric_latest_join",
            "sql": """
                SELECT sr.ts, sr.state
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space=%s AND sr.metric_id=%s
                ORDER BY sr.ts DESC
                LIMIT 1
            """,
            "params": (lab_name, metric_id),
            "sample_sql": """
                SELECT sr.ts, sr.state
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space=%s AND sr.metric_id=%s
                ORDER BY sr.ts DESC
                LIMIT 1
            """,
            "sample_params": (lab_name, metric_id),
        },
        {
            "name": "lab_metric_latest_device_filter",
            "sql": """
                SELECT ts, state
                FROM public.sensor_readings
                WHERE metric_id=%s AND device_id=ANY(%s)
                ORDER BY ts DESC
                LIMIT 1
            """,
            "params": (metric_id, smart_ids),
            "sample_sql": """
                SELECT ts, state
                FROM public.sensor_readings
                WHERE metric_id=%s AND device_id=ANY(%s)
                ORDER BY ts DESC
                LIMIT 1
            """,
            "sample_params": (metric_id, smart_ids),
        },
        {
            "name": "lab_metric_7d_agg_join",
            "sql": """
                SELECT date_trunc('hour', sr.ts) AS h, avg(sr.state) AS v
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space=%s AND sr.metric_id=%s
                  AND sr.ts >= now() - interval '7 days'
                GROUP BY h
                ORDER BY h
            """,
            "params": (lab_name, metric_id),
            "sample_sql": """
                SELECT date_trunc('hour', sr.ts) AS h, avg(sr.state) AS v
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space=%s AND sr.metric_id=%s
                  AND sr.ts >= now() - interval '7 days'
                GROUP BY h
                ORDER BY h DESC
                LIMIT 5
            """,
            "sample_params": (lab_name, metric_id),
        },
        {
            "name": "smart_vs_concrete_24h_compare",
            "sql": """
                SELECT d.space, avg(sr.state) AS avg_state
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space = ANY(%s)
                  AND sr.metric_id=%s
                  AND sr.ts >= now() - interval '24 hours'
                GROUP BY d.space
                ORDER BY d.space
            """,
            "params": (["smart_lab", "concrete_lab"], metric_id),
            "sample_sql": """
                SELECT d.space, avg(sr.state) AS avg_state
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space = ANY(%s)
                  AND sr.metric_id=%s
                  AND sr.ts >= now() - interval '24 hours'
                GROUP BY d.space
                ORDER BY d.space
            """,
            "sample_params": (["smart_lab", "concrete_lab"], metric_id),
        },
        {
            "name": "concrete_count_join",
            "sql": """
                SELECT count(*)
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space='concrete_lab'
            """,
            "params": (),
            "sample_sql": """
                SELECT count(*)
                FROM public.sensor_readings sr
                JOIN public.device d ON d.device_id = sr.device_id
                WHERE d.space='concrete_lab'
            """,
            "sample_params": (),
        },
        {
            "name": "concrete_count_device_filter",
            "sql": """
                SELECT count(*)
                FROM public.sensor_readings
                WHERE device_id=ANY(%s)
            """,
            "params": (concrete_ids,),
            "sample_sql": """
                SELECT count(*)
                FROM public.sensor_readings
                WHERE device_id=ANY(%s)
            """,
            "sample_params": (concrete_ids,),
        },
    ]

    results: list[dict[str, Any]] = []
    for item in queries:
        exec_ms = _exec_time_ms(cur, item["sql"], item["params"])
        sample_rows = _fetch_sample(cur, item["sample_sql"], item["sample_params"])
        if not sample_rows:
            # Keep examples useful even when a specific window has no rows.
            if item["name"] in {"lab_metric_latest_join", "lab_metric_latest_device_filter"}:
                sample_rows = _fetch_sample(
                    cur,
                    """
                    SELECT sr.ts, sr.state
                    FROM public.sensor_readings sr
                    JOIN public.device d ON d.device_id = sr.device_id
                    WHERE d.space=%s AND sr.metric_id=%s
                    ORDER BY sr.ts DESC
                    LIMIT 1
                    """,
                    (lab_name, metric_id),
                )
            elif item["name"] == "lab_metric_7d_agg_join":
                sample_rows = _fetch_sample(
                    cur,
                    """
                    SELECT date_trunc('hour', sr.ts) AS h, avg(sr.state) AS v
                    FROM public.sensor_readings sr
                    JOIN public.device d ON d.device_id = sr.device_id
                    WHERE d.space=%s AND sr.metric_id=%s
                    GROUP BY h
                    ORDER BY h DESC
                    LIMIT 5
                    """,
                    (lab_name, metric_id),
                )
        results.append(
            {
                "name": item["name"],
                "execution_time_ms": exec_ms,
                "sample_output": sample_rows,
            }
        )

    payload = {
        "generated_at_epoch": int(time.time()),
        "duration_seconds": round(time.time() - started, 3),
        "dsn_hint": "DATABASE_URL env or default campus_iot DSN",
        "fixture": {
            "metric_id": metric_id,
            "lab_name": lab_name,
            "smart_lab_device_count": len(smart_ids),
            "concrete_lab_device_count": len(concrete_ids),
        },
        "results": results,
    }

    output_path = os.path.join(os.path.dirname(__file__), "sensor_readings_benchmark_latest.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)

    print(json.dumps(payload, indent=2, default=str))
    print("\nWrote:", output_path)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
