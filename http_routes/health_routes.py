"""Health and root endpoints."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

try:
    from query_routing.observability import (
        evaluate_rollout_slo,
        get_error_observability_snapshot,
        get_http_observability_snapshot,
        get_observability_kpis,
        get_observability_snapshot,
    )
except ImportError:
    from ..query_routing.observability import (
        evaluate_rollout_slo,
        get_error_observability_snapshot,
        get_http_observability_snapshot,
        get_observability_kpis,
        get_observability_snapshot,
    )


router = APIRouter()


@router.get("/")
async def root():
    return {
        "message": "Environment Cards RAG API",
        "version": "2.0.0",
        "endpoints": {
            "query": "POST /query - Routed query endpoint",
            "query_stream": "POST /query/stream - Routed streaming query",
            "query_route": "POST /query/route - Intent router preview",
            "health": "GET /health - Health check",
            "router_health": "GET /health/router - Router safety metrics",
            "observability": "GET /observability - Live observability dashboard",
            "observability_json": "GET /observability/kpis - Full KPI snapshot",
        },
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/health/router")
async def router_health():
    snapshot = get_observability_snapshot()
    slo = evaluate_rollout_slo(snapshot)
    thresholds = {
        "planner_fallback_rate_target": 0.05,
        "planner_fallback_rate_max": 0.10,
        "shadow_diff_rate_target": 0.10,
        "shadow_diff_rate_max": 0.20,
        "sync_stream_flip_rate_target": 0.0,
        "sync_stream_flip_rate_max": 0.01,
    }
    return {
        "status": "healthy",
        "router_rollout": {
            "strategy": "policy_engine_only",
        },
        "router_mode": {
            "strategy": "policy_engine_only",
        },
        "metrics": snapshot,
        "thresholds": thresholds,
        "slo": slo,
    }


@router.get("/observability/metrics")
async def observability_metrics():
    return {
        "router": get_observability_snapshot(),
        "http": get_http_observability_snapshot(),
        "errors": get_error_observability_snapshot(),
    }


@router.get("/observability/kpis")
async def observability_kpis():
    return get_observability_kpis()


@router.get("/observability", response_class=HTMLResponse)
async def observability_dashboard():
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RAG Observability Dashboard</title>
    <style>
      :root {
        --bg: #0b1220;
        --panel: #101a2f;
        --text: #e8edf7;
        --muted: #9fb0cc;
        --accent: #4da3ff;
        --good: #3ecf8e;
        --warn: #f6c760;
        --bad: #ff6b6b;
      }
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
      }
      .wrap {
        max-width: 1200px;
        margin: 20px auto;
        padding: 0 16px 20px;
      }
      h1 {
        margin: 0 0 10px;
        font-size: 24px;
      }
      .sub {
        color: var(--muted);
        margin-bottom: 16px;
      }
      .kpis {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 12px;
        margin-bottom: 14px;
      }
      .card {
        background: var(--panel);
        border: 1px solid #1d2b49;
        border-radius: 10px;
        padding: 12px;
      }
      .label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 6px;
      }
      .val {
        font-size: 22px;
        font-weight: 700;
      }
      .sections {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
        line-height: 1.4;
      }
      .status-good { color: var(--good); }
      .status-warn { color: var(--warn); }
      .status-bad { color: var(--bad); }
      @media (max-width: 980px) {
        .sections { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>RAG Observability Dashboard</h1>
      <div class="sub">Live API + Router KPIs (auto-refresh every 5s)</div>
      <div id="kpis" class="kpis"></div>
      <div class="sections">
        <div class="card">
          <div class="label">Router + SLO</div>
          <pre id="router"></pre>
        </div>
        <div class="card">
          <div class="label">HTTP + Errors</div>
          <pre id="http"></pre>
        </div>
      </div>
    </div>
    <script>
      const fmt = (v) => {
        if (typeof v !== 'number') return String(v ?? '-');
        if (Math.abs(v) >= 1000) return v.toFixed(0);
        if (Math.abs(v) >= 100) return v.toFixed(1);
        return v.toFixed(3);
      };
      const pct = (v) => `${(Number(v || 0) * 100).toFixed(2)}%`;
      const statusClass = (label, value) => {
        if (label.includes('error') || label.includes('failure') || label.includes('flip') || label.includes('fallback')) {
          if (value < 0.02) return 'status-good';
          if (value < 0.05) return 'status-warn';
          return 'status-bad';
        }
        return 'status-good';
      };
      async function refresh() {
        const res = await fetch('/observability/kpis', { cache: 'no-store' });
        const data = await res.json();
        const k = data.kpis || {};
        const items = [
          ['Requests', k.requests_total ?? 0],
          ['Errors', k.request_errors_total ?? 0],
          ['Availability', pct(k.availability_rate)],
          ['Error Rate', pct(k.request_error_rate)],
          ['Throughput (rps)', fmt(k.throughput_rps)],
          ['p95 Latency (ms)', fmt(k.latency_p95_ms)],
          ['p99 Latency (ms)', fmt(k.latency_p99_ms)],
          ['Router Fallback Rate', pct(k.router_planner_fallback_rate)],
          ['Shadow Diff Rate', pct(k.router_shadow_diff_rate)],
          ['Sync/Stream Flip Rate', pct(k.router_sync_stream_flip_rate)],
          ['Runtime Errors', k.runtime_errors_total ?? 0]
        ];
        const container = document.getElementById('kpis');
        container.innerHTML = items.map(([label, value]) => {
          const cls = statusClass(label.toLowerCase(), Number(String(value).replace('%','')) / 100 || 0);
          return `<div class="card"><div class="label">${label}</div><div class="val ${cls}">${value}</div></div>`;
        }).join('');
        document.getElementById('router').textContent = JSON.stringify({
          router: data.router,
          router_slo: data.router_slo
        }, null, 2);
        document.getElementById('http').textContent = JSON.stringify({
          http: data.http,
          errors: data.errors
        }, null, 2);
      }
      refresh();
      setInterval(refresh, 5000);
    </script>
  </body>
</html>
    """.strip()
    return HTMLResponse(content=html)

