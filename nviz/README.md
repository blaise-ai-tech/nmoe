# NVIZ: NMoE Metrics Dashboard

Next.js dashboard for visualizing NMoE training runs.

## Setup

```bash
cd nviz
bun install
bun run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Configuration

Training writes timeseries metrics to DuckDB files:
- `/data/metrics/{run_id}/rank_{rank}.duckdb`

NVIZ reads metrics by attaching all per-rank DuckDB files for a run (read-only).
Point NVIZ at the parent metrics directory:
```bash
export NVIZ_METRICS_DIR=/data/metrics
```

## Features

- **Run Comparison**: Compare loss curves and throughput across runs
- **Real-time Updates**: SQLite WAL mode enables live metric streaming
- **Experiment Tracking**: Filter by config hash, preset, dtype
- **GPU Telemetry**: Memory usage, utilization (when NVML available)

## Kubernetes Deployment

```bash
kubectl apply -f k8s/nviz.yaml
```

The pod mounts `/data` read-only and exposes port 3000.

## Metrics Schema

Timeseries metrics are stored in DuckDB by `nmoe/metrics.py` as a single table:
- `metrics(run, tag, step, ts_ms, value)`

## Development

```bash
bun run dev    # Development server with hot reload
bun run build  # Production build
bun run start  # Production server
```
