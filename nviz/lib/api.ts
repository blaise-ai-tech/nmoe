// Lightweight client wrapper for the DuckDB-backed endpoints

export type RunInfo = {
  run: string
  last_ts: number
  last_step: number
  status?: string
  experiment_id?: string
  started_at?: string
  ended_at?: string | null
}

export type ExperimentSummary = {
  id: string
  name: string
  project: string
  description?: string | null
  created_at: string
  runs_total: number
  runs_running: number
  runs_completed: number
  runs_failed: number
  runs_killed: number
  last_started_at?: string | null
}

export type ExperimentRunRow = {
  run: string
  experiment_id: string
  status: string
  started_at: string
  ended_at?: string | null
  git_hash?: string | null
  git_dirty?: number | null
  config_json?: string | null
  results_json?: string | null
  last_ts: number
  last_step: number
}

export const api = {
  async runs(): Promise<RunInfo[]> {
    const r = await fetch(`/api/runs`, { cache: 'no-store' })
    const j = await r.json()
    return j.runs || []
  },
  async experiments(): Promise<ExperimentSummary[]> {
    const r = await fetch(`/api/experiments`, { cache: 'no-store' })
    const j = await r.json()
    return j.experiments || []
  },
  async experimentRuns(experiment_id?: string, limit = 200): Promise<ExperimentRunRow[]> {
    const params = new URLSearchParams()
    if (experiment_id) params.set('experiment_id', experiment_id)
    params.set('limit', String(limit))
    const qs = params.toString()
    const r = await fetch(`/api/experiments/runs${qs ? `?${qs}` : ''}`, { cache: 'no-store' })
    const j = await r.json()
    return j.runs || []
  },
  async datasets(path?: string): Promise<{ path: string; entries: Array<{ name: string; kind: 'dir' | 'file'; size: number; mtime_ms: number }> }> {
    const params = new URLSearchParams()
    if (path) params.set('path', path)
    const qs = params.toString()
    const r = await fetch(`/api/datasets${qs ? `?${qs}` : ''}`, { cache: 'no-store' })
    return await r.json()
  },
  async datasetPreview(path: string, maxBytes = 65536): Promise<{ path: string; bytes: number; text: string }> {
    const params = new URLSearchParams()
    params.set('path', path)
    params.set('max_bytes', String(maxBytes))
    const r = await fetch(`/api/datasets/preview?${params.toString()}`, { cache: 'no-store' })
    return await r.json()
  },
  async summary(run: string): Promise<Record<string, number>> {
    const r = await fetch(`/api/summary?run=${encodeURIComponent(run)}`, { cache: 'no-store' })
    const j = await r.json()
    return j.summary || {}
  },
  async series(run: string, tag: string, points = 1000): Promise<Array<{ step: number; value: number; ts_ms: number }>> {
    const r = await fetch(`/api/series?run=${encodeURIComponent(run)}&tag=${encodeURIComponent(tag)}&points=${points}`, { cache: 'no-store' })
    const j = await r.json()
    return j.series || []
  },
  async router(run: string): Promise<{ layers: Array<{ idx: number; cv?: number; entropy?: number; max_load?: number; bias_range?: number }>; agg: Record<string, number> }>{
    const r = await fetch(`/api/router?run=${encodeURIComponent(run)}`, { cache: 'no-store' })
    return await r.json()
  },
  async gpu(run: string): Promise<{ gpus: any[]; agg: Record<string, number> }>{
    const r = await fetch(`/api/gpu?run=${encodeURIComponent(run)}`, { cache: 'no-store' })
    return await r.json()
  },
}
