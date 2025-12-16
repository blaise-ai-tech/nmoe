// Server-only experiment/run metadata reader backed by SQLite.
//
// nmoe writes experiment/run metadata to a single SQLite DB (default: /data/experiments.db).
// Metrics live in per-rank DuckDB files under NVIZ_METRICS_DIR.

import { copyFileSync, existsSync, mkdirSync, rmSync, statSync } from 'node:fs'

export type SqliteRun = {
  run: string
  experiment_id: string
  status: string
  started_at: string
  ended_at?: string | null
  git_hash?: string | null
  git_dirty?: number | null
}

export type SqliteRunFull = SqliteRun & {
  config_json?: string | null
  results_json?: string | null
}

function experimentsDbPath(): string {
  return process.env.NVIZ_EXPERIMENTS_DB || '/data/experiments.db'
}

function requireBunRuntime(): void {
  // NVIZ is intentionally Bun-first (see docker/Dockerfile.nviz). Using bun:sqlite
  // on a Node runtime will fail; treat that as a configuration error.
  const bun = (process as any)?.versions?.bun
  if (!bun) {
    throw new Error("NVIZ requires Bun runtime (use `bun run dev` / `bun run start`).")
  }
}

let _snapshotSig: string | null = null
let _snapshotPath: string | null = null

function snapshotPaths(src: string): { src: string; dst: string; walSrc: string; walDst: string; shmSrc: string; shmDst: string } {
  const dstDir = '/tmp/nviz_experiments'
  mkdirSync(dstDir, { recursive: true })
  const dst = `${dstDir}/experiments.db`
  return {
    src,
    dst,
    walSrc: `${src}-wal`,
    walDst: `${dst}-wal`,
    shmSrc: `${src}-shm`,
    shmDst: `${dst}-shm`,
  }
}

function computeSig(src: string): string {
  const st = statSync(src)
  const parts = [`${src}`, `${st.size}`, `${st.mtimeMs}`]
  const wal = `${src}-wal`
  const shm = `${src}-shm`
  if (existsSync(wal)) {
    const w = statSync(wal)
    parts.push(`${w.size}`, `${w.mtimeMs}`)
  } else {
    parts.push('no-wal')
  }
  if (existsSync(shm)) {
    const s = statSync(shm)
    parts.push(`${s.size}`, `${s.mtimeMs}`)
  } else {
    parts.push('no-shm')
  }
  return parts.join(':')
}

function ensureSnapshot(src: string): string {
  const sig = computeSig(src)
  if (_snapshotSig === sig && _snapshotPath && existsSync(_snapshotPath)) return _snapshotPath

  const p = snapshotPaths(src)
  copyFileSync(p.src, p.dst)
  if (existsSync(p.walSrc)) copyFileSync(p.walSrc, p.walDst)
  else rmSync(p.walDst, { force: true })
  if (existsSync(p.shmSrc)) copyFileSync(p.shmSrc, p.shmDst)
  else rmSync(p.shmDst, { force: true })

  _snapshotSig = sig
  _snapshotPath = p.dst
  return p.dst
}

function tryOpenDb(): any | null {
  requireBunRuntime()
  const src = experimentsDbPath()
  if (!existsSync(src)) return null

  // Some PVCs don't support SQLite locking semantics (WAL/shared-memory),
  // so we snapshot-copy to local /tmp and query that copy.
  const path = ensureSnapshot(src)
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { Database } = require('bun:sqlite')
    return new Database(path, { readonly: true })
  } catch (e: any) {
    throw new Error(`Failed to open experiments DB snapshot at ${path}: ${String(e)}`)
  }
}

export function listRuns(): SqliteRun[] {
  let db: any | null = null
  try {
    db = tryOpenDb()
    if (!db) return []
    const q = db.query(`
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty
      FROM runs
      ORDER BY started_at DESC
    `)
    return q.all() as SqliteRun[]
  } catch (e: any) {
    // Best-effort: NVIZ should keep serving DuckDB metrics even if the
    // experiment DB can't be opened on this filesystem.
    console.error(`[nviz] experiments.db unavailable: ${String(e)}`)
    return []
  } finally {
    try { db?.close?.() } catch {}
  }
}

export type SqliteExperiment = {
  id: string
  name: string
  project: string
  description?: string | null
  created_at: string
}

export function listExperiments(): SqliteExperiment[] {
  let db: any | null = null
  try {
    db = tryOpenDb()
    if (!db) return []
    const q = db.query(`
      SELECT id, name, project, description, created_at
      FROM experiments
      ORDER BY created_at DESC
    `)
    return q.all() as SqliteExperiment[]
  } catch (e: any) {
    console.error(`[nviz] experiments.db unavailable: ${String(e)}`)
    return []
  } finally {
    try { db?.close?.() } catch {}
  }
}

export type ExperimentSummary = SqliteExperiment & {
  runs_total: number
  runs_running: number
  runs_completed: number
  runs_failed: number
  runs_killed: number
  last_started_at?: string | null
}

export function listExperimentSummaries(): ExperimentSummary[] {
  let db: any | null = null
  try {
    db = tryOpenDb()
    if (!db) return []
    const q = db.query(`
      SELECT
        e.id,
        e.name,
        e.project,
        e.description,
        e.created_at,
        COUNT(r.id) AS runs_total,
        SUM(CASE WHEN r.status = 'running' THEN 1 ELSE 0 END) AS runs_running,
        SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END) AS runs_completed,
        SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) AS runs_failed,
        SUM(CASE WHEN r.status = 'killed' THEN 1 ELSE 0 END) AS runs_killed,
        MAX(r.started_at) AS last_started_at
      FROM experiments e
      LEFT JOIN runs r ON r.experiment_id = e.id
      GROUP BY e.id, e.name, e.project, e.description, e.created_at
      ORDER BY (MAX(r.started_at) IS NULL) ASC, MAX(r.started_at) DESC, e.created_at DESC
    `)
    return (q.all() as any[]).map((r) => ({
      id: String(r.id),
      name: String(r.name),
      project: String(r.project),
      description: r.description == null ? null : String(r.description),
      created_at: String(r.created_at),
      runs_total: Number(r.runs_total || 0),
      runs_running: Number(r.runs_running || 0),
      runs_completed: Number(r.runs_completed || 0),
      runs_failed: Number(r.runs_failed || 0),
      runs_killed: Number(r.runs_killed || 0),
      last_started_at: r.last_started_at == null ? null : String(r.last_started_at),
    }))
  } catch (e: any) {
    console.error(`[nviz] experiments.db unavailable: ${String(e)}`)
    return []
  } finally {
    try { db?.close?.() } catch {}
  }
}

export function listRunsForExperiment(experiment_id?: string, limit = 200): SqliteRunFull[] {
  const lim = Math.max(1, Math.min(Number(limit) || 200, 2000))
  let db: any | null = null
  try {
    db = tryOpenDb()
    if (!db) return []
    if (experiment_id) {
      const q = db.query(`
        SELECT
          id AS run,
          experiment_id,
          status,
          started_at,
          ended_at,
          git_hash,
          git_dirty,
          config_json,
          results_json
        FROM runs
        WHERE experiment_id = ?
        ORDER BY started_at DESC
        LIMIT ?
      `)
      return q.all(experiment_id, lim) as SqliteRunFull[]
    }
    const q = db.query(`
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty,
        config_json,
        results_json
      FROM runs
      ORDER BY started_at DESC
      LIMIT ?
    `)
    return q.all(lim) as SqliteRunFull[]
  } catch (e: any) {
    console.error(`[nviz] experiments.db unavailable: ${String(e)}`)
    return []
  } finally {
    try { db?.close?.() } catch {}
  }
}

export function getRun(run: string): SqliteRunFull | null {
  let db: any | null = null
  try {
    db = tryOpenDb()
    if (!db) return null
    const q = db.query(`
      SELECT
        id AS run,
        experiment_id,
        status,
        started_at,
        ended_at,
        git_hash,
        git_dirty,
        config_json,
        results_json
      FROM runs
      WHERE id = ?
      LIMIT 1
    `)
    const rows = q.all(run) as any[]
    return (rows && rows.length) ? (rows[0] as SqliteRunFull) : null
  } catch (e: any) {
    console.error(`[nviz] experiments.db unavailable: ${String(e)}`)
    return null
  } finally {
    try { db?.close?.() } catch {}
  }
}
