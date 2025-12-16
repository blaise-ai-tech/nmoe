// Server-only metrics readers backed by DuckDB.
//
// nmoe writes per-rank DuckDB metrics files:
//   /data/metrics/{run_id}/rank_{rank}.duckdb
//
// NVIZ attaches all rank DBs for a run (read-only) and queries them with
// window functions to dedupe and downsample efficiently.
import { readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { DuckDBInstance } from '@duckdb/node-api'

export type Row = { run: string; tag: string; step: number; ts_ms: number; value: number }
export type RunSummary = { run: string; last_ts: number; last_step: number }

function sqlLit(s: string): string {
  return `'${s.replaceAll("'", "''")}'`
}

export function metricsDir(): string {
  const dir = process.env.NVIZ_METRICS_DIR
  if (!dir) throw new Error('NVIZ_METRICS_DIR must be set')
  return dir
}

function safeRunDir(metricsRoot: string, run: string): string {
  // `run` is user-supplied via query params. Keep it strictly as a directory name.
  if (!run) throw new Error("Missing 'run'")
  if (!/^[A-Za-z0-9._-]+$/.test(run)) throw new Error('Invalid run')
  const root = resolve(metricsRoot)
  const target = resolve(join(root, run))
  if (target !== root && !target.startsWith(root + '/')) {
    throw new Error('Invalid run')
  }
  return target
}

function listRunDirs(dir: string): string[] {
  try {
    return readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
  } catch {
    return []
  }
}

function listDuckdbFilesForRun(dir: string, run: string): string[] {
  const runDir = safeRunDir(dir, run)
  try {
    const files = readdirSync(runDir, { withFileTypes: true })
      .filter((e) => e.isFile() && e.name.endsWith('.duckdb'))
      .map((e) => join(runDir, e.name))
    // Prefer rank_*.duckdb (ignore any other *.duckdb artifacts).
    return files.filter((p) => /\/rank_\d+\.duckdb$/.test(p)).sort()
  } catch {
    return []
  }
}

function representativeDuckdbFile(dir: string, run: string): string | null {
  const files = listDuckdbFilesForRun(dir, run)
  if (files.length === 0) return null
  const rank0 = files.find((p) => p.endsWith('/rank_0.duckdb'))
  return rank0 || files[0] || null
}

async function withRunAttached<T>(run: string, fn: (aliases: string[], q: (sql: string) => Promise<any[]>) => Promise<T>): Promise<T> {
  const dir = metricsDir()
  const files = listDuckdbFilesForRun(dir, run)
  if (files.length === 0) return await fn([], async () => [])

  const inst = await DuckDBInstance.create(':memory:')
  const conn = await inst.connect()
  try {
    const aliases: string[] = []
    for (let i = 0; i < files.length; i++) {
      const alias = `r${i}`
      aliases.push(alias)
      await conn.run(`ATTACH ${sqlLit(files[i])} AS ${alias} (READ_ONLY);`)
    }

    const q = async (sql: string): Promise<any[]> => {
      const reader = await conn.runAndReadAll(sql)
      const cols = reader.columnNames()
      const rows = reader.getRows()
      return rows.map((r: any[]) => Object.fromEntries(cols.map((c, i) => [c, r[i]])))
    }

    return await fn(aliases, q)
  } finally {
    try { (conn as any).close?.() } catch {}
    try { (inst as any).close?.() } catch {}
  }
}

function unionSql(aliases: string[], select: string): string {
  if (aliases.length === 0) return 'SELECT NULL AS run, NULL AS tag, NULL::INTEGER AS step, NULL::BIGINT AS ts_ms, NULL::DOUBLE AS value WHERE FALSE'
  return aliases.map((a) => select.replaceAll('$DB', a)).join('\nUNION ALL\n')
}

let allRunsCache: { at_ms: number; runs: RunSummary[] } | null = null

export async function allRuns(): Promise<RunSummary[]> {
  const now = Date.now()
  if (allRunsCache && (now - allRunsCache.at_ms) < 2000) {
    return allRunsCache.runs
  }

  const dir = metricsDir()
  const runs = listRunDirs(dir)
  const out: RunSummary[] = []
  for (const run of runs) {
    const file = representativeDuckdbFile(dir, run)
    if (!file) continue
    const inst = await DuckDBInstance.create(':memory:')
    const conn = await inst.connect()
    try {
      await conn.run(`ATTACH ${sqlLit(file)} AS r0 (READ_ONLY);`)
      const reader = await conn.runAndReadAll(`SELECT max(ts_ms) AS last_ts, max(step) AS last_step FROM r0.metrics`)
      const rows = reader.getRows()
      const last_ts = Number(rows?.[0]?.[0] ?? 0)
      const last_step = Number(rows?.[0]?.[1] ?? 0)
      out.push({ run, last_ts, last_step })
    } finally {
      try { (conn as any).close?.() } catch {}
      try { (inst as any).close?.() } catch {}
    }
  }
  out.sort((a, b) => b.last_ts - a.last_ts || b.last_step - a.last_step || a.run.localeCompare(b.run))
  allRunsCache = { at_ms: now, runs: out }
  return out
}

export async function latestForTags(run: string, tags: string[]): Promise<Record<string, number>> {
  if (tags.length === 0) return {}
  return await withRunAttached(run, async (aliases, q) => {
    const inList = tags.map(sqlLit).join(',')
    const u = unionSql(aliases, `SELECT run, tag, step, ts_ms, value FROM $DB.metrics WHERE run = ${sqlLit(run)} AND tag IN (${inList})`)
    const rows = await q(`
      WITH u AS (${u})
      SELECT tag, value
      FROM u
      QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1
    `)
    const out: Record<string, number> = {}
    for (const r of rows as any[]) out[String(r.tag)] = Number(r.value)
    return out
  })
}

export async function latestForPrefixes(run: string, prefixes: string[]): Promise<Record<string, number>> {
  if (prefixes.length === 0) return {}
  return await withRunAttached(run, async (aliases, q) => {
    const where = prefixes.map((p) => `tag LIKE ${sqlLit(p + '%')}`).join(' OR ')
    const u = unionSql(aliases, `SELECT run, tag, step, ts_ms, value FROM $DB.metrics WHERE run = ${sqlLit(run)} AND (${where})`)
    const rows = await q(`
      WITH u AS (${u})
      SELECT tag, value, ts_ms, step
      FROM u
      QUALIFY row_number() OVER (PARTITION BY tag ORDER BY ts_ms DESC, step DESC) = 1
    `)
    const out: Record<string, number> = {}
    for (const r of rows as any[]) out[String(r.tag)] = Number(r.value)
    return out
  })
}

export async function scalarsSampled(run: string, tag: string, buckets: number): Promise<Array<{ step: number; ts_ms: number; value: number }>> {
  const n = Math.max(8, Math.min(buckets, 20000))
  return await withRunAttached(run, async (aliases, q) => {
    const u = unionSql(aliases, `SELECT run, tag, step, ts_ms, value FROM $DB.metrics WHERE run = ${sqlLit(run)} AND tag = ${sqlLit(tag)}`)
    const rows = await q(`
      WITH u AS (${u}),
      dedup AS (
        SELECT step, ts_ms, value
        FROM u
        QUALIFY row_number() OVER (PARTITION BY step ORDER BY ts_ms DESC) = 1
      ),
      bucketed AS (
        SELECT step, ts_ms, value,
               ntile(${n}) OVER (ORDER BY step) AS b
        FROM dedup
      )
      SELECT step, ts_ms, value
      FROM bucketed
      QUALIFY
        row_number() OVER (PARTITION BY b ORDER BY step ASC) = 1
        OR row_number() OVER (PARTITION BY b ORDER BY step DESC) = 1
      ORDER BY step ASC
    `)
    return (rows as any[]).map((r) => ({ step: Number(r.step), ts_ms: Number(r.ts_ms), value: Number(r.value) }))
  })
}
