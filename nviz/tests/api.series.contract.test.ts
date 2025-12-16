import { describe, it, expect } from 'bun:test'
import { NextRequest } from 'next/server'
import { GET as seriesHandler } from '@/app/api/series/route'

import { mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'
import { DuckDBInstance } from '@duckdb/node-api'

describe('/api/series contract', () => {
  it('respects points cap and preserves first/last', async () => {
    const dir = mkdtempSync(join(os.tmpdir(), 'nviz-metrics-'))
    process.env.NVIZ_METRICS_DIR = dir
    mkdirSync(join(dir, 'runA'), { recursive: true })
    const dbPath = join(dir, 'runA', 'rank_0.duckdb')

    const inst = await DuckDBInstance.create(dbPath)
    const conn = await inst.connect()
    try {
      await conn.run(`
        CREATE TABLE IF NOT EXISTS metrics (
          run   TEXT NOT NULL,
          tag   TEXT NOT NULL,
          step  INTEGER NOT NULL,
          ts_ms BIGINT NOT NULL,
          value DOUBLE NOT NULL,
          PRIMARY KEY (run, tag, step)
        );
      `)
      // 10k points fast-path via DuckDB range + scalar math.
      await conn.run(`
        INSERT OR REPLACE INTO metrics
        SELECT
          'runA' AS run,
          'train/loss' AS tag,
          i AS step,
          i AS ts_ms,
          sin(i::DOUBLE / 50.0) + i::DOUBLE * 1e-3 AS value
        FROM range(1, 10001) tbl(i)
      `)
    } finally {
      try { await conn.close() } catch {}
      try { await inst.close() } catch {}
    }

    const url = new URL('http://localhost/api/series')
    url.searchParams.set('run','runA')
    url.searchParams.set('tag','train/loss')
    url.searchParams.set('points','500')
    const req = new NextRequest(url)
    const res = await seriesHandler(req)
    const body = await res.json() as any
    if (!res.ok) {
      throw new Error(body?.error ? String(body.error) : JSON.stringify(body))
    }
    const series = body.series as Array<{ step:number, value:number, ts_ms:number }>
    expect(Array.isArray(series)).toBe(true)
    expect(series.length).toBeLessThanOrEqual(500)
    expect(series[0].step).toBe(1)
    expect(series[series.length-1].step).toBe(10000)
    for (let i=1;i<series.length;i++) expect(series[i].step).toBeGreaterThan(series[i-1].step)
    try { rmSync(dir, { recursive: true, force: true }) } catch {}
  })
})
