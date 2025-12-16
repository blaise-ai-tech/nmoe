import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

import { DuckDBInstance } from '@duckdb/node-api'
import { allRuns, scalarsSampled } from '@/lib/server/duckdb'

let dir: string
let dbPath: string

beforeAll(() => {
  dir = mkdtempSync(join(os.tmpdir(), 'nviz-metrics-'))
  process.env.NVIZ_METRICS_DIR = dir
  mkdirSync(join(dir, 'runA'), { recursive: true })
  dbPath = join(dir, 'runA', 'rank_0.duckdb')
})

afterAll(() => {
  try { rmSync(dir, { recursive: true, force: true }) } catch {}
})

describe('duckdb layer', () => {
  beforeAll(async () => {
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
      const now = Date.now()
      await conn.run(`INSERT OR REPLACE INTO metrics VALUES ('runA','train/loss',1,${now-3000},10.0)`)
      await conn.run(`INSERT OR REPLACE INTO metrics VALUES ('runA','train/loss',2,${now-2000},7.5)`)
      await conn.run(`INSERT OR REPLACE INTO metrics VALUES ('runA','train/loss',3,${now-1000},6.2)`)
    } finally {
      try { await conn.close() } catch {}
      try { await inst.close() } catch {}
    }
  })

  it('lists runs sorted by recency', async () => {
    const runs = await allRuns()
    expect(runs.length).toBe(1)
    expect(runs[0].run).toBe('runA')
    expect(runs[0].last_step).toBe(3)
  })

  it('returns scalar series (sampled) preserving endpoints', async () => {
    const s = await scalarsSampled('runA','train/loss', 16)
    expect(s.length).toBeGreaterThanOrEqual(2)
    expect(s[0].step).toBe(1)
    expect(s[s.length-1].step).toBe(3)
  })
  
  it('requires NVIZ_METRICS_DIR to be set', async () => {
    const prev = process.env.NVIZ_METRICS_DIR
    // unset and expect throws when scanning
    // @ts-ignore
    delete process.env.NVIZ_METRICS_DIR
    await expect(async () => await allRuns()).toThrow()
    process.env.NVIZ_METRICS_DIR = prev
  })
})
