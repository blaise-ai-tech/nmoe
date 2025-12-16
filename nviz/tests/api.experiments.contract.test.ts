import { describe, it, expect } from 'bun:test'
import { NextRequest } from 'next/server'
import { GET as experimentsHandler } from '@/app/api/experiments/route'
import { GET as runsHandler } from '@/app/api/experiments/runs/route'

import { mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'
import { DuckDBInstance } from '@duckdb/node-api'

describe('/api/experiments endpoints', () => {
  it('lists experiment summaries and runs', async () => {
    const root = mkdtempSync(join(os.tmpdir(), 'nviz-exp-'))
    const metricsDir = join(root, 'metrics')
    const expDb = join(root, 'experiments.db')
    process.env.NVIZ_METRICS_DIR = metricsDir
    process.env.NVIZ_EXPERIMENTS_DB = expDb

    // SQLite seed.
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { Database } = require('bun:sqlite')
    const db = new Database(expDb)
    db.exec(`
      CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        project TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
    `)
    db.exec(`
      CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        config_json TEXT NOT NULL,
        git_hash TEXT,
        git_dirty INTEGER DEFAULT 0,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at TEXT,
        status TEXT NOT NULL DEFAULT 'running',
        results_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id)
      );
    `)
    db.run(`INSERT OR REPLACE INTO experiments(id,name,project,description,created_at) VALUES (?,?,?,?,datetime('now'))`, [
      'expA','A','proj','', 
    ])
    db.run(`INSERT OR REPLACE INTO runs(id,experiment_id,config_json,git_hash,git_dirty,status,started_at) VALUES (?,?,?,?,?,?,datetime('now'))`, [
      'runA','expA','{}','deadbeef',0,'completed'
    ])
    db.close()

    // DuckDB seed for runA.
    mkdirSync(join(metricsDir, 'runA'), { recursive: true })
    const dbPath = join(metricsDir, 'runA', 'rank_0.duckdb')
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
      await conn.run(`INSERT OR REPLACE INTO metrics VALUES ('runA','train/loss',5,${now},3.14)`)
    } finally {
      try { await conn.close() } catch {}
      try { await inst.close() } catch {}
    }

    {
      const req = new NextRequest(new URL('http://localhost/api/experiments'))
      const res = await experimentsHandler(req as any)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.experiments.length).toBe(1)
      expect(body.experiments[0].id).toBe('expA')
      expect(body.experiments[0].runs_total).toBe(1)
    }

    {
      const url = new URL('http://localhost/api/experiments/runs')
      url.searchParams.set('experiment_id','expA')
      const req = new NextRequest(url)
      const res = await runsHandler(req)
      const body = await res.json() as any
      expect(res.ok).toBe(true)
      expect(body.runs.length).toBe(1)
      expect(body.runs[0].run).toBe('runA')
      expect(body.runs[0].last_step).toBe(5)
    }

    try { rmSync(root, { recursive: true, force: true }) } catch {}
  })
})

