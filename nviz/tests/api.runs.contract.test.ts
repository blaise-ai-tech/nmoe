import { describe, it, expect } from 'bun:test'
import { NextRequest } from 'next/server'
import { GET as runsHandler } from '@/app/api/runs/route'

import { mkdtempSync, rmSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import os from 'node:os'

import { DuckDBInstance } from '@duckdb/node-api'

describe('/api/runs contract (DuckDB + SQLite merge)', () => {
  it('includes sqlite status and duckdb last_step', async () => {
    const root = mkdtempSync(join(os.tmpdir(), 'nviz-fixture-'))
    const metricsDir = join(root, 'metrics')
    const expDb = join(root, 'experiments.db')

    process.env.NVIZ_METRICS_DIR = metricsDir
    process.env.NVIZ_EXPERIMENTS_DB = expDb

    // Create DuckDB metrics for runA.
    const runA = 'runA'
    mkdirSync(join(metricsDir, runA), { recursive: true })
    const dbPath = join(metricsDir, runA, 'rank_0.duckdb')
    {
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
        await conn.run(`INSERT OR REPLACE INTO metrics VALUES ('${runA}','train/loss',1,${now - 2000},10.0)`)
        await conn.run(`INSERT OR REPLACE INTO metrics VALUES ('${runA}','train/loss',2,${now - 1000},8.0)`)
      } finally {
        try { await conn.close() } catch {}
        try { await inst.close() } catch {}
      }
    }

    // Create SQLite experiments DB with runA + runB (no metrics).
    {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { Database } = require('bun:sqlite')
      const db = new Database(expDb)
      try {
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
          'exp1','exp','proj','',
        ])
        db.run(`INSERT OR REPLACE INTO runs(id,experiment_id,config_json,git_hash,git_dirty,status,started_at) VALUES (?,?,?,?,?,?,datetime('now'))`, [
          runA, 'exp1', '{}', 'deadbeef', 0, 'running',
        ])
        db.run(`INSERT OR REPLACE INTO runs(id,experiment_id,config_json,git_hash,git_dirty,status,started_at) VALUES (?,?,?,?,?,?,datetime('now','-1 hour'))`, [
          'runB', 'exp1', '{}', 'deadbeef', 0, 'completed',
        ])
      } finally {
        try { db.close() } catch {}
      }
    }

    const url = new URL('http://localhost/api/runs')
    const req = new NextRequest(url)
    const res = await runsHandler(req as any)
    const body = await res.json() as any
    if (!res.ok) throw new Error(body?.error ? String(body.error) : JSON.stringify(body))
    const runs = body.runs as any[]
    expect(Array.isArray(runs)).toBe(true)

    const a = runs.find(r => r.run === runA)
    expect(a).toBeTruthy()
    expect(a.status).toBe('running')
    expect(a.last_step).toBe(2)

    const b = runs.find(r => r.run === 'runB')
    expect(b).toBeTruthy()
    expect(b.status).toBe('completed')
    expect(b.last_step).toBe(0)
    expect(b.last_ts).toBe(0)

    try { rmSync(root, { recursive: true, force: true }) } catch {}
  })
})
