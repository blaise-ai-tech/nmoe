import { NextResponse } from 'next/server'
import { allRuns } from '@/lib/server/duckdb'
import { listRuns as listSqliteRuns } from '@/lib/server/experiments'

export async function GET(_req?: any) {
  try {
    const metricsRuns = await allRuns()
    const sqliteRuns = listSqliteRuns()

    const byId = new Map<string, any>()
    for (const r of metricsRuns) {
      byId.set(r.run, { ...r })
    }
    for (const r of sqliteRuns) {
      const cur = byId.get(r.run) || { run: r.run, last_ts: 0, last_step: 0 }
      byId.set(r.run, {
        ...cur,
        experiment_id: r.experiment_id,
        status: r.status,
        started_at: r.started_at,
        ended_at: r.ended_at ?? null,
      })
    }

    const runs = Array.from(byId.values())
    runs.sort((a, b) => {
      const ats = Number(a.last_ts || 0)
      const bts = Number(b.last_ts || 0)
      if (bts !== ats) return bts - ats
      const as = Date.parse(a.started_at || '') || 0
      const bs = Date.parse(b.started_at || '') || 0
      if (bs !== as) return bs - as
      const astep = Number(a.last_step || 0)
      const bstep = Number(b.last_step || 0)
      if (bstep !== astep) return bstep - astep
      return String(a.run).localeCompare(String(b.run))
    })

    return NextResponse.json({ runs }, { headers: { 'Cache-Control': 'private, max-age=2' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
