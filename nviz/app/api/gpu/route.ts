import { NextRequest, NextResponse } from 'next/server'
import { latestForPrefixes } from '@/lib/server/duckdb'

export async function GET(req: NextRequest) {
  const run = req.nextUrl.searchParams.get('run')
  if (!run) return NextResponse.json({ error: "Missing 'run'" }, { status: 400 })
  try {
    const vals = await latestForPrefixes(run, ['gpu/', 'gpu_agg/'])
    const keys = new Set([
      'utilization_gpu','memory_used_gib','memory_total_gib','power_draw_w','power_limit_w','temperature_c','clocks_sm_mhz',
      'throttle_thermal','throttle_power','throttle_hw_slowdown','throttle_apps','ecc_corrected','ecc_uncorrected'
    ])
    const gpusByIdx = new Map<number, any>()
    for (const [tag, value] of Object.entries(vals)) {
      if (tag.startsWith('gpu_agg/')) continue
      const m = tag.match(/^gpu\/(\d{1,2})\/([A-Za-z0-9_]+)$/)
      if (!m) continue
      const idx = parseInt(m[1], 10)
      const key = m[2]
      if (!keys.has(key)) continue
      if (!gpusByIdx.has(idx)) gpusByIdx.set(idx, { index: idx })
      gpusByIdx.get(idx)![key] = value
    }
    const gpus = Array.from(gpusByIdx.values()).sort((a, b) => a.index - b.index)
    const agg: Record<string, number> = {}
    for (const [tag, value] of Object.entries(vals)) {
      if (!tag.startsWith('gpu_agg/')) continue
      agg[tag] = value
    }
    return NextResponse.json({ run, gpus, agg }, { headers: { 'Cache-Control': 'private, max-age=2' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
