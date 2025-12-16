import { NextRequest, NextResponse } from 'next/server'
import { latestForPrefixes } from '@/lib/server/duckdb'

export async function GET(req: NextRequest) {
  const run = req.nextUrl.searchParams.get('run')
  if (!run) return NextResponse.json({ error: "Missing 'run'" }, { status: 400 })
  try {
    const vals = await latestForPrefixes(run, ['router/layer_', 'router_agg/'])
    const layersByIdx = new Map<number, any>()
    for (const [tag, value] of Object.entries(vals)) {
      if (tag.startsWith('router_agg/')) continue
      const m = tag.match(/^router\/layer_(\d{2})\/(cv|entropy|max_load|bias_range|experts_active)$/)
      if (!m) continue
      const idx = parseInt(m[1], 10)
      const key = m[2]
      if (!layersByIdx.has(idx)) layersByIdx.set(idx, { idx })
      layersByIdx.get(idx)![key] = value
    }
    const layers = Array.from(layersByIdx.values()).sort((a, b) => a.idx - b.idx)
    const agg: Record<string, number> = {}
    for (const [tag, value] of Object.entries(vals)) {
      if (!tag.startsWith('router_agg/')) continue
      agg[tag] = value
    }
    return NextResponse.json({ run, layers, agg }, { headers: { 'Cache-Control': 'private, max-age=5' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
