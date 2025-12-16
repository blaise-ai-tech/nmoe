import { NextRequest, NextResponse } from 'next/server'
import { scalarsSampled } from '@/lib/server/duckdb'

type Pt = { step: number; value: number; ts_ms: number }

function lttb(series: Pt[], threshold: number): Pt[] {
  if (threshold >= series.length || threshold === 0) return series
  const sampled: Pt[] = [series[0]]
  const bucketSize = (series.length - 2) / (threshold - 2)
  let a = 0
  for (let i = 0; i < threshold - 2; i++) {
    const rangeStart = Math.floor((i + 1) * bucketSize) + 1
    const rangeEnd = Math.floor((i + 2) * bucketSize) + 1
    const avgRangeStart = Math.floor((i + 2) * bucketSize) + 1
    const avgRangeEnd = Math.floor((i + 3) * bucketSize) + 1
    let avgX = 0, avgY = 0
    const next = series.slice(avgRangeStart, Math.min(avgRangeEnd, series.length))
    if (next.length) {
      for (const p of next) { avgX += p.step; avgY += p.value }
      avgX /= next.length; avgY /= next.length
    } else {
      avgX = series[series.length - 1].step; avgY = series[series.length - 1].value
    }
    let maxArea = -1, nextA = Math.min(rangeStart, series.length - 2)
    const jStart = Math.min(Math.max(rangeStart, 1), series.length - 2)
    const jEnd = Math.min(Math.max(rangeEnd, jStart + 1), series.length - 1)
    for (let j = jStart; j < jEnd; j++) {
      const pa = series[a], pb = series[j]
      const area = Math.abs((pa.step - avgX) * (pb.value - pa.value) - (pa.step - pb.step) * (avgY - pa.value)) * 0.5
      if (area > maxArea) { maxArea = area; nextA = j }
    }
    sampled.push(series[nextA])
    a = nextA
  }
  sampled.push(series[series.length - 1])
  return sampled
}

export async function GET(req: NextRequest) {
  const run = req.nextUrl.searchParams.get('run')
  const tag = req.nextUrl.searchParams.get('tag')
  const points = parseInt(req.nextUrl.searchParams.get('points') || '1000', 10)
  if (!run || !tag) return NextResponse.json({ error: "Missing 'run' or 'tag'" }, { status: 400 })
  try {
    // Sample in SQL (windowed) to keep payload bounded, then LTTB on the client shape.
    const raw = await scalarsSampled(run, tag, Math.max(64, Math.min(points * 4, 20000)))
    const series = raw.map(r => ({ step: r.step, ts_ms: r.ts_ms, value: r.value }))
    const ds = lttb(series, Math.max(200, Math.min(points, 5000)))
    return NextResponse.json({ run, tag, series: ds }, { headers: { 'Cache-Control': 'private, max-age=2' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
