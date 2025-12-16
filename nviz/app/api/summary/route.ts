import { NextRequest, NextResponse } from 'next/server'
import { latestForTags } from '@/lib/server/duckdb'

export async function GET(req: NextRequest) {
  const run = req.nextUrl.searchParams.get('run')
  if (!run) return NextResponse.json({ error: "Missing 'run'" }, { status: 400 })
  try {
    const tags = [
      'train/loss','optimizer/lr','throughput/tokens_per_s_gpu','throughput/ms_per_step','efficiency/tflops',
      'efficiency/fp8_tflops','efficiency/bf16_tflops',
      'gpu_agg/mean_utilization_gpu','gpu_agg/total_memory_used_gib','gpu_agg/total_power_w','gpu_agg/max_temperature_c'
    ]
    const s = await latestForTags(run, tags)
    return NextResponse.json({ run, summary: s }, { headers: { 'Cache-Control': 'private, max-age=2' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}
