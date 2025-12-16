import { NextResponse } from 'next/server'
import { listExperimentSummaries } from '@/lib/server/experiments'

export async function GET() {
  try {
    const experiments = listExperimentSummaries()
    return NextResponse.json({ experiments }, { headers: { 'Cache-Control': 'private, max-age=5' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}

