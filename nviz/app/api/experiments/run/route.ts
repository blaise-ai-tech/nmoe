import { NextRequest, NextResponse } from 'next/server'
import { getRun } from '@/lib/server/experiments'

export async function GET(req: NextRequest) {
  const run = req.nextUrl.searchParams.get('run')
  if (!run) return NextResponse.json({ error: "Missing 'run'" }, { status: 400 })
  try {
    const info = getRun(run)
    if (!info) return NextResponse.json({ error: 'Not found' }, { status: 404 })
    return NextResponse.json({ run: info }, { headers: { 'Cache-Control': 'private, max-age=5' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 500 })
  }
}

