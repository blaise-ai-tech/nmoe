import { NextRequest, NextResponse } from 'next/server'
import { previewFile } from '@/lib/server/datasets'

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get('path')
  if (!path) return NextResponse.json({ error: "Missing 'path'" }, { status: 400 })
  const maxBytes = Math.max(1024, Math.min(parseInt(req.nextUrl.searchParams.get('max_bytes') || '65536', 10), 1024 * 1024))
  try {
    const out = await previewFile(path, maxBytes)
    return NextResponse.json(out, { headers: { 'Cache-Control': 'private, max-age=2' } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

