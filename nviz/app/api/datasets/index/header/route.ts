import { NextRequest, NextResponse } from "next/server"
import { indexHeaderStats } from "@/lib/server/datasets_inspect"

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path")
  if (!path) return NextResponse.json({ error: "Missing 'path'" }, { status: 400 })
  const sampleDocs = parseInt(req.nextUrl.searchParams.get("sample_docs") || "5000", 10)
  try {
    const out = await indexHeaderStats(path, { sample_docs: sampleDocs })
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

