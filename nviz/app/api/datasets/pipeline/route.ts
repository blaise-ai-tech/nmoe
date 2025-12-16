import { NextRequest, NextResponse } from "next/server"
import { inspectFlowDir } from "@/lib/server/datasets_pipeline"

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path") || ""
  try {
    const out = inspectFlowDir(path)
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

