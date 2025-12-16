import { NextRequest, NextResponse } from "next/server"
import { datasetsMeta } from "@/lib/server/datasets_inspect"

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path") || ""
  try {
    const out = datasetsMeta(path)
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

