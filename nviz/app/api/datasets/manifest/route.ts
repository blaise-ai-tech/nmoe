import { NextRequest, NextResponse } from "next/server"
import { datasetManifestPreview } from "@/lib/server/datasets_inspect"

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path") || ""
  const limitShards = Math.max(0, Math.min(parseInt(req.nextUrl.searchParams.get("limit_shards") || "200", 10), 2000))
  try {
    const out = datasetManifestPreview(path, limitShards)
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

