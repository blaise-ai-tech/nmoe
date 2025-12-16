import { NextRequest, NextResponse } from "next/server"
import { indexDocView } from "@/lib/server/datasets_inspect"

export async function GET(req: NextRequest) {
  const shardPath = req.nextUrl.searchParams.get("shard_path")
  if (!shardPath) return NextResponse.json({ error: "Missing 'shard_path'" }, { status: 400 })
  const docIdx = parseInt(req.nextUrl.searchParams.get("doc_idx") || "0", 10)
  const maxTokens = parseInt(req.nextUrl.searchParams.get("max_tokens") || "4096", 10)
  try {
    const out = await indexDocView(shardPath, docIdx, { max_tokens: maxTokens })
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

