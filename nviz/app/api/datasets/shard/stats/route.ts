import { NextRequest, NextResponse } from "next/server"
import { tokenShardStats } from "@/lib/server/datasets_inspect"

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path")
  if (!path) return NextResponse.json({ error: "Missing 'path'" }, { status: 400 })

  const vocab = req.nextUrl.searchParams.get("vocab_size")
  const eos = req.nextUrl.searchParams.get("eos_token_id")
  const scan = req.nextUrl.searchParams.get("scan_tokens")
  const sample = req.nextUrl.searchParams.get("sample_tokens")
  const topK = req.nextUrl.searchParams.get("top_k")

  try {
    const out = await tokenShardStats(path, {
      vocab_size: vocab ? parseInt(vocab, 10) : undefined,
      eos_token_id: eos ? parseInt(eos, 10) : undefined,
      scan_tokens: scan ? parseInt(scan, 10) : undefined,
      sample_tokens: sample ? parseInt(sample, 10) : undefined,
      top_k: topK ? parseInt(topK, 10) : undefined,
    })
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

