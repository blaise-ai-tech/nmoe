import { NextRequest, NextResponse } from "next/server"
import { tokenWindow } from "@/lib/server/datasets_inspect"

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get("path")
  if (!path) return NextResponse.json({ error: "Missing 'path'" }, { status: 400 })
  const offset = parseInt(req.nextUrl.searchParams.get("offset") || "0", 10)
  const length = parseInt(req.nextUrl.searchParams.get("length") || "256", 10)
  try {
    const out = await tokenWindow(path, offset, length)
    return NextResponse.json(out, { headers: { "Cache-Control": "private, max-age=2" } })
  } catch (e: any) {
    return NextResponse.json({ error: String(e) }, { status: 400 })
  }
}

