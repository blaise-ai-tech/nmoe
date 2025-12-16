import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export type ParsedRun = {
  raw: string
  base: string
  date?: Date
  batch?: number
  seqlen?: number
  config?: string
}

// Parse run names like: "nmoe/20251106-114542_b8_s4096_stream"
export function parseRunName(run: string): ParsedRun {
  const base = run.split("/").pop() || run
  if (run === ".") {
    return { raw: run, base: "Current Run" }
  }
  const m = base.match(/^(\d{8})-(\d{6})_b(\d+)_s(\d+)(?:_(.+))?$/)
  if (!m) {
    return { raw: run, base }
  }
  const [, ymd, hms, b, s, cfg] = m
  const y = Number(ymd.slice(0, 4))
  const mo = Number(ymd.slice(4, 6)) - 1
  const d = Number(ymd.slice(6, 8))
  const hh = Number(hms.slice(0, 2))
  const mm = Number(hms.slice(2, 4))
  const ss = Number(hms.slice(4, 6))
  const dt = new Date(Date.UTC(y, mo, d, hh, mm, ss))
  return {
    raw: run,
    base,
    date: dt,
    batch: Number(b),
    seqlen: Number(s),
    config: cfg,
  }
}

export function formatRunLabel(run: string, opts?: { compact?: boolean }): string {
  const { compact } = opts || {}
  if (run === ".") return "Current Run"
  const p = parseRunName(run)
  if (!p.date || !p.batch || !p.seqlen) {
    return p.base
  }
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
  const y = p.date.getUTCFullYear()
  const m = months[p.date.getUTCMonth()]
  const d = p.date.getUTCDate()
  const hh = String(p.date.getUTCHours()).padStart(2, '0')
  const mm = String(p.date.getUTCMinutes()).padStart(2, '0')
  const dateStr = compact ? `${m} ${d} ${hh}:${mm} UTC` : `${m} ${d}, ${y} ${hh}:${mm} UTC`
  const cfg = p.config ? ` • ${p.config}` : ""
  return `${dateStr} • b${p.batch} • s${p.seqlen}${cfg}`
}
