// Server-only dataset browser.
//
// Contract:
// - Read-only
// - Paths are relative to NVIZ_DATASETS_DIR (default: /data)
// - No path traversal allowed

import { createReadStream, readdirSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

export type DatasetEntry = {
  name: string
  kind: 'dir' | 'file'
  size: number
  mtime_ms: number
}

export function datasetsRoot(): string {
  return process.env.NVIZ_DATASETS_DIR || '/data'
}

export function safeResolve(relPath: string): string {
  const root = resolve(datasetsRoot())
  const target = resolve(join(root, relPath || '.'))
  if (target !== root && !target.startsWith(root + '/')) {
    throw new Error('Invalid path')
  }
  return target
}

export function listDir(relPath: string): { path: string; entries: DatasetEntry[] } {
  const abs = safeResolve(relPath)
  const entries = readdirSync(abs, { withFileTypes: true })
    .filter((e) => e.name !== '.' && e.name !== '..')
    .map((e) => {
      const st = statSync(join(abs, e.name))
      return {
        name: e.name,
        kind: e.isDirectory() ? 'dir' : 'file',
        size: Number(st.size || 0),
        mtime_ms: Number(st.mtimeMs || 0),
      } satisfies DatasetEntry
    })
    .sort((a, b) => {
      if (a.kind !== b.kind) return a.kind === 'dir' ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  return { path: relPath || '', entries }
}

export async function previewFile(relPath: string, maxBytes = 64 * 1024): Promise<{ path: string; bytes: number; text: string }> {
  const abs = safeResolve(relPath)
  const st = statSync(abs)
  if (!st.isFile()) throw new Error('Not a file')
  const cap = Math.max(1024, Math.min(maxBytes, 1024 * 1024))
  const chunks: Buffer[] = []
  let read = 0
  await new Promise<void>((resolveP, rejectP) => {
    const s = createReadStream(abs, { start: 0, end: cap - 1 })
    s.on('data', (b) => {
      chunks.push(b as Buffer)
      read += (b as Buffer).length
    })
    s.on('error', rejectP)
    s.on('end', resolveP)
  })
  const buf = Buffer.concat(chunks)

  // Heuristic: if most bytes are printable UTF-8-ish, treat as text; else show hex.
  // We keep this simple and deterministic; the viewer is for humans, not parsers.
  let printable = 0
  for (const b of buf) {
    if (b === 9 || b === 10 || b === 13) { printable++; continue } // \t \n \r
    if (b >= 32 && b <= 126) printable++
  }
  const ratio = buf.length ? (printable / buf.length) : 1.0
  if (ratio >= 0.8) {
    return { path: relPath || '', bytes: read, text: buf.toString('utf8') }
  }
  const hex = buf.subarray(0, Math.min(buf.length, 256)).toString('hex').replace(/(.{2})/g, '$1 ').trim()
  return {
    path: relPath || '',
    bytes: read,
    text: `[binary] first ${Math.min(buf.length, 256)} bytes (hex):\n${hex}\n`,
  }
}
