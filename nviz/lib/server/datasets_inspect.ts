import { existsSync, openSync, readFileSync, readSync, closeSync, statSync } from "node:fs"
import { open } from "node:fs/promises"
import { extname, join, dirname } from "node:path"

import { safeResolve } from "@/lib/server/datasets"

type NpyHeader = {
  descr: string
  fortran_order: boolean
  shape: number[]
  header_len: number
  data_offset: number
}

export type DatasetsMeta = {
  path: string
  kind: "dir" | "file"
  size: number
  mtime_ms: number
}

export type DatasetManifestPreview = {
  path: string
  manifest_path: string
  found: boolean
  dataset?: string
  version?: string
  tokenizer?: string
  vocab_size?: number
  eos_token_id?: number
  total_tokens?: number
  total_documents?: number
  num_shards?: number
  shards_preview?: Array<{
    path: string
    index_path: string
    shard_rel: string
    index_rel: string
    num_tokens: number
    num_documents: number
    checksum: string
  }>
}

export type TokenShardStats = {
  path: string
  kind: "npy_uint32_1d" | "raw_u32"
  tokens: number
  bytes: number
  min: number | null
  max: number | null
  eos_token_id: number | null
  vocab_size: number | null
  scan_tokens: number
  eos_count: number | null
  double_eos_count: number | null
  repetition_ratio: number | null
  out_of_range_count: number | null
  out_of_range_ratio: number | null
  top_tokens: Array<{ token: number; count: number }>
  has_index: boolean
}

export type TokenWindow = {
  path: string
  offset: number
  length: number
  tokens: number[]
}

export type IndexHeaderStats = {
  path: string
  version: number
  num_docs: number
  doc_len: {
    min: number | null
    p50: number | null
    p90: number | null
    p99: number | null
    max: number | null
  }
  histogram: Array<{ bucket: string; count: number }>
}

export type IndexDocView = {
  idx_path: string
  shard_path: string
  doc_idx: number
  start: number
  end: number
  length: number
  tokens: number[]
}

export function datasetsMeta(relPath: string): DatasetsMeta {
  const abs = safeResolve(relPath)
  const st = statSync(abs)
  return {
    path: relPath || "",
    kind: st.isDirectory() ? "dir" : "file",
    size: Number(st.size || 0),
    mtime_ms: Number(st.mtimeMs || 0),
  }
}

function tryReadJsonFile(absPath: string): any | null {
  try {
    return JSON.parse(readFileSync(absPath, "utf8"))
  } catch {
    return null
  }
}

function manifestPathForDir(dirRelPath: string): { rel: string; abs: string } {
  const rel = join(dirRelPath || "", "manifest.json")
  const abs = safeResolve(rel)
  return { rel, abs }
}

export function datasetManifestPreview(dirRelPath: string, limitShards = 200): DatasetManifestPreview {
  const { rel, abs } = manifestPathForDir(dirRelPath)
  if (!existsSync(abs)) {
    return { path: dirRelPath || "", manifest_path: rel, found: false }
  }
  const j = tryReadJsonFile(abs)
  if (!j) {
    return { path: dirRelPath || "", manifest_path: rel, found: false }
  }

  const shards: any[] = Array.isArray(j.shards) ? j.shards : []
  const normalizeRel = (p: string): string => {
    const raw = String(p || "")
    if (!raw) return ""
    if (raw.startsWith("/")) {
      // Some producers may emit absolute paths; for NVIZ we only support inspection
      // under the manifest directory, so we normalize to basename.
      const base = raw.split("/").filter(Boolean).slice(-1)[0] || ""
      return join(dirRelPath || "", base)
    }
    return join(dirRelPath || "", raw)
  }

  const shards_preview = shards.slice(0, Math.max(0, Math.min(limitShards, 2000))).map((s) => {
    const shardPath = String(s.path || "")
    const idxPath = String(s.index_path || "")
    return {
      path: shardPath,
      index_path: idxPath,
      shard_rel: normalizeRel(shardPath),
      index_rel: normalizeRel(idxPath),
      num_tokens: Number(s.num_tokens || 0),
      num_documents: Number(s.num_documents || 0),
      checksum: String(s.checksum || ""),
    }
  })

  return {
    path: dirRelPath || "",
    manifest_path: rel,
    found: true,
    dataset: typeof j.dataset === "string" ? j.dataset : undefined,
    version: typeof j.version === "string" ? j.version : undefined,
    tokenizer: typeof j.tokenizer === "string" ? j.tokenizer : undefined,
    vocab_size: Number.isFinite(j.vocab_size) ? Number(j.vocab_size) : undefined,
    eos_token_id: Number.isFinite(j.eos_token_id) ? Number(j.eos_token_id) : undefined,
    total_tokens: Number.isFinite(j.total_tokens) ? Number(j.total_tokens) : undefined,
    total_documents: Number.isFinite(j.total_documents) ? Number(j.total_documents) : undefined,
    num_shards: Number.isFinite(j.num_shards) ? Number(j.num_shards) : undefined,
    shards_preview,
  }
}

function fnv1a32(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

async function readBytes(absPath: string, offset: number, length: number): Promise<Buffer> {
  const fh = await open(absPath, "r")
  try {
    const buf = Buffer.alloc(length)
    const { bytesRead } = await fh.read(buf, 0, length, offset)
    return bytesRead === length ? buf : buf.subarray(0, bytesRead)
  } finally {
    await fh.close()
  }
}

function parseNpyHeaderFromBytes(prefix: Buffer): { versionMajor: number; versionMinor: number; headerLenSize: 2 | 4; headerLen: number; headerStart: number } {
  if (prefix.length < 12) throw new Error("Invalid .npy: truncated header")
  const magic = prefix.subarray(0, 6)
  if (magic.toString("latin1") !== "\u0093NUMPY") throw new Error("Invalid .npy: bad magic")
  const versionMajor = prefix[6]
  const versionMinor = prefix[7]
  const headerLenSize: 2 | 4 = versionMajor <= 1 ? 2 : 4
  const headerStart = 6 + 2 + headerLenSize
  const headerLen = headerLenSize === 2 ? prefix.readUInt16LE(8) : prefix.readUInt32LE(8)
  if (!Number.isFinite(headerLen) || headerLen <= 0) throw new Error("Invalid .npy: bad header len")
  return { versionMajor, versionMinor, headerLenSize, headerLen, headerStart }
}

function parseNpyHeaderDict(s: string): Pick<NpyHeader, "descr" | "fortran_order" | "shape"> {
  const descr = /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(s)?.[1]
  const fortran = /['"]fortran_order['"]\s*:\s*(True|False)/.exec(s)?.[1]
  const shapeRaw = /['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(s)?.[1]
  if (!descr || !fortran || shapeRaw === undefined) {
    const snippet = s.length > 240 ? `${s.slice(0, 240)}…` : s
    throw new Error(`Invalid .npy: unsupported header dict: ${snippet}`)
  }
  const parts = shapeRaw
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
  const shape = parts.map((p) => parseInt(p, 10)).filter((n) => Number.isFinite(n) && n >= 0)
  if (!shape.length) throw new Error("Invalid .npy: missing shape")
  return { descr, fortran_order: fortran === "True", shape }
}

async function readNpyHeader(absPath: string): Promise<NpyHeader> {
  const prefix = await readBytes(absPath, 0, 64)
  const { headerLen, headerStart } = parseNpyHeaderFromBytes(prefix.length >= 12 ? prefix : Buffer.concat([prefix, Buffer.alloc(12 - prefix.length)]))
  const headerBuf = await readBytes(absPath, headerStart, headerLen)
  const headerStr = headerBuf.toString("latin1").trim()
  const dict = parseNpyHeaderDict(headerStr)
  return {
    ...dict,
    header_len: headerLen,
    data_offset: headerStart + headerLen,
  }
}

function assertU32Header(h: NpyHeader): void {
  if (h.fortran_order) throw new Error("Unsupported .npy: fortran_order=True")
  if (h.shape.length !== 1) throw new Error("Unsupported .npy: expected 1D array")
  if (h.descr !== "<u4" && h.descr !== "|u4") throw new Error(`Unsupported .npy dtype: ${h.descr} (expected uint32)`)
}

async function u32Window(absPath: string, dataOffset: number, tokenOffset: number, tokenLen: number): Promise<Uint32Array> {
  const bytes = Math.max(0, tokenLen) * 4
  const buf = await readBytes(absPath, dataOffset + tokenOffset * 4, bytes)
  const out = new Uint32Array(Math.floor(buf.length / 4))
  for (let i = 0; i < out.length; i++) {
    out[i] = buf.readUInt32LE(i * 4)
  }
  return out
}

function inferManifestNearFile(relPath: string): { vocab_size: number | null; eos_token_id: number | null } {
  try {
    const dir = dirname(relPath)
    const m = datasetManifestPreview(dir, 0)
    if (!m.found) return { vocab_size: null, eos_token_id: null }
    const vocab = Number.isFinite(m.vocab_size) ? Number(m.vocab_size) : null
    const eos = Number.isFinite(m.eos_token_id) ? Number(m.eos_token_id) : null
    return { vocab_size: vocab, eos_token_id: eos }
  } catch {
    return { vocab_size: null, eos_token_id: null }
  }
}

export async function tokenShardStats(
  relPath: string,
  opts?: {
    vocab_size?: number
    eos_token_id?: number
    scan_tokens?: number
    sample_tokens?: number
    top_k?: number
  }
): Promise<TokenShardStats> {
  const abs = safeResolve(relPath)
  const st = statSync(abs)
  if (!st.isFile()) throw new Error("Not a file")

  const ext = extname(relPath).toLowerCase()
  let kind: TokenShardStats["kind"]
  let dataOffset = 0
  let tokens = 0

  if (ext === ".npy") {
    const h = await readNpyHeader(abs)
    assertU32Header(h)
    kind = "npy_uint32_1d"
    dataOffset = h.data_offset
    tokens = Number(h.shape[0] || 0)
  } else {
    kind = "raw_u32"
    if ((st.size % 4) !== 0) throw new Error("Invalid raw shard: file size not divisible by 4")
    tokens = Math.floor(st.size / 4)
  }

  const inferred = inferManifestNearFile(relPath)
  const vocab_size = Number.isFinite(opts?.vocab_size) ? Number(opts?.vocab_size) : inferred.vocab_size
  const eos_token_id = Number.isFinite(opts?.eos_token_id) ? Number(opts?.eos_token_id) : inferred.eos_token_id

  const scanCap = Math.max(4096, Math.min(Number.isFinite(opts?.scan_tokens) ? Number(opts?.scan_tokens) : 250_000, 2_000_000))
  const scanTokens = Math.min(tokens, scanCap)
  const scan = scanTokens > 0 ? await u32Window(abs, dataOffset, 0, scanTokens) : new Uint32Array(0)

  let min: number | null = null
  let max: number | null = null
  let eosCount: number | null = eos_token_id == null ? null : 0
  let doubleEos: number | null = eos_token_id == null ? null : 0
  let rep: number | null = scan.length >= 2 ? 0 : null

  for (let i = 0; i < scan.length; i++) {
    const v = scan[i]!
    if (min == null || v < min) min = v
    if (max == null || v > max) max = v
    if (eosCount != null && eos_token_id != null && v === eos_token_id) eosCount++
    if (doubleEos != null && eos_token_id != null && i > 0 && scan[i - 1] === eos_token_id && v === eos_token_id) doubleEos++
    if (rep != null && i > 0 && scan[i - 1] === v) rep++
  }

  const repetition_ratio = rep == null ? null : (scan.length >= 2 ? rep / (scan.length - 1) : 0)

  const sampleCap = Math.max(4096, Math.min(Number.isFinite(opts?.sample_tokens) ? Number(opts?.sample_tokens) : 65_536, 1_000_000))
  const sampleTokens = Math.min(tokens, sampleCap)
  const topK = Math.max(1, Math.min(Number.isFinite(opts?.top_k) ? Number(opts?.top_k) : 32, 256))

  const rng = mulberry32(fnv1a32(relPath))
  const blockLen = Math.min(2048, Math.max(256, Math.floor(sampleTokens / 64) || 256))
  const blocks = Math.max(1, Math.min(256, Math.ceil(sampleTokens / blockLen)))

  const freq = new Map<number, number>()
  let outOfRange = vocab_size == null ? null : 0
  let observed = 0
  for (let b = 0; b < blocks; b++) {
    if (tokens === 0) break
    const maxStart = Math.max(0, tokens - blockLen)
    const start = maxStart > 0 ? Math.floor(rng() * maxStart) : 0
    const chunk = await u32Window(abs, dataOffset, start, Math.min(blockLen, tokens - start))
    for (let i = 0; i < chunk.length; i++) {
      const v = chunk[i]!
      freq.set(v, (freq.get(v) || 0) + 1)
      if (outOfRange != null && vocab_size != null && v >= vocab_size) outOfRange++
    }
    observed += chunk.length
  }

  const top = Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([token, count]) => ({ token, count }))

  const out_of_range_ratio = outOfRange == null ? null : (observed ? (outOfRange / observed) : 0)

  const idxRel = relPath.replace(/\.(npy|bin)$/i, ".idx")
  let hasIndex = false
  try {
    const idxAbs = safeResolve(idxRel)
    hasIndex = existsSync(idxAbs)
  } catch {
    hasIndex = false
  }

  return {
    path: relPath || "",
    kind,
    tokens,
    bytes: Number(st.size || 0),
    min,
    max,
    eos_token_id,
    vocab_size,
    scan_tokens: scanTokens,
    eos_count: eosCount,
    double_eos_count: doubleEos,
    repetition_ratio,
    out_of_range_count: outOfRange,
    out_of_range_ratio,
    top_tokens: top,
    has_index: hasIndex,
  }
}

export async function tokenWindow(relPath: string, offset: number, length: number): Promise<TokenWindow> {
  const abs = safeResolve(relPath)
  const st = statSync(abs)
  if (!st.isFile()) throw new Error("Not a file")

  const ext = extname(relPath).toLowerCase()
  let dataOffset = 0
  let tokens = 0
  if (ext === ".npy") {
    const h = await readNpyHeader(abs)
    assertU32Header(h)
    dataOffset = h.data_offset
    tokens = Number(h.shape[0] || 0)
  } else {
    if ((st.size % 4) !== 0) throw new Error("Invalid raw shard: file size not divisible by 4")
    tokens = Math.floor(st.size / 4)
  }

  const off = Math.max(0, Math.min(Math.floor(offset), Math.max(0, tokens)))
  const maxLen = Math.max(1, Math.min(Math.floor(length), 4096))
  const end = Math.min(tokens, off + maxLen)
  const arr = await u32Window(abs, dataOffset, off, Math.max(0, end - off))
  return { path: relPath || "", offset: off, length: arr.length, tokens: Array.from(arr) }
}

function idxHeader(absIdxPath: string): { version: number; numDocs: number } {
  const fd = openSync(absIdxPath, "r")
  try {
    const hdr = Buffer.alloc(32)
    const bytesRead = readSync(fd, hdr, 0, 32, 0)
    if (bytesRead !== 32) throw new Error("Invalid .idx: truncated header")
    const magic = hdr.subarray(0, 8).toString("latin1")
    if (magic !== "NMOEIDX\u0000") throw new Error("Invalid .idx: bad magic")
    const version = Number(hdr.readBigUInt64LE(8))
    const numDocs = Number(hdr.readBigUInt64LE(16))
    if (!Number.isFinite(version) || !Number.isFinite(numDocs)) throw new Error("Invalid .idx: bad header")
    return { version, numDocs }
  } finally {
    closeSync(fd)
  }
}

function percentile(sorted: number[], p: number): number | null {
  if (!sorted.length) return null
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))
  return sorted[idx] ?? null
}

export async function indexHeaderStats(relIdxPath: string, opts?: { sample_docs?: number }): Promise<IndexHeaderStats> {
  const abs = safeResolve(relIdxPath)
  const st = statSync(abs)
  if (!st.isFile()) throw new Error("Not a file")

  const { version, numDocs } = idxHeader(abs)
  const sampleDocs = Math.max(0, Math.min(Number.isFinite(opts?.sample_docs) ? Number(opts?.sample_docs) : 5000, 50000))
  const take = Math.min(numDocs, sampleDocs)

  const lens: number[] = []
  if (take > 0) {
    const buf = await readBytes(abs, 32, take * 16)
    const pairs = Math.floor(buf.length / 16)
    for (let i = 0; i < pairs; i++) {
      const start = Number(buf.readBigUInt64LE(i * 16))
      const end = Number(buf.readBigUInt64LE(i * 16 + 8))
      const len = Math.max(0, end - start)
      lens.push(len)
    }
  }
  lens.sort((a, b) => a - b)

  const min = lens.length ? lens[0]! : null
  const max = lens.length ? lens[lens.length - 1]! : null
  const p50 = percentile(lens, 0.5)
  const p90 = percentile(lens, 0.9)
  const p99 = percentile(lens, 0.99)

  // Simple histogram buckets by doc length.
  const buckets = [
    { label: "0-32", lo: 0, hi: 32 },
    { label: "33-128", lo: 33, hi: 128 },
    { label: "129-512", lo: 129, hi: 512 },
    { label: "513-2k", lo: 513, hi: 2048 },
    { label: "2k-8k", lo: 2049, hi: 8192 },
    { label: "8k+", lo: 8193, hi: Number.POSITIVE_INFINITY },
  ]
  const hist = buckets.map((b) => ({ bucket: b.label, count: 0 }))
  for (const l of lens) {
    for (let i = 0; i < buckets.length; i++) {
      const b = buckets[i]!
      if (l >= b.lo && l <= b.hi) { hist[i]!.count++; break }
    }
  }

  return {
    path: relIdxPath || "",
    version,
    num_docs: numDocs,
    doc_len: { min, p50, p90, p99, max },
    histogram: hist,
  }
}

export async function indexDocView(
  relShardPath: string,
  docIdx: number,
  opts?: { max_tokens?: number }
): Promise<IndexDocView> {
  const shardAbs = safeResolve(relShardPath)
  const shardSt = statSync(shardAbs)
  if (!shardSt.isFile()) throw new Error("Shard not a file")

  const idxRel = relShardPath.replace(/\.(npy|bin)$/i, ".idx")
  const idxAbs = safeResolve(idxRel)
  const idxSt = statSync(idxAbs)
  if (!idxSt.isFile()) throw new Error("Missing .idx")

  const { numDocs } = idxHeader(idxAbs)
  const i = Math.max(0, Math.min(Math.floor(docIdx), Math.max(0, numDocs - 1)))
  const pair = await readBytes(idxAbs, 32 + i * 16, 16)
  if (pair.length < 16) throw new Error("Invalid .idx: truncated body")
  const start = Number(pair.readBigUInt64LE(0))
  const end = Number(pair.readBigUInt64LE(8))
  const length = Math.max(0, end - start)

  const maxTokens = Math.max(1, Math.min(Number.isFinite(opts?.max_tokens) ? Number(opts?.max_tokens) : 4096, 65536))
  const take = Math.min(length, maxTokens)
  const win = await tokenWindow(relShardPath, start, take)
  return {
    idx_path: idxRel,
    shard_path: relShardPath,
    doc_idx: i,
    start,
    end,
    length,
    tokens: win.tokens,
  }
}
