"use client"

import * as React from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { ArrowLeft, RefreshCw } from "lucide-react"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"

import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbSeparator } from "@/components/ui/breadcrumb"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

type Meta = { path: string; kind: "dir" | "file"; size: number; mtime_ms: number }
type ManifestPreview = {
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

type ShardStats = {
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

type TokenWindow = { path: string; offset: number; length: number; tokens: number[] }
type IndexHeaderStats = {
  path: string
  version: number
  num_docs: number
  doc_len: { min: number | null; p50: number | null; p90: number | null; p99: number | null; max: number | null }
  histogram: Array<{ bucket: string; count: number }>
}
type IndexDocView = { idx_path: string; shard_path: string; doc_idx: number; start: number; end: number; length: number; tokens: number[] }

type FlowInspect = {
  path: string
  is_flow_dir: boolean
  stages: { raw: boolean; hydra: boolean; kept: boolean; rephrase: boolean; prep: boolean }
  flow_spec: any | null
  hydra_summary: null | {
    total: number
    kept: number
    band: number
    dropped: number
    tau_drop: number | null
    tau_keep: number | null
    top_sources: Array<{ source: string; kept: number; band: number; dropped: number }>
  }
  training_shards: null | {
    path: string
    manifest: ManifestPreview
  }
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return "—"
  if (n < 1024) return `${n} B`
  const kb = n / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  const mb = kb / 1024
  if (mb < 1024) return `${mb.toFixed(1)} MB`
  const gb = mb / 1024
  return `${gb.toFixed(2)} GB`
}

function fmtTime(ms: number): string {
  if (!ms) return "—"
  const d = new Date(ms)
  return d.toISOString().replace("T", " ").replace("Z", "")
}

export function DatasetsInspector() {
  const router = useRouter()
  const sp = useSearchParams()
  const path = sp.get("path") || ""

  const [meta, setMeta] = React.useState<Meta | null>(null)
  const [manifest, setManifest] = React.useState<ManifestPreview | null>(null)
  const [stats, setStats] = React.useState<ShardStats | null>(null)
  const [tokenWindow, setTokenWindow] = React.useState<TokenWindow | null>(null)
  const [idx, setIdx] = React.useState<IndexHeaderStats | null>(null)
  const [doc, setDoc] = React.useState<IndexDocView | null>(null)
  const [flow, setFlow] = React.useState<FlowInspect | null>(null)

  const [offset, setOffset] = React.useState("0")
  const [length, setLength] = React.useState("256")
  const [docIdx, setDocIdx] = React.useState("0")

  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const crumbs = React.useMemo(() => {
    const parts = path.split("/").filter(Boolean)
    const out: Array<{ label: string; path: string }> = [{ label: "/data", path: "" }]
    let cur = ""
    for (const p of parts) {
      cur = cur ? `${cur}/${p}` : p
      out.push({ label: p, path: cur })
    }
    return out
  }, [path])

  const refresh = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    setMeta(null)
    setManifest(null)
    setStats(null)
    setTokenWindow(null)
    setIdx(null)
    setDoc(null)
    setFlow(null)

    try {
      const metaUrl = new URL("/api/datasets/meta", window.location.origin)
      if (path) metaUrl.searchParams.set("path", path)
      const mRes = await fetch(metaUrl.toString(), { cache: "no-store" })
      const mJson = await mRes.json()
      if (!mRes.ok) throw new Error(mJson?.error ? String(mJson.error) : "Failed to stat path")
      setMeta(mJson as Meta)

      const kind = (mJson as Meta).kind
      const lower = path.toLowerCase()

      if (kind === "dir") {
        // Flow-first: dataprep shard workspace.
        {
          const flowUrl = new URL("/api/datasets/pipeline", window.location.origin)
          if (path) flowUrl.searchParams.set("path", path)
          const fr = await fetch(flowUrl.toString(), { cache: "no-store" })
          const fj = await fr.json()
          if (fr.ok) setFlow(fj as FlowInspect)
        }

        // Canonical dataset view: dir with manifest.json.
        {
          const manUrl = new URL("/api/datasets/manifest", window.location.origin)
          if (path) manUrl.searchParams.set("path", path)
          manUrl.searchParams.set("limit_shards", "200")
          const r = await fetch(manUrl.toString(), { cache: "no-store" })
          const j = await r.json()
          if (!r.ok) throw new Error(j?.error ? String(j.error) : "Failed to load manifest")
          setManifest(j as ManifestPreview)
        }
        return
      }

      const isShard = lower.endsWith(".npy") || lower.endsWith(".bin")
      const isIdx = lower.endsWith(".idx")

      if (isShard) {
        const sUrl = new URL("/api/datasets/shard/stats", window.location.origin)
        sUrl.searchParams.set("path", path)
        const r = await fetch(sUrl.toString(), { cache: "no-store" })
        const j = await r.json()
        if (!r.ok) throw new Error(j?.error ? String(j.error) : "Failed to inspect shard")
        setStats(j as ShardStats)

        {
          const parent = path.includes("/") ? path.split("/").slice(0, -1).join("/") : ""
          const manUrl = new URL("/api/datasets/manifest", window.location.origin)
          if (parent) manUrl.searchParams.set("path", parent)
          manUrl.searchParams.set("limit_shards", "0")
          const mr = await fetch(manUrl.toString(), { cache: "no-store" })
          const mj = await mr.json()
          if (mr.ok) setManifest(mj as ManifestPreview)
        }

        const idxPath = path.replace(/\.(npy|bin)$/i, ".idx")
        if ((j as ShardStats).has_index) {
          const iUrl = new URL("/api/datasets/index/header", window.location.origin)
          iUrl.searchParams.set("path", idxPath)
          const ir = await fetch(iUrl.toString(), { cache: "no-store" })
          const ij = await ir.json()
          if (ir.ok) setIdx(ij as IndexHeaderStats)
        }
        return
      }

      if (isIdx) {
        const iUrl = new URL("/api/datasets/index/header", window.location.origin)
        iUrl.searchParams.set("path", path)
        const ir = await fetch(iUrl.toString(), { cache: "no-store" })
        const ij = await ir.json()
        if (!ir.ok) throw new Error(ij?.error ? String(ij.error) : "Failed to inspect index")
        setIdx(ij as IndexHeaderStats)
        return
      }

      // Fallback: try manifest in parent dir for context.
      const parent = path.includes("/") ? path.split("/").slice(0, -1).join("/") : ""
      {
        const manUrl = new URL("/api/datasets/manifest", window.location.origin)
        if (parent) manUrl.searchParams.set("path", parent)
        manUrl.searchParams.set("limit_shards", "0")
        const r = await fetch(manUrl.toString(), { cache: "no-store" })
        const j = await r.json()
        if (r.ok) setManifest(j as ManifestPreview)
      }
    } catch (e: any) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [path])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  const loadWindow = async () => {
    setError(null)
    try {
      const o = parseInt(offset || "0", 10)
      const l = parseInt(length || "256", 10)
      const u = new URL("/api/datasets/shard/window", window.location.origin)
      u.searchParams.set("path", path)
      u.searchParams.set("offset", String(o))
      u.searchParams.set("length", String(l))
      const r = await fetch(u.toString(), { cache: "no-store" })
      const j = await r.json()
      if (!r.ok) throw new Error(j?.error ? String(j.error) : "Failed to load window")
      setTokenWindow(j as TokenWindow)
    } catch (e: any) {
      setError(String(e))
      setTokenWindow(null)
    }
  }

  const loadDoc = async (nextIdx?: number) => {
    setError(null)
    try {
      const n = nextIdx != null ? nextIdx : parseInt(docIdx || "0", 10)
      const u = new URL("/api/datasets/index/doc", window.location.origin)
      u.searchParams.set("shard_path", path)
      u.searchParams.set("doc_idx", String(n))
      u.searchParams.set("max_tokens", "4096")
      const r = await fetch(u.toString(), { cache: "no-store" })
      const j = await r.json()
      if (!r.ok) throw new Error(j?.error ? String(j.error) : "Failed to load document")
      setDoc(j as IndexDocView)
      setDocIdx(String((j as IndexDocView).doc_idx))
    } catch (e: any) {
      setError(String(e))
      setDoc(null)
    }
  }

  const goBack = () => {
    const parent = path.includes("/") ? path.split("/").slice(0, -1).join("/") : ""
    const url = new URL("/datasets", window.location.origin)
    if (parent) url.searchParams.set("path", parent)
    router.push(url.pathname + url.search)
  }

  const showPipeline = Boolean(flow?.is_flow_dir)

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={goBack}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Breadcrumb>
            <BreadcrumbList>
              {crumbs.map((c, i) => (
                <React.Fragment key={c.path || "__root__"}>
                  <BreadcrumbItem>
                    <BreadcrumbLink asChild>
                      <a href={`/datasets/inspect?path=${encodeURIComponent(c.path)}`} className="hover:underline">
                        {c.label}
                      </a>
                    </BreadcrumbLink>
                  </BreadcrumbItem>
                  {i < crumbs.length - 1 && <BreadcrumbSeparator />}
                </React.Fragment>
              ))}
            </BreadcrumbList>
          </Breadcrumb>
        </div>

        <Button variant="outline" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && <div className="text-sm text-destructive">{error}</div>}

      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          {showPipeline ? <TabsTrigger value="pipeline">Pipeline</TabsTrigger> : null}
          {stats ? <TabsTrigger value="shard">Shard</TabsTrigger> : null}
          {idx ? <TabsTrigger value="index">Index</TabsTrigger> : null}
          {manifest?.found ? <TabsTrigger value="manifest">Manifest</TabsTrigger> : null}
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="font-mono text-sm">{path || "/data"}</span>
                <span className="flex items-center gap-2">
                  {meta ? <Badge variant="outline">{meta.kind}</Badge> : null}
                  {stats?.kind ? <Badge variant="secondary">{stats.kind}</Badge> : null}
                  {showPipeline ? <Badge variant="secondary">flow</Badge> : null}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div>
                <div className="text-xs text-muted-foreground">Size</div>
                <div className="font-mono text-sm">{meta ? fmtBytes(meta.size) : "—"}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Modified</div>
                <div className="font-mono text-sm">{meta ? fmtTime(meta.mtime_ms) : "—"}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Tokens</div>
                <div className="font-mono text-sm">{stats ? stats.tokens.toLocaleString() : "—"}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">EOS</div>
                <div className="font-mono text-sm">{stats?.eos_token_id ?? "—"}</div>
              </div>
            </CardContent>
          </Card>

          {manifest?.found ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Context (manifest)</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <div>
                  <div className="text-xs text-muted-foreground">Dataset</div>
                  <div className="font-mono text-sm">{manifest.dataset ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Version</div>
                  <div className="font-mono text-sm">{manifest.version ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Total tokens</div>
                  <div className="font-mono text-sm">{manifest.total_tokens?.toLocaleString() ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Total docs</div>
                  <div className="font-mono text-sm">{manifest.total_documents?.toLocaleString() ?? "—"}</div>
                </div>
              </CardContent>
            </Card>
          ) : null}
        </TabsContent>

        {showPipeline ? (
          <TabsContent value="pipeline" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Stages</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-wrap gap-2">
                <StageBadge name="raw" ok={Boolean(flow?.stages.raw)} />
                <StageBadge name="hydra" ok={Boolean(flow?.stages.hydra)} />
                <StageBadge name="kept" ok={Boolean(flow?.stages.kept)} />
                <StageBadge name="rephrase" ok={Boolean(flow?.stages.rephrase)} />
                <StageBadge name="prep" ok={Boolean(flow?.stages.prep)} />
              </CardContent>
            </Card>

            {flow?.hydra_summary ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">HYDRA summary</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
                  <div>
                    <div className="text-xs text-muted-foreground">Total</div>
                    <div className="font-mono text-sm">{flow.hydra_summary.total.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Kept</div>
                    <div className="font-mono text-sm">{flow.hydra_summary.kept.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Band</div>
                    <div className="font-mono text-sm">{flow.hydra_summary.band.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">τ_keep</div>
                    <div className="font-mono text-sm">{flow.hydra_summary.tau_keep == null ? "—" : flow.hydra_summary.tau_keep.toFixed(4)}</div>
                  </div>
                </CardContent>
                {flow.hydra_summary.top_sources.length ? (
                  <CardContent>
                    <div className="text-xs text-muted-foreground mb-2">Top sources (by kept)</div>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Source</TableHead>
                          <TableHead>Kept</TableHead>
                          <TableHead>Band</TableHead>
                          <TableHead>Dropped</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {flow.hydra_summary.top_sources.map((s) => (
                          <TableRow key={s.source}>
                            <TableCell className="font-mono text-xs">{s.source}</TableCell>
                            <TableCell className="font-mono text-xs">{s.kept.toLocaleString()}</TableCell>
                            <TableCell className="font-mono text-xs">{s.band.toLocaleString()}</TableCell>
                            <TableCell className="font-mono text-xs">{s.dropped.toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                ) : null}
              </Card>
            ) : (
              <div className="text-xs text-muted-foreground">No `hydra_grades/summary.json` found.</div>
            )}

            {flow?.training_shards?.manifest?.found ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center justify-between">
                    <span>Training shards</span>
                    <Button variant="outline" size="sm" asChild>
                      <a href={`/datasets/inspect?path=${encodeURIComponent(flow.training_shards.path)}`}>Inspect</a>
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
                  <div>
                    <div className="text-xs text-muted-foreground">Dataset</div>
                    <div className="font-mono text-sm">{flow.training_shards.manifest.dataset ?? "—"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Tokens</div>
                    <div className="font-mono text-sm">{flow.training_shards.manifest.total_tokens?.toLocaleString() ?? "—"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Docs</div>
                    <div className="font-mono text-sm">{flow.training_shards.manifest.total_documents?.toLocaleString() ?? "—"}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground">Shards</div>
                    <div className="font-mono text-sm">{flow.training_shards.manifest.num_shards?.toLocaleString() ?? "—"}</div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="text-xs text-muted-foreground">No `training_shards/manifest.json` found.</div>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">flow_spec.json</CardTitle>
              </CardHeader>
              <CardContent>
                {flow?.flow_spec ? (
                  <pre className="text-xs p-3 rounded bg-muted overflow-auto max-h-[70vh]">
                    {JSON.stringify(flow.flow_spec, null, 2)}
                  </pre>
                ) : (
                  <div className="text-xs text-muted-foreground">Unable to read flow_spec.json.</div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        ) : null}

        {stats ? (
          <TabsContent value="shard" className="space-y-4">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              <Card>
                <CardHeader><CardTitle className="text-sm">Sanity</CardTitle></CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Min</span><span className="font-mono">{stats.min ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Max</span><span className="font-mono">{stats.max ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">EOS count (scan)</span><span className="font-mono">{stats.eos_count ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Double EOS (scan)</span><span className="font-mono">{stats.double_eos_count ?? "—"}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Repetition ratio</span><span className="font-mono">{stats.repetition_ratio == null ? "—" : stats.repetition_ratio.toFixed(4)}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Out of range (sample)</span><span className="font-mono">{stats.out_of_range_count ?? "—"}</span></div>
                </CardContent>
              </Card>

              <Card className="lg:col-span-2">
                <CardHeader><CardTitle className="text-sm">Top tokens (sample)</CardTitle></CardHeader>
                <CardContent className="h-[240px]">
                  <ChartContainer
                    className="h-[220px] w-full"
                    config={{ count: { label: "count", color: "hsl(var(--primary))" } }}
                  >
                    <BarChart data={stats.top_tokens.map((t) => ({ token: String(t.token), count: t.count }))}>
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="token" tick={{ fontSize: 10 }} interval={0} angle={-45} height={60} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Bar dataKey="count" fill="var(--color-count)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ChartContainer>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Token window</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex flex-col gap-2 md:flex-row md:items-end">
                  <div className="flex-1">
                    <div className="text-xs text-muted-foreground mb-1">Offset (token idx)</div>
                    <Input value={offset} onChange={(e) => setOffset(e.target.value)} className="font-mono" />
                  </div>
                  <div className="w-[160px]">
                    <div className="text-xs text-muted-foreground mb-1">Length</div>
                    <Input value={length} onChange={(e) => setLength(e.target.value)} className="font-mono" />
                  </div>
                  <Button onClick={() => void loadWindow()}>Load</Button>
                </div>
                {tokenWindow ? (
                  <pre className="text-xs p-3 rounded bg-muted overflow-auto max-h-[50vh]">
                    {tokenWindow.tokens.join(" ")}
                  </pre>
                ) : (
                  <div className="text-xs text-muted-foreground">No window loaded.</div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        ) : null}

        {idx ? (
          <TabsContent value="index" className="space-y-4">
            <Card>
              <CardHeader><CardTitle className="text-sm">Index header</CardTitle></CardHeader>
              <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <div>
                  <div className="text-xs text-muted-foreground">Docs</div>
                  <div className="font-mono text-sm">{idx.num_docs.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">p50 len</div>
                  <div className="font-mono text-sm">{idx.doc_len.p50 ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">p90 len</div>
                  <div className="font-mono text-sm">{idx.doc_len.p90 ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">p99 len</div>
                  <div className="font-mono text-sm">{idx.doc_len.p99 ?? "—"}</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle className="text-sm">Doc length histogram (prefix sample)</CardTitle></CardHeader>
              <CardContent className="h-[220px]">
                <ChartContainer
                  className="h-[200px] w-full"
                  config={{ count: { label: "count", color: "hsl(var(--primary))" } }}
                >
                  <BarChart data={idx.histogram.map((h) => ({ bucket: h.bucket, count: h.count }))}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="count" fill="var(--color-count)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            {stats?.has_index ? (
              <Card>
                <CardHeader><CardTitle className="text-sm">Document view</CardTitle></CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-col gap-2 md:flex-row md:items-end">
                    <div className="w-[200px]">
                      <div className="text-xs text-muted-foreground mb-1">Doc idx</div>
                      <Input value={docIdx} onChange={(e) => setDocIdx(e.target.value)} className="font-mono" />
                    </div>
                    <Button onClick={() => void loadDoc()}>Load</Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        const n = idx.num_docs > 0 ? Math.floor(Math.random() * idx.num_docs) : 0
                        void loadDoc(n)
                      }}
                    >
                      Random
                    </Button>
                  </div>
                  {doc ? (
                    <div className="space-y-2">
                      <div className="text-xs text-muted-foreground">
                        doc {doc.doc_idx} [{doc.start},{doc.end}) len={doc.length.toLocaleString()}
                      </div>
                      <pre className="text-xs p-3 rounded bg-muted overflow-auto max-h-[50vh]">
                        {doc.tokens.join(" ")}
                      </pre>
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground">No document loaded.</div>
                  )}
                </CardContent>
              </Card>
            ) : (
              <div className="text-xs text-muted-foreground">
                Open a token shard (`.npy`) to view documents.
              </div>
            )}
          </TabsContent>
        ) : null}

        {manifest?.found ? (
          <TabsContent value="manifest" className="space-y-4">
            <Card>
              <CardHeader><CardTitle className="text-sm">Manifest</CardTitle></CardHeader>
              <CardContent className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <div>
                  <div className="text-xs text-muted-foreground">Dataset</div>
                  <div className="font-mono text-sm">{manifest.dataset ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Tokenizer</div>
                  <div className="font-mono text-sm">{manifest.tokenizer ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Vocab</div>
                  <div className="font-mono text-sm">{manifest.vocab_size ?? "—"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">EOS</div>
                  <div className="font-mono text-sm">{manifest.eos_token_id ?? "—"}</div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle className="text-sm">Shards (preview)</CardTitle></CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Path</TableHead>
                      <TableHead>Tokens</TableHead>
                      <TableHead>Docs</TableHead>
                      <TableHead>Checksum</TableHead>
                      <TableHead>Inspect</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(manifest.shards_preview || []).map((s) => (
                      <TableRow key={s.path}>
                        <TableCell className="font-mono text-xs">{s.path}</TableCell>
                        <TableCell className="font-mono text-xs">{s.num_tokens.toLocaleString()}</TableCell>
                        <TableCell className="font-mono text-xs">{s.num_documents.toLocaleString()}</TableCell>
                        <TableCell className="font-mono text-xs">{s.checksum ? `${s.checksum.slice(0, 10)}…` : "—"}</TableCell>
                        <TableCell>
                          <Button variant="outline" size="sm" asChild>
                            <a href={`/datasets/inspect?path=${encodeURIComponent(s.shard_rel)}`}>Inspect</a>
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                    {(manifest.shards_preview || []).length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={5} className="text-xs text-muted-foreground">No shards in preview.</TableCell>
                      </TableRow>
                    ) : null}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        ) : null}
      </Tabs>
    </div>
  )
}

function StageBadge({ name, ok }: { name: string; ok: boolean }) {
  return (
    <Badge variant={ok ? "default" : "outline"} className={ok ? "" : "text-muted-foreground"}>
      {name}
    </Badge>
  )
}

