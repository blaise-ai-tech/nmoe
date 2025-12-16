"use client"

import * as React from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Folder, FileText, RefreshCw } from "lucide-react"

import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbSeparator } from "@/components/ui/breadcrumb"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

type Entry = { name: string; kind: "dir" | "file"; size: number; mtime_ms: number }

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

export function DatasetsViewer() {
  const router = useRouter()
  const sp = useSearchParams()
  const path = sp.get("path") || ""

  const [entries, setEntries] = React.useState<Entry[]>([])
  const [loading, setLoading] = React.useState(true)
  const [filter, setFilter] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)

  const fetchDir = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const url = new URL("/api/datasets", window.location.origin)
      if (path) url.searchParams.set("path", path)
      const r = await fetch(url.toString(), { cache: "no-store" })
      const j = await r.json()
      if (!r.ok) throw new Error(j?.error ? String(j.error) : "Failed to list datasets")
      setEntries(j.entries || [])
    } catch (e: any) {
      setError(String(e))
      setEntries([])
    } finally {
      setLoading(false)
    }
  }, [path])

  React.useEffect(() => {
    fetchDir()
  }, [fetchDir])

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

  const filtered = React.useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return entries
    return entries.filter((e) => e.name.toLowerCase().includes(q))
  }, [entries, filter])

  const goto = (nextPath: string) => {
    const url = new URL(window.location.href)
    if (nextPath) url.searchParams.set("path", nextPath)
    else url.searchParams.delete("path")
    router.push(url.pathname + url.search)
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <Breadcrumb>
          <BreadcrumbList>
            {crumbs.map((c, i) => (
              <React.Fragment key={c.path || "__root__"}>
                <BreadcrumbItem>
                  <BreadcrumbLink asChild>
                    <button className="hover:underline" onClick={() => goto(c.path)}>
                      {c.label}
                    </button>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                {i < crumbs.length - 1 && <BreadcrumbSeparator />}
              </React.Fragment>
            ))}
          </BreadcrumbList>
        </Breadcrumb>

        <div className="flex items-center gap-2">
          <Input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter…" className="w-[260px]" />
          <Button variant="outline" onClick={fetchDir} disabled={loading}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <div className="text-sm text-destructive">{error}</div>
      )}

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>Modified</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading && (
            <TableRow>
              <TableCell colSpan={5} className="text-sm text-muted-foreground">
                Loading…
              </TableCell>
            </TableRow>
          )}
          {!loading && filtered.map((e) => {
            const isDir = e.kind === "dir"
            const nextPath = path ? `${path}/${e.name}` : e.name
            const isTokenShard = !isDir && (e.name.toLowerCase().endsWith(".npy") || e.name.toLowerCase().endsWith(".bin"))
            const isIndex = !isDir && e.name.toLowerCase().endsWith(".idx")
            return (
              <TableRow key={e.name}>
                <TableCell>
                  <button
                    className="flex items-center gap-2 hover:underline"
                    onClick={() => { if (isDir) goto(nextPath) }}
                    disabled={!isDir}
                  >
                    {isDir ? <Folder className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
                    <span className={isDir ? "font-medium" : "font-mono text-xs"}>{e.name}</span>
                  </button>
                </TableCell>
                <TableCell>
                  {isDir ? <Badge variant="outline">dir</Badge> : <Badge variant="outline">file</Badge>}
                </TableCell>
                <TableCell className="font-mono text-xs">{isDir ? "—" : fmtBytes(e.size)}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{fmtTime(e.mtime_ms)}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <a href={`/datasets/inspect?path=${encodeURIComponent(nextPath)}`}>
                        Inspect
                      </a>
                    </Button>
                    {isDir ? null : <FilePreviewDialog path={nextPath} name={e.name} />}
                    {!isDir && (isTokenShard || isIndex) ? (
                      <Badge variant="secondary" className="ml-1">
                        {isTokenShard ? "tokens" : "index"}
                      </Badge>
                    ) : null}
                  </div>
                </TableCell>
              </TableRow>
            )
          })}
          {!loading && filtered.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="text-sm text-muted-foreground">
                No entries.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}

function FilePreviewDialog({ path, name }: { path: string; name: string }) {
  const [text, setText] = React.useState<string>("")
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const url = new URL("/api/datasets/preview", window.location.origin)
      url.searchParams.set("path", path)
      url.searchParams.set("max_bytes", "65536")
      const r = await fetch(url.toString(), { cache: "no-store" })
      const j = await r.json()
      if (!r.ok) throw new Error(j?.error ? String(j.error) : "Failed to preview")
      setText(String(j.text || ""))
    } catch (e: any) {
      setError(String(e))
      setText("")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" onClick={() => { void load() }}>
          View
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm">{name}</DialogTitle>
        </DialogHeader>
        {loading && <div className="text-sm text-muted-foreground">Loading…</div>}
        {error && <div className="text-sm text-destructive">{error}</div>}
        <pre className="text-xs p-3 rounded bg-muted overflow-auto max-h-[70vh]">{text}</pre>
      </DialogContent>
    </Dialog>
  )
}
