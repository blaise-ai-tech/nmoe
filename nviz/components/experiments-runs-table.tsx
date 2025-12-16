"use client"

import * as React from "react"
import { api, type ExperimentRunRow, type ExperimentSummary } from "@/lib/api"
import { useRuns } from "@/lib/run-context"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

function fmtTs(s?: string | null): string {
  if (!s) return "—"
  return s.replace("T", " ").replace("+00:00", "Z")
}

function durationSeconds(started?: string | null, ended?: string | null): number | null {
  if (!started) return null
  const a = Date.parse(started)
  if (!Number.isFinite(a)) return null
  const b = ended ? Date.parse(ended) : Date.now()
  if (!Number.isFinite(b)) return null
  return Math.max(0, (b - a) / 1000)
}

function fmtDuration(sec: number | null): string {
  if (sec == null) return "—"
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = sec / 60
  if (m < 60) return `${m.toFixed(1)}m`
  const h = m / 60
  return `${h.toFixed(1)}h`
}

function parseResults(results_json?: string | null): any | null {
  if (!results_json) return null
  try { return JSON.parse(results_json) } catch { return null }
}

export function ExperimentsRunsTable() {
  const { selectedRuns, setSelectedRuns } = useRuns()
  const [experiments, setExperiments] = React.useState<ExperimentSummary[]>([])
  const [experimentId, setExperimentId] = React.useState<string>("__all__")
  const [rows, setRows] = React.useState<ExperimentRunRow[]>([])
  const [q, setQ] = React.useState("")
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    let alive = true
    ;(async () => {
      try {
        const exps = await api.experiments()
        if (!alive) return
        setExperiments(exps)
      } finally {
        if (alive) setLoading(false)
      }
    })()
    return () => { alive = false }
  }, [])

  React.useEffect(() => {
    let alive = true
    ;(async () => {
      const exp = experimentId === "__all__" ? undefined : experimentId
      const rs = await api.experimentRuns(exp, 500)
      if (!alive) return
      setRows(rs)
    })().catch((e) => console.error(e))
    return () => { alive = false }
  }, [experimentId])

  const filtered = React.useMemo(() => {
    const qq = q.trim().toLowerCase()
    if (!qq) return rows
    return rows.filter((r) => r.run.toLowerCase().includes(qq) || (r.status || "").toLowerCase().includes(qq))
  }, [rows, q])

  const toggle = (run: string) => {
    if (selectedRuns.includes(run)) {
      if (selectedRuns.length > 1) setSelectedRuns(selectedRuns.filter((x) => x !== run))
      return
    }
    setSelectedRuns([...selectedRuns, run])
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <Select value={experimentId} onValueChange={setExperimentId}>
            <SelectTrigger className="w-[360px]">
              <SelectValue placeholder="Select experiment" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All experiments</SelectItem>
              {experiments.map((e) => (
                <SelectItem key={e.id} value={e.id}>
                  {e.project}/{e.name} ({e.runs_total})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {loading && <Badge variant="outline">Loading…</Badge>}
        </div>
        <div className="flex items-center gap-2">
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Filter runs…"
            className="w-[260px]"
          />
          <Button
            variant="outline"
            onClick={() => {
              const all = filtered.map((r) => r.run)
              if (all.length) setSelectedRuns(all.slice(0, 8))
            }}
          >
            Select top 8
          </Button>
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead />
            <TableHead>Run</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Started</TableHead>
            <TableHead>Duration</TableHead>
            <TableHead>Steps</TableHead>
            <TableHead>Final loss</TableHead>
            <TableHead>Details</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filtered.map((r) => {
            const selected = selectedRuns.includes(r.run)
            const dur = durationSeconds(r.started_at, r.ended_at ?? null)
            const results = parseResults(r.results_json)
            const finalLoss = results?.final_loss
            const steps = r.last_step || results?.steps_completed || 0
            return (
              <TableRow key={r.run} data-state={selected ? "selected" : undefined}>
                <TableCell>
                  <Checkbox checked={selected} onCheckedChange={() => toggle(r.run)} />
                </TableCell>
                <TableCell className="font-mono text-xs">
                  <button className="hover:underline" onClick={() => toggle(r.run)}>
                    {r.run}
                  </button>
                </TableCell>
                <TableCell>
                  {r.status === "running" && <Badge variant="outline">Running</Badge>}
                  {r.status === "completed" && <Badge variant="outline">Completed</Badge>}
                  {r.status === "failed" && <Badge variant="destructive">Failed</Badge>}
                  {r.status === "killed" && <Badge variant="outline">Killed</Badge>}
                  {!["running","completed","failed","killed"].includes(r.status) && <Badge variant="outline">{r.status || "unknown"}</Badge>}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">{fmtTs(r.started_at)}</TableCell>
                <TableCell className="text-xs text-muted-foreground">{fmtDuration(dur)}</TableCell>
                <TableCell className="font-mono text-xs">{steps}</TableCell>
                <TableCell className="font-mono text-xs">{finalLoss != null ? Number(finalLoss).toFixed(3) : "—"}</TableCell>
                <TableCell>
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button variant="outline" size="sm">View</Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-3xl">
                      <DialogHeader>
                        <DialogTitle className="font-mono text-sm">{r.run}</DialogTitle>
                      </DialogHeader>
                      <div className="grid grid-cols-1 gap-4">
                        <div className="text-xs text-muted-foreground">
                          <div>Experiment: <span className="font-mono">{r.experiment_id}</span></div>
                          <div>Status: <span className="font-mono">{r.status}</span></div>
                          <div>Started: <span className="font-mono">{fmtTs(r.started_at)}</span></div>
                          <div>Ended: <span className="font-mono">{fmtTs(r.ended_at ?? null)}</span></div>
                          <div>Git: <span className="font-mono">{(r.git_hash || "").slice(0, 12) || "—"}</span></div>
                        </div>
                        <div>
                          <div className="text-xs font-medium mb-2">Config</div>
                          <pre className="text-xs p-3 rounded bg-muted overflow-auto max-h-[300px]">{r.config_json || ""}</pre>
                        </div>
                        <div>
                          <div className="text-xs font-medium mb-2">Results</div>
                          <pre className="text-xs p-3 rounded bg-muted overflow-auto max-h-[200px]">{r.results_json || ""}</pre>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                </TableCell>
              </TableRow>
            )
          })}
          {filtered.length === 0 && (
            <TableRow>
              <TableCell colSpan={8} className="text-sm text-muted-foreground">
                No runs found.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
