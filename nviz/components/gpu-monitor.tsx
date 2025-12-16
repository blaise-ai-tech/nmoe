"use client"

import * as React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts"
import { useRuns } from "@/lib/run-context"

interface GpuInfo {
  id: number
  name: string
  temperature: number
  power: number
  powerLimit: number
  utilization: number
  memoryUsed: number
  memoryTotal: number
}

interface GpuProcess {
  gpu: number
  pid: number
  user: string
  gpuMem: number
  cpuMem: number
  cpu: number
  command: string
  time: string
}

interface HistoryPoint {
  time: number
  [key: string]: number // gpu0_util, gpu0_mem, etc.
}

export function GpuMonitor() {
  const { currentRun } = useRuns()
  const [gpus, setGpus] = React.useState<GpuInfo[]>([])
  const [processes, setProcesses] = React.useState<GpuProcess[]>([])
  const [history, setHistory] = React.useState<HistoryPoint[]>([])
  const [loading, setLoading] = React.useState(true)

  const toPct = (v: number | undefined): number => {
    if (v == null || Number.isNaN(v)) return 0
    // Some exporters log 0..1 instead of 0..100; normalize to percent
    return v <= 1 ? v * 100 : v
  }

  React.useEffect(() => {
    let cancelled = false
    let interval: any
    const start = () => {
      if (interval) return
      interval = setInterval(fetchGpuData, 2000)
    }
    const stop = () => {
      if (interval) { clearInterval(interval); interval = null }
    }

    async function fetchGpuData() {
      if (!currentRun) { setLoading(false); return }

      try {
        const res = await fetch(`/api/gpu?run=${encodeURIComponent(currentRun)}`)
        if (!res.ok) throw new Error(await res.text())
        if (cancelled) return
        const data = await res.json()
        const list: GpuInfo[] = (data.gpus || []).map((g: any) => ({
          id: g.id,
          name: 'NVIDIA GPU',
          temperature: g.temperatureC ?? 0,
          power: g.powerW ?? 0,
          powerLimit: g.powerLimitW ?? 0,
          utilization: toPct(g.utilizationGpu),
          memoryUsed: (g.memoryUsedGiB ?? 0) * 1024,
          memoryTotal: (g.memoryTotalGiB ?? 0) * 1024,
        }))
        setGpus(list)

        // Prefer server-provided series if available
        if (data.series && (data.series.avgUtil?.length || data.series.avgMemPercent?.length)) {
          type StepPoint = { time: number; avgUtil?: number; avgMem?: number }
          const byStep = new Map<number, StepPoint>()
          for (const p of (data.series.avgUtil || [])) {
            const s = Number(p.step)
            const o: StepPoint = byStep.get(s) ?? { time: s }
            o.avgUtil = toPct(p.value)
            byStep.set(s, o)
          }
          for (const p of (data.series.avgMemPercent || [])) {
            const s = Number(p.step)
            const o: StepPoint = byStep.get(s) ?? { time: s }
            o.avgMem = Number(p.value)
            byStep.set(s, o)
          }
          const merged = Array.from(byStep.values()).sort((a, b) => a.time - b.time) as HistoryPoint[]
          setHistory(merged.slice(-120))
        } else if (list.length > 0) {
          const avgUtil = list.reduce((sum, x) => sum + toPct(x.utilization || 0), 0) / list.length
          const avgMem = list.reduce((sum, x) => sum + (x.memoryTotal ? (x.memoryUsed / x.memoryTotal * 100) : 0), 0) / list.length
          setHistory(prev => [...prev.slice(-119), { time: Date.now(), avgUtil, avgMem }])
        }
        setLoading(false)
      } catch (e) {
        console.error('Failed to fetch GPU data:', e)
        if (!cancelled) setLoading(false)
      }
    }
    fetchGpuData()
    start()
    const onVis = () => {
      if (document.hidden) { stop() } else { fetchGpuData(); start() }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => { cancelled = true; stop(); document.removeEventListener('visibilitychange', onVis) }
  }, [currentRun])

  const getUtilColorStyle = (util: number) => {
    if (util < 20) return { color: 'hsl(var(--chart-4))' } // Green
    if (util < 60) return { color: 'hsl(var(--chart-2))' } // Yellow
    return { color: 'hsl(var(--chart-1))' } // Red
  }

  const getMemColorStyle = (used: number, total: number) => {
    const percent = (used / total) * 100
    if (percent < 50) return { color: 'hsl(var(--chart-4))' }
    if (percent < 80) return { color: 'hsl(var(--chart-2))' }
    return { color: 'hsl(var(--chart-1))' }
  }

  const getBarColorStyle = (percent: number) => {
    if (percent < 50) return { backgroundColor: 'hsl(var(--chart-4))' }
    if (percent < 80) return { backgroundColor: 'hsl(var(--chart-2))' }
    return { backgroundColor: 'hsl(var(--chart-1))' }
  }

  // Compute header metrics with preference for server aggregates when available
  const lastAgg = React.useRef<{ util?: number; memPct?: number }>({})
  const headerUtil = React.useMemo(() => {
    // Prefer server-provided aggregate if we have it in last history point (avgUtil in 0..100) or API agg
    if (history.length > 0 && typeof history[history.length - 1].avgUtil === 'number') {
      lastAgg.current.util = history[history.length - 1].avgUtil
    }
    // Fallback: mean across GPUs
    if (!lastAgg.current.util && gpus.length > 0) {
      lastAgg.current.util = gpus.reduce((sum, gpu) => sum + (gpu.utilization || 0), 0) / gpus.length
    }
    return lastAgg.current.util || 0
  }, [history, gpus])

  const headerMem = React.useMemo(() => {
    // Prefer server-provided series avgMem if present; otherwise compute weighted by total memory
    if (history.length > 0 && typeof history[history.length - 1].avgMem === 'number') {
      lastAgg.current.memPct = history[history.length - 1].avgMem
    }
    if (!lastAgg.current.memPct && gpus.length > 0) {
      const total = gpus.reduce((s, g) => s + (g.memoryTotal || 0), 0)
      const used = gpus.reduce((s, g) => s + (g.memoryUsed || 0), 0)
      lastAgg.current.memPct = total > 0 ? (used / total) * 100 : 0
    }
    return lastAgg.current.memPct || 0
  }, [history, gpus])

  if (!loading && gpus.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle>GPU Stats</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">No GPU metrics found for this run.</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* GPU Table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="font-mono text-sm text-muted-foreground font-normal">
            NVIDIA-SMI 570.124.06 | Driver Version: 570.124.06 | CUDA Version: 12.8
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <Table>
            <TableHeader>
              <TableRow className="font-mono text-xs text-muted-foreground">
                <TableHead className="w-12">GPU</TableHead>
                <TableHead className="w-24">Temp</TableHead>
                <TableHead className="w-12">Perf</TableHead>
                <TableHead className="w-28">Pwr:Usg/Cap</TableHead>
                <TableHead className="w-44">Memory-Usage</TableHead>
                <TableHead className="w-32">GPU-Util</TableHead>
                <TableHead className="w-24">Compute M.</TableHead>
                <TableHead className="flex-1 min-w-[300px]">MEM:</TableHead>
                <TableHead className="flex-1 min-w-[300px]">UTL:</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {gpus.map((gpu) => {
                const memPercent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                return (
                  <TableRow key={gpu.id} className="font-mono text-xs">
                    <TableCell className="py-2">{gpu.id}</TableCell>
                    <TableCell className="py-2" style={gpu.temperature > 70 ? { color: 'hsl(var(--chart-1))' } : { color: 'hsl(var(--chart-2))' }}>
                      {Math.floor(gpu.temperature)}C
                    </TableCell>
                    <TableCell className="py-2">P2</TableCell>
                    <TableCell className="py-2" style={gpu.power > 200 ? { color: 'hsl(var(--chart-1))' } : { color: 'hsl(var(--chart-2))' }}>
                      {Math.floor(gpu.power)}W / {gpu.powerLimit}W
                    </TableCell>
                    <TableCell className="py-2">
                      {Math.floor(gpu.memoryUsed)}MiB / {gpu.memoryTotal}MiB
                    </TableCell>
                    <TableCell className="py-2" style={getUtilColorStyle(gpu.utilization)}>
                      {Math.floor(gpu.utilization)}%
                    </TableCell>
                    <TableCell className="py-2">Default</TableCell>
                    <TableCell className="py-2">
                      <div className="flex items-center gap-2 w-full">
                        <span className="text-xs text-muted-foreground">MEM:</span>
                        <div className="flex-1 h-4 bg-muted rounded-sm overflow-hidden">
                          <div
                            className="h-full transition-all"
                            style={{ width: `${memPercent}%`, ...getBarColorStyle(memPercent) }}
                          />
                        </div>
                        <span className="text-xs min-w-[3rem] text-right" style={getMemColorStyle(gpu.memoryUsed, gpu.memoryTotal)}>
                          {Math.floor(memPercent)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="py-2">
                      <div className="flex items-center gap-2 w-full">
                        <span className="text-xs text-muted-foreground">UTL:</span>
                        <div className="flex-1 h-4 bg-muted rounded-sm overflow-hidden">
                          <div
                            className="h-full transition-all"
                            style={{ width: `${gpu.utilization}%`, ...getBarColorStyle(gpu.utilization) }}
                          />
                        </div>
                        <span className="text-xs min-w-[3rem] text-right" style={getUtilColorStyle(gpu.utilization)}>
                          {Math.floor(gpu.utilization)}%
                        </span>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="font-mono text-sm font-normal">AVG GPU UTL: {Math.floor(headerUtil)}%</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history}>
                  <defs>
                    <linearGradient id="utilGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.6}/>
                      <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload) return null
                      return (
                        <div className="bg-background border rounded p-2 text-xs font-mono">
                          <div>Utilization: {Math.floor(payload[0]?.value as number)}%</div>
                        </div>
                      )
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="avgUtil"
                    stroke="hsl(var(--chart-1))"
                    fill="url(#utilGradient)"
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="font-mono text-sm font-normal">AVG GPU MEM: {Math.floor(headerMem)}%</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={history}>
                  <defs>
                    <linearGradient id="memGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--chart-2))" stopOpacity={0.6}/>
                      <stop offset="95%" stopColor="hsl(var(--chart-2))" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[0, 100]} hide />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload) return null
                      return (
                        <div className="bg-background border rounded p-2 text-xs font-mono">
                          <div>Memory: {Math.floor(payload[0]?.value as number)}%</div>
                        </div>
                      )
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="avgMem"
                    stroke="hsl(var(--chart-2))"
                    fill="url(#memGradient)"
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

    </div>
  )
}
