"use client"

import { useEffect, useState } from "react"
import { AlertTriangle } from "lucide-react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
// TB removed; data comes from /api/router
import { useRuns } from "@/lib/run-context"

interface LayerMetrics {
  layer: number
  cv: number
  entropy: number
  experts_active: number
  bias_range: number
}

export function RouterHealthHeatmap() {
  const { currentRun } = useRuns()
  const [layers, setLayers] = useState<LayerMetrics[]>([])
  const [initialLoading, setInitialLoading] = useState(true)
  const [agg, setAgg] = useState<any | null>(null)

  useEffect(() => {
    let cancelled = false
    async function fetchRouterData() {
      if (!currentRun) { setInitialLoading(false); return }
      try {
        const res = await fetch(`/api/router?run=${encodeURIComponent(currentRun)}`)
        if (!res.ok) throw new Error(await res.text())
        if (cancelled) return
        const data = await res.json()
        setLayers(data.layers || [])
        setAgg(data.agg || null)
        setInitialLoading(false)
      } catch (err) {
        console.error('Failed to fetch router health:', err)
        if (!cancelled) setInitialLoading(false)
      }
    }
    fetchRouterData()
    let interval: any = setInterval(fetchRouterData, 10000)
    const onVis = () => {
      if (document.hidden) { clearInterval(interval); interval = null }
      else { fetchRouterData(); if (!interval) interval = setInterval(fetchRouterData, 10000) }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => { cancelled = true; if (interval) clearInterval(interval); document.removeEventListener('visibilitychange', onVis) }
  }, [currentRun])

  // Color scale helpers
  const getCVColor = (cv: number) => {
    // CV < 1.0 = good (green), 1.0-2.0 = warning (yellow), > 2.0 = bad (red)
    if (cv < 1.0) return "hsl(var(--chart-4))" // Foam (good)
    if (cv < 2.0) return "hsl(var(--chart-2))" // Gold (warning)
    return "hsl(var(--chart-1))" // Love (bad)
  }

  const getEntropyColor = (entropy: number) => {
    // Higher entropy is better (more uniform distribution)
    if (entropy > 0.8) return "hsl(var(--chart-4))" // Foam (good)
    if (entropy > 0.6) return "hsl(var(--chart-2))" // Gold (warning)
    return "hsl(var(--chart-1))" // Love (bad)
  }

  const getExpertsColor = (active: number, total: number = 64) => {
    const ratio = active / total
    if (ratio > 0.95) return "hsl(var(--chart-4))" // Foam (good)
    if (ratio > 0.85) return "hsl(var(--chart-2))" // Gold (warning)
    return "hsl(var(--chart-1))" // Love (bad)
  }

  const fmt = (v: number | undefined, digits = 2) =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—'

  if (initialLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-96 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (layers.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Router Health</CardTitle>
          <CardDescription>No router data available</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const getRowStatus = (layer: LayerMetrics) => {
    // Treat unknown as warning to avoid overstating health
    if (layer.experts_active == null) return "warning"
    // Critical: < 40 experts
    if (layer.experts_active < 40) return "critical"
    // Warning: 40-60 experts
    if (layer.experts_active <= 60) return "warning"
    // Good: 61-64 experts
    return "good"
  }

  const getRowBgColor = (status: string) => {
    if (status === "critical") return "rgba(235, 111, 146, 0.1)" // Love with 10% opacity
    if (status === "warning") return "rgba(246, 193, 119, 0.1)" // Gold with 10% opacity
    return "transparent"
  }

  // Order strictly by layer index (ascending) per request
  const sortedLayers = [...layers].sort((a, b) => a.layer - b.layer)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Router Health by Layer</CardTitle>
        <CardDescription>
          {layers.length} layers
        </CardDescription>
        {agg && (
          <div className="mt-1 flex flex-wrap gap-4 text-xs font-mono">
            {typeof agg.mean_cv === 'number' && (
              <div>
                <span className="text-muted-foreground">Mean CV: </span>
                <span style={{ color: getCVColor(agg.mean_cv) }}>{agg.mean_cv.toFixed(2)}</span>
              </div>
            )}
            {typeof agg.mean_entropy === 'number' && (
              <div>
                <span className="text-muted-foreground">Mean Entropy: </span>
                <span style={{ color: getEntropyColor(agg.mean_entropy) }}>{agg.mean_entropy.toFixed(2)}</span>
              </div>
            )}
            {typeof agg.min_entropy === 'number' && (
              <div>
                <span className="text-muted-foreground">Min Entropy: </span>
                <span style={{ color: getEntropyColor(agg.min_entropy) }}>{agg.min_entropy.toFixed(2)}</span>
              </div>
            )}
            {typeof agg.std_cv === 'number' && (
              <div>
                <span className="text-muted-foreground">Std CV: </span>
                <span>{agg.std_cv.toFixed(2)}</span>
              </div>
            )}
            {typeof agg.dead_experts_count === 'number' && (
              <div>
                <span className="text-muted-foreground">Dead Experts: </span>
                <span style={{ color: agg.dead_experts_count > 0 ? 'hsl(var(--chart-1))' : 'inherit' }}>{agg.dead_experts_count}</span>
              </div>
            )}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {/* Header */}
          <div className="grid grid-cols-[60px_1fr_1fr_1fr_1fr] gap-2 text-xs font-medium text-muted-foreground pb-2 border-b">
            <div>Layer</div>
            <div>CV</div>
            <div>Entropy</div>
            <div>Experts</div>
            <div>Bias Range</div>
          </div>

          {/* Rows */}
          {sortedLayers.map(layer => {
            const status = getRowStatus(layer)
            return (
              <div
                key={layer.layer}
                className="grid grid-cols-[60px_1fr_1fr_1fr_1fr] gap-2 items-center py-2 px-2 rounded transition-colors"
                style={{ backgroundColor: getRowBgColor(status) }}
              >
                <div className="font-mono text-sm font-medium">L{layer.layer}</div>

                {/* CV */}
                <div className="text-sm font-mono" style={{ color: getCVColor(layer.cv) }}>{fmt(layer.cv)}</div>

                {/* Entropy */}
                <div className="text-sm font-mono" style={{ color: getEntropyColor(layer.entropy) }}>{fmt(layer.entropy)}</div>

                {/* Experts Active */}
                <div className="text-sm font-mono" style={{ color: getExpertsColor(layer.experts_active ?? 0) }}>{layer.experts_active ?? '—'}/64</div>

                {/* Bias Range */}
                <div className="text-sm font-mono text-muted-foreground">{fmt(layer.bias_range)}</div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
