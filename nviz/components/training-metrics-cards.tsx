"use client"

import { useEffect, useState } from "react"
import { TrendingDown, TrendingUp, AlertTriangle } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"

interface MetricCardData {
  label: string
  value: string
  trend: number
  status: "good" | "warning" | "error"
  footer: string
}

export function TrainingMetricsCards() {
  const [metrics, setMetrics] = useState<MetricCardData[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const runs = await api.runs()
        if (!runs.length) { setLoading(false); return }
        const currentRun = runs[0].run
        const summary = await api.summary(currentRun)

        // Helper to compute trend from last two points
        async function trendPct(tag: string): Promise<{ trend: number; latest?: { step: number; value: number } }> {
          const series = await api.series(currentRun, tag, 200)
          if (!series.length) return { trend: 0 }
          const latest = series[series.length - 1]
          const prev = series.length > 1 ? series[series.length - 2] : undefined
          let trend = 0
          if (prev && Math.abs(prev.value) > 1e-9) {
            trend = ((latest.value - prev.value) / prev.value) * 100
            if (!Number.isFinite(trend)) trend = 0
          }
          return { trend, latest: { step: latest.step, value: latest.value } }
        }

        const metricData: MetricCardData[] = []

        // Loss
        {
          const { trend, latest } = await trendPct('train/loss')
          const val = summary['train/loss'] ?? latest?.value
          if (val !== undefined) {
            metricData.push({
              label: 'Training Loss',
              value: val.toFixed(2),
              trend,
              status: trend < 0 ? 'good' : 'warning',
              footer: latest ? `Step ${latest.step}` : '—',
            })
          }
        }

        // MFU removed for mixed precision; rely on throughput and tflops

        // Throughput (GPU)
        {
          const { trend, latest } = await trendPct('throughput/tokens_per_s_gpu')
          const val = summary['throughput/tokens_per_s_gpu'] ?? latest?.value
          if (val !== undefined) {
            metricData.push({
              label: 'Throughput',
              value: `${(val / 1000).toFixed(1)}k tok/s`,
              trend,
              status: 'good',
              footer: 'Tokens per second',
            })
          }
        }

        setMetrics(metricData)
      } catch (error) {
        console.error("Failed to fetch metrics:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
    // Refresh every 10 seconds
    const interval = setInterval(fetchMetrics, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-4 px-4 lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="@container/card">
            <CardHeader>
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-32" />
            </CardHeader>
          </Card>
        ))}
      </div>
    )
  }

  if (metrics.length === 0) {
    return (
      <div className="px-4 lg:px-6">
        <Card>
          <CardHeader>
            <CardTitle>No Data Available</CardTitle>
          <CardDescription>
            No metrics found for the selected run
          </CardDescription>
          </CardHeader>
        </Card>
      </div>
    )
  }

  return (
    <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 px-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:shadow-xs lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
      {metrics.map((metric, i) => (
        <Card key={i} className="@container/card">
          <CardHeader>
            <CardDescription>{metric.label}</CardDescription>
            <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
              {metric.value}
            </CardTitle>
            <CardAction>
              <Badge
                variant={
                  metric.status === "error" ? "destructive" :
                  metric.status === "warning" ? "outline" :
                  "outline"
                }
              >
                {metric.trend !== 0 && (
                  <>
                    {metric.trend > 0 ? <TrendingUp className="size-3" /> : <TrendingDown className="size-3" />}
                    {metric.trend > 0 ? "+" : ""}{metric.trend.toFixed(1)}%
                  </>
                )}
                {metric.trend === 0 && "—"}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardFooter className="flex-col items-start gap-1.5 text-sm">
            <div className="line-clamp-1 flex gap-2 font-medium items-center">
              {metric.status === "error" && <AlertTriangle className="size-4 text-destructive" />}
              {metric.footer}
            </div>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
