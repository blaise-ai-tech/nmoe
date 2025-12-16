"use client"

import { useEffect, useState } from "react"
import { useVisibilityPoll } from "@/hooks/use-visibility-poll"
import { TrendingUp, TrendingDown } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useRuns } from "@/lib/run-context"

interface StatCardProps {
  title: string
  value: string
  description: string
  trend?: number
  loading?: boolean
}

function StatCard({ title, value, description, trend, loading }: StatCardProps) {
  const trendColor = trend && trend > 0 ? "text-green-600" : trend && trend < 0 ? "text-red-600" : "text-muted-foreground"
  const TrendIcon = trend && trend > 0 ? TrendingUp : TrendingDown

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
          {trend !== undefined && (
            <div className={`flex items-center gap-1 text-xs font-medium ${trendColor}`}>
              <TrendIcon className="h-3 w-3" />
              {Math.abs(trend).toFixed(1)}%
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">
            <div className="h-8 w-24 bg-muted animate-pulse rounded" />
            <div className="h-4 w-32 bg-muted animate-pulse rounded" />
          </div>
        ) : (
          <>
            <div className="text-3xl font-bold">{value}</div>
            <CardDescription className="text-xs mt-1">{description}</CardDescription>
          </>
        )}
      </CardContent>
    </Card>
  )
}

export function SummaryStats() {
  const { currentRun } = useRuns()
  const [stats, setStats] = useState({
    loss: { current: 0, min: 0, trend: 0 },
    throughput: { current: 0 },
  })
  const [initialLoading, setInitialLoading] = useState(true)

  useVisibilityPoll(async () => {
    if (!currentRun) { setInitialLoading(false); return }
    try {
      const sres = await fetch(`/api/summary?run=${encodeURIComponent(currentRun)}`, { cache: 'no-store' })
      if (!sres.ok) throw new Error(await sres.text())
      const sj = await sres.json()
      const summary = sj.summary || {}
      // compute trend for loss from last two points
      let trend = 0, lossCurrent = summary['train/loss'] || 0, lossMin = lossCurrent
      try {
        const seriesRes = await fetch(`/api/series?run=${encodeURIComponent(currentRun)}&tag=${encodeURIComponent('train/loss')}&points=400`, { cache: 'no-store' })
        const ser = (await seriesRes.json()).series || []
        if (ser.length) {
          lossCurrent = ser[ser.length - 1].value
          lossMin = ser.reduce((m: number, p: any) => Math.min(m, p.value), ser[0].value)
          if (ser.length > 1) {
            const prev = ser[ser.length - 2].value
            if (Math.abs(prev) > 1e-9) {
              trend = ((lossCurrent - prev) / prev) * 100
              if (!Number.isFinite(trend)) trend = 0
            }
          }
        }
      } catch {}
      setStats({
        loss: { current: lossCurrent, min: lossMin, trend },
        throughput: { current: summary['throughput/tokens_per_s_gpu'] || 0 },
      })
      setInitialLoading(false)
    } catch (error) {
      console.error("Failed to fetch summary stats:", error)
      setInitialLoading(false)
    }
  }, 10000, !!currentRun)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
      <StatCard
        title="Training Loss"
        value={stats.loss.current.toFixed(2)}
        description="Loss stable/decreasing"
        trend={stats.loss.trend}
        loading={initialLoading}
      />
      <StatCard
        title="Throughput"
        value={`${(stats.throughput.current / 1000).toFixed(1)}k tok/s`}
        description="Tokens per second"
        loading={initialLoading}
      />
    </div>
  )
}
