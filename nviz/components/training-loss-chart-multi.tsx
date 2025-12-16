"use client"

import * as React from "react"
import { Area, AreaChart, CartesianGrid, Line, LineChart, XAxis, YAxis, Brush, ResponsiveContainer } from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"
import { useVisibilityPoll } from "@/hooks/use-visibility-poll"
import { formatRunLabel } from "@/lib/utils"
import { useRuns } from "@/lib/run-context"

export function TrainingLossChartMulti() {
  const { selectedRuns } = useRuns()
  const [runsData, setRunsData] = React.useState<Record<string, Array<{ step: number; loss: number }>>>({})
  const [loading, setLoading] = React.useState(true)

  useVisibilityPoll(async () => {
    if (selectedRuns.length === 0) { setLoading(false); return }
    try {
      const mapped: Record<string, Array<{ step: number; loss: number }>> = {}
      const results = await Promise.all(selectedRuns.map(async (run) => {
        const s = await api.series(run, 'train/loss', 2000)
        return [run, s.map(d => ({ step: d.step + 1, loss: d.value }))] as const
      }))
      for (const [run, arr] of results) mapped[run] = arr
      setRunsData(mapped)
    } catch (error) {
      console.error("Failed to fetch multi-run loss:", error)
    } finally {
      setLoading(false)
    }
  }, 5000, selectedRuns.length > 0)

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Training Loss</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[600px] w-full" />
        </CardContent>
      </Card>
    )
  }

  // Merge all runs data into single array with run identifier
  const mergedData: Array<{ step: number; [key: string]: number }> = []
  const allSteps = new Set<number>()

  // Collect all unique steps
  Object.values(runsData).forEach(data => {
    data.forEach(d => allSteps.add(d.step))
  })

  // Create merged data structure
  Array.from(allSteps).sort((a, b) => a - b).forEach(step => {
    const dataPoint: any = { step }
    Object.entries(runsData).forEach(([run, data]) => {
      const point = data.find(d => d.step === step)
      if (point) {
        dataPoint[run] = point.loss
      }
    })
    mergedData.push(dataPoint)
  })

  // Calculate stats for description
  const allLosses = Object.values(runsData).flatMap(data => data.map(d => d.loss))
  const minLoss = allLosses.length > 0 ? Math.min(...allLosses) : 0
  const maxLoss = allLosses.length > 0 ? Math.max(...allLosses) : 0

  // Build dynamic chart config
  const chartConfig: ChartConfig = {}
  const colors = [
    "hsl(var(--chart-1))",
    "hsl(var(--chart-2))",
    "hsl(var(--chart-3))",
    "hsl(var(--chart-4))",
    "hsl(var(--chart-5))",
  ]

  selectedRuns.forEach((run, index) => {
    const displayName = formatRunLabel(run, { compact: true })
    chartConfig[run] = {
      label: displayName,
      color: colors[index % colors.length],
    }
  })

  return (
    <Card className="@container/card">
      <CardHeader>
        <CardTitle>Training Loss</CardTitle>
        <CardDescription>
          {selectedRuns.length} run{selectedRuns.length > 1 ? "s" : ""} • Min: {minLoss.toFixed(3)}
        </CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[600px] w-full"
        >
          <LineChart data={mergedData}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.15} stroke="hsl(var(--muted-foreground))" />
            <XAxis
              dataKey="step"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={(value) => `${value}`}
              tick={{ fill: 'hsl(var(--foreground))' }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => value.toFixed(1)}
              domain={[minLoss * 0.95, maxLoss * 1.05]}
              tick={{ fill: 'hsl(var(--foreground))' }}
            />
            <ChartTooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                return (
                  <div className="rounded-lg border p-2 shadow-sm" style={{ backgroundColor: "rgba(255, 255, 255, 0.03)" }}>
                    <div className="grid gap-2">
                      <div className="font-medium">Step {payload[0].payload.step}</div>
                      {payload.map((entry, index) => {
                        const runName = entry.dataKey as string
                        const displayName = chartConfig[runName]?.label || runName
                        return (
                          <div key={index} className="flex items-center gap-2">
                            <div
                              className="w-2 h-2 rounded-full"
                              style={{ backgroundColor: entry.color }}
                            />
                            <span className="text-xs text-muted-foreground">{displayName}:</span>
                            <span className="text-xs font-mono font-medium">{Number(entry.value).toFixed(3)}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              }}
            />
            {selectedRuns.map((run) => (
              <Line
                key={run}
                dataKey={run}
                type="monotone"
                stroke={chartConfig[run].color}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ))}
            <Brush
              dataKey="step"
              height={30}
              stroke="hsl(var(--muted-foreground))"
              fill="hsl(var(--muted))"
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
