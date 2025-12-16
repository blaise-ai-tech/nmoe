"use client"

import { useEffect, useState } from "react"
import { X } from "lucide-react"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"
import { formatRunLabel } from "@/lib/utils"
import { useRuns } from "@/lib/run-context"

interface MetricDetailModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  metricTag: string
  metricName: string
}

export function MetricDetailModal({
  open,
  onOpenChange,
  metricTag,
  metricName,
}: MetricDetailModalProps) {
  const { selectedRuns } = useRuns()
  const [data, setData] = useState<Record<string, Array<{ step: number; value: number }>>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!open || !metricTag) return

    async function fetchData() {
      setLoading(true)
      try {
        const allData: Record<string, Array<{ step: number; value: number }>> = {}

        for (const run of selectedRuns) {
          try {
            const metricData = await api.series(run, metricTag, 2000)
            allData[run] = metricData.map(point => ({ step: point.step + 1, value: point.value }))
          } catch (error) {
            console.error(`Failed to fetch ${metricTag} for run ${run}:`, error)
          }
        }

        setData(allData)
      } catch (error) {
        console.error("Failed to fetch metric data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [open, metricTag, selectedRuns])

  // Create chart config dynamically for each run
  const chartConfig: ChartConfig = {}
  selectedRuns.forEach((run, idx) => {
    chartConfig[run] = {
      label: formatRunLabel(run, { compact: true }),
      color: `hsl(var(--chart-${(idx % 5) + 1}))`,
    }
  })

  // Merge data from all runs by step
  const mergedData: Array<{ step: number; [key: string]: number }> = []
  const allSteps = new Set<number>()

  Object.values(data).forEach(runData => {
    runData.forEach(point => allSteps.add(point.step))
  })

  Array.from(allSteps).sort((a, b) => a - b).forEach(step => {
    const dataPoint: any = { step }
    Object.entries(data).forEach(([run, runData]) => {
      const point = runData.find(p => p.step === step)
      if (point) {
        dataPoint[run] = point.value
      }
    })
    mergedData.push(dataPoint)
  })

  const latestValues = Object.entries(data).map(([run, runData]) => ({
    run,
    value: runData[runData.length - 1]?.value || 0,
  }))

  const minValue = Math.min(...mergedData.flatMap(d =>
    Object.entries(d).filter(([k]) => k !== 'step').map(([, v]) => v as number)
  ))
  const maxValue = Math.max(...mergedData.flatMap(d =>
    Object.entries(d).filter(([k]) => k !== 'step').map(([, v]) => v as number)
  ))

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="max-w-[90vw] max-h-[90vh] overflow-y-auto"
        style={{ backgroundColor: "rgba(223, 218, 217, 0.03)" }}
      >
        <DialogHeader>
          <DialogTitle>{metricName}</DialogTitle>
          <DialogDescription>
            {selectedRuns.length} run{selectedRuns.length > 1 ? 's' : ''} • Tag: {metricTag}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="space-y-4">
            <Skeleton className="h-[400px] w-full" />
          </div>
        ) : mergedData.length === 0 ? (
          <div className="flex items-center justify-center h-[400px] text-muted-foreground">
            No data available for this metric
          </div>
        ) : (
          <div className="space-y-6">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-xs text-muted-foreground">Min</div>
                <div className="text-lg font-mono">{minValue.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Max</div>
                <div className="text-lg font-mono">{maxValue.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Steps</div>
                <div className="text-lg font-mono">{mergedData.length}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Latest</div>
                <div className="text-lg font-mono">
                  {latestValues.length === 1
                    ? latestValues[0].value.toFixed(3)
                    : `${latestValues.length} values`}
                </div>
              </div>
            </div>

            {/* Chart */}
            <ChartContainer config={chartConfig} className="h-[400px] w-full">
              <AreaChart data={mergedData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis
                  dataKey="step"
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  tick={{ fill: 'hsl(var(--foreground))' }}
                />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  domain={[minValue * 0.95, maxValue * 1.05]}
                  tick={{ fill: 'hsl(var(--foreground))' }}
                  tickFormatter={(value) => value.toFixed(2)}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                {selectedRuns.map((run, idx) => (
                  <Area
                    key={run}
                    dataKey={run}
                    type="monotone"
                    fill={`hsl(var(--chart-${(idx % 5) + 1}))`}
                    fillOpacity={0.2}
                    stroke={`hsl(var(--chart-${(idx % 5) + 1}))`}
                    strokeWidth={2}
                  />
                ))}
              </AreaChart>
            </ChartContainer>

            {/* Run Legend */}
            {selectedRuns.length > 1 && (
              <div className="flex flex-wrap gap-4">
                {latestValues.map((item, idx) => (
                  <div key={item.run} className="flex items-center gap-2">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: `hsl(var(--chart-${(idx % 5) + 1}))` }}
                    />
                    <div className="text-sm">
                      <span className="font-medium">{item.run}</span>
                      <span className="text-muted-foreground ml-2">{item.value.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
