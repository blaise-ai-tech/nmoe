"use client"

import * as React from "react"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"

import {
  Card,
  CardAction,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"
import { useVisibilityPoll } from "@/hooks/use-visibility-poll"
import { useRuns } from "@/lib/run-context"

const chartConfig = {
  loss: {
    label: "Loss",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig

export function TrainingLossChart() {
  const { currentRun } = useRuns()
  const [data, setData] = React.useState<Array<{ step: number; loss: number }>>([])
  const [initialLoading, setInitialLoading] = React.useState(true)

  useVisibilityPoll(async () => {
    if (!currentRun) { setInitialLoading(false); return }
    try {
      const lossData = await api.series(currentRun, "train/loss", 2000)
      setData(lossData.map(point => ({ step: point.step + 1, loss: point.value })))
      setInitialLoading(false)
    } catch (error) {
      console.error("Failed to fetch loss data:", error)
      setInitialLoading(false)
    }
  }, 2000, !!currentRun)

  const filteredData = data // Show all steps

  if (initialLoading) {
    return (
      <Card className="@container/card">
        <CardHeader>
          <Skeleton className="h-6 w-40" />
          <Skeleton className="h-4 w-64" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[250px] w-full" />
        </CardContent>
      </Card>
    )
  }

  if (data.length === 0) {
    return (
      <Card className="@container/card">
        <CardHeader>
          <CardTitle>Training Loss</CardTitle>
          <CardDescription>No loss data available</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-[250px] text-muted-foreground">
          No loss series available
        </CardContent>
      </Card>
    )
  }

  const minLoss = Math.min(...filteredData.map(d => d.loss))
  const maxLoss = Math.max(...filteredData.map(d => d.loss))
  const latestLoss = filteredData[filteredData.length - 1]?.loss || 0
  const currentStep = filteredData[filteredData.length - 1]?.step || 0

  return (
    <Card className="@container/card">
      <CardHeader>
        <CardTitle>Training Loss</CardTitle>
        <CardDescription>
          Current: {latestLoss.toFixed(3)} • Min: {minLoss.toFixed(3)} • Step {currentStep}
        </CardDescription>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[600px] w-full"
        >
          <AreaChart data={filteredData}>
            <defs>
              <linearGradient id="fillLoss" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-loss)"
                  stopOpacity={0.8}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-loss)"
                  stopOpacity={0.1}
                />
              </linearGradient>
            </defs>
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
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => `Step ${value}`}
                  formatter={(value) => [`${Number(value).toFixed(3)}`, "Loss"]}
                  indicator="dot"
                />
              }
            />
            <Area
              dataKey="loss"
              type="monotone"
              fill="url(#fillLoss)"
              stroke="var(--color-loss)"
              strokeWidth={2}
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
