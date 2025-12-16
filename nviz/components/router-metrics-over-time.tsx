"use client"

import { useEffect, useState } from "react"
import { Line, LineChart, CartesianGrid, XAxis, YAxis, ResponsiveContainer } from "recharts"

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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api } from "@/lib/api"
import { useRuns } from "@/lib/run-context"

interface LayerTimeSeries {
  layer: number
  data: Array<{ step: number; value: number }>
}

const chartConfig = {
  value: {
    label: "Value",
    color: "hsl(var(--chart-3))",
  },
} satisfies ChartConfig

export function RouterMetricsOverTime() {
  const { currentRun } = useRuns()
  const [cvData, setCvData] = useState<LayerTimeSeries[]>([])
  const [entropyData, setEntropyData] = useState<LayerTimeSeries[]>([])
  const [expertsData, setExpertsData] = useState<LayerTimeSeries[]>([])
  const [initialLoading, setInitialLoading] = useState(true)
  const [selectedLayers, setSelectedLayers] = useState<"all" | "problem">("all")
  const [activeTab, setActiveTab] = useState<"cv" | "entropy" | "experts">("cv")

  useEffect(() => {
    let cancelled = false
    let interval: any
    async function fetchTimeSeriesData() {
      if (!currentRun) { setInitialLoading(false); return }
      try {
        const layers = Array.from({ length: 16 }, (_, i) => i)
        const load = async (metric: string) => {
          const arr = await Promise.all(layers.map(async layer => {
            const tag = `router/layer_${String(layer).padStart(2,'0')}/${metric}`
            const s = await api.series(currentRun, tag, 1000)
            return { layer, data: s.map(p => ({ step: p.step, value: p.value })) }
          }))
          // keep only layers with data
          return arr.filter(x => x.data.length)
        }
        const [cv, ent, exp] = await Promise.all([load('cv'), load('entropy'), load('experts_active')])
        if (cancelled) return
        setCvData(cv)
        setEntropyData(ent)
        setExpertsData(exp)
        setInitialLoading(false)
      } catch (err) {
        console.error('Failed to fetch router timeseries:', err)
        if (!cancelled) setInitialLoading(false)
      }
    }
    fetchTimeSeriesData()
    interval = setInterval(fetchTimeSeriesData, 30000)
    const onVis = () => {
      if (document.hidden) { clearInterval(interval); interval = null }
      else { fetchTimeSeriesData(); if (!interval) interval = setInterval(fetchTimeSeriesData, 30000) }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => { cancelled = true; if (interval) clearInterval(interval); document.removeEventListener('visibilitychange', onVis) }
  }, [currentRun])

  if (initialLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[400px] w-full" />
        </CardContent>
      </Card>
    )
  }

  // Identify problem layers based on latest values
  const problemLayers = cvData
    .filter(layer => {
      const latestCV = layer.data[layer.data.length - 1]?.value || 0
      return latestCV > 2.0
    })
    .map(l => l.layer)

  const layersToShow = selectedLayers === "problem"
    ? cvData.filter(l => problemLayers.includes(l.layer))
    : cvData

  // Generate color palette for layers using theme colors
  const getLayerColor = (layerIndex: number, total: number) => {
    const colors = [
      'hsl(var(--chart-1))',
      'hsl(var(--chart-2))',
      'hsl(var(--chart-3))',
      'hsl(var(--chart-4))',
      'hsl(var(--chart-5))',
    ]
    return colors[layerIndex % colors.length]
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Router Metrics Over Time</CardTitle>
            <CardDescription>
              Track how router metrics evolve across training
            </CardDescription>
          </div>
          <Select value={selectedLayers} onValueChange={(v) => setSelectedLayers(v as any)}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Layers</SelectItem>
              <SelectItem value="problem">Problem Layers ({problemLayers.length})</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent data-active-tab={activeTab}>
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="space-y-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="cv">Load Balance (CV)</TabsTrigger>
            <TabsTrigger value="entropy">Entropy</TabsTrigger>
            <TabsTrigger value="experts">Active Experts</TabsTrigger>
          </TabsList>

          <TabsContent value="cv" className="space-y-4">
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} stroke="hsl(var(--muted-foreground))" />
                  <XAxis
                    dataKey="step"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(value) => `${value}`}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis
                    label={{ value: 'CV', angle: -90, position: 'insideLeft', fill: 'hsl(var(--foreground))' }}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      return (
                        <div className="rounded-lg border p-2 shadow-sm" style={{ backgroundColor: "rgba(255, 255, 255, 0.12)" }}>
                          <div className="grid gap-2">
                            <div className="font-medium">Step {payload[0].payload.step}</div>
                            {payload.map((entry, index) => (
                              <div key={index} className="flex items-center gap-2 text-xs">
                                <div
                                  className="h-2 w-2 rounded-full"
                                  style={{ backgroundColor: entry.color }}
                                />
                                <span>{entry.name}: {Number(entry.value).toFixed(2)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    }}
                  />
                  {layersToShow.map((layer, idx) => (
                    <Line
                      key={layer.layer}
                      data={layer.data}
                      type="monotone"
                      dataKey="value"
                      name={`Layer ${layer.layer}`}
                      stroke={getLayerColor(idx, layersToShow.length)}
                      strokeWidth={1.5}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="text-xs text-muted-foreground">
              Lower is better. CV &lt; 1.0 = excellent load balance, &gt; 2.0 = poor balance
            </div>
          </TabsContent>

          <TabsContent value="entropy" className="space-y-4">
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} stroke="hsl(var(--muted-foreground))" />
                  <XAxis
                    dataKey="step"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis
                    label={{ value: 'Entropy', angle: -90, position: 'insideLeft', fill: 'hsl(var(--foreground))' }}
                    domain={[0, 1]}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      return (
                        <div className="rounded-lg border p-2 shadow-sm" style={{ backgroundColor: "rgba(255, 255, 255, 0.12)" }}>
                          <div className="grid gap-2">
                            <div className="font-medium">Step {payload[0].payload.step}</div>
                            {payload.map((entry, index) => (
                              <div key={index} className="flex items-center gap-2 text-xs">
                                <div
                                  className="h-2 w-2 rounded-full"
                                  style={{ backgroundColor: entry.color }}
                                />
                                <span>{entry.name}: {Number(entry.value).toFixed(3)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    }}
                  />
                  {entropyData
                    .filter(l => selectedLayers === "all" || problemLayers.includes(l.layer))
                    .map((layer, idx) => (
                      <Line
                        key={layer.layer}
                        data={layer.data}
                        type="monotone"
                        dataKey="value"
                        name={`Layer ${layer.layer}`}
                        stroke={getLayerColor(idx, layersToShow.length)}
                        strokeWidth={1.5}
                        dot={false}
                      />
                    ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="text-xs text-muted-foreground">
              Higher is better. Entropy &gt; 0.8 = uniform distribution, &lt; 0.6 = concentrated routing
            </div>
          </TabsContent>

          <TabsContent value="experts" className="space-y-4">
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} stroke="hsl(var(--muted-foreground))" />
                  <XAxis
                    dataKey="step"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <YAxis
                    label={{ value: 'Active Experts', angle: -90, position: 'insideLeft', fill: 'hsl(var(--foreground))' }}
                    domain={[0, 64]}
                    tick={{ fill: 'hsl(var(--foreground))' }}
                  />
                  <ChartTooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      return (
                        <div className="rounded-lg border p-2 shadow-sm" style={{ backgroundColor: "rgba(255, 255, 255, 0.12)" }}>
                          <div className="grid gap-2">
                            <div className="font-medium">Step {payload[0].payload.step}</div>
                            {payload.map((entry, index) => (
                              <div key={index} className="flex items-center gap-2 text-xs">
                                <div
                                  className="h-2 w-2 rounded-full"
                                  style={{ backgroundColor: entry.color }}
                                />
                                <span>{entry.name}: {Number(entry.value).toFixed(0)}/64</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )
                    }}
                  />
                  {expertsData
                    .filter(l => selectedLayers === "all" || problemLayers.includes(l.layer))
                    .map((layer, idx) => (
                      <Line
                        key={layer.layer}
                        data={layer.data}
                        type="monotone"
                        dataKey="value"
                        name={`Layer ${layer.layer}`}
                        stroke={getLayerColor(idx, layersToShow.length)}
                        strokeWidth={1.5}
                        dot={false}
                      />
                    ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="text-xs text-muted-foreground">
              Higher is better. Ideally all 64 experts should be active.
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
