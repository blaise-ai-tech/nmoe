"use client"

import * as React from "react"
import { Brain, Play, CheckCircle2, Clock } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from "@/components/ui/sidebar"
import { ThemeSwitcher } from "@/components/theme-switcher"
import { api } from "@/lib/api"
import { parseRunName } from "@/lib/utils"
import { useRuns } from "@/lib/run-context"
import { Checkbox } from "@/components/ui/checkbox"
import { formatRunLabel } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"

interface RunInfo {
  name: string
  displayName: string
  status: "running" | "completed" | "failed" | "killed" | "unknown"
  currentLoss?: number
  lastUpdate?: number
}

export function RunsSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const [runs, setRuns] = React.useState<RunInfo[]>([])
  const [loading, setLoading] = React.useState(true)
  const { selectedRuns, setSelectedRuns } = useRuns()

  React.useEffect(() => {
    async function fetchRuns() {
      try {
        const list = await api.runs()
        const runsInfo: RunInfo[] = []
        for (const r of list) {
          const displayName = formatRunLabel(r.run, { compact: true })
          const status: RunInfo["status"] =
            r.status === "running" ? "running" :
            r.status === "completed" ? "completed" :
            r.status === "failed" ? "failed" :
            r.status === "killed" ? "killed" :
            ((Date.now() - (r.last_ts || 0)) < 3000 ? "running" : "unknown")
          let currentLoss: number | undefined
          try {
            const summary = await api.summary(r.run)
            currentLoss = summary['train/loss']
          } catch {}
          runsInfo.push({ name: r.run, displayName, status, currentLoss, lastUpdate: r.last_ts / 1000 })
        }
        setRuns(runsInfo)

        // Selection is handled by the run context to prevent deselection on refetch
      } catch (error) {
        console.error("Failed to fetch runs:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchRuns()
    const interval = setInterval(fetchRuns, 10000)
    return () => clearInterval(interval)
  }, [])

  const toggleRun = (runName: string) => {
    if (selectedRuns.includes(runName)) {
      // Don't allow deselecting all runs
      if (selectedRuns.length > 1) {
        setSelectedRuns(selectedRuns.filter(r => r !== runName))
      }
    } else {
      setSelectedRuns([...selectedRuns, runName])
    }
  }

  const getRunColor = (index: number) => {
    const colors = [
      "hsl(var(--chart-1))",
      "hsl(var(--chart-2))",
      "hsl(var(--chart-3))",
      "hsl(var(--chart-4))",
      "hsl(var(--chart-5))",
    ]
    return colors[index % colors.length]
  }

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              size="lg"
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <a href="/" className="flex items-center">
                <span style={{ fontSize: "1.3em" }} className="font-semibold">nmoe</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>
            Runs ({runs.length})
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="space-y-1">
              {loading ? (
                <div className="px-3 py-2 text-xs text-muted-foreground">
                  Loading runs...
                </div>
              ) : runs.length > 0 ? (
                runs.map((run, index) => (
                  <div
                    key={run.name}
                    className="flex items-center gap-2 px-3 py-2 rounded-md hover:bg-accent/50 cursor-pointer transition-colors"
                    onClick={() => toggleRun(run.name)}
                  >
                    <Checkbox
                      checked={selectedRuns.includes(run.name)}
                      onCheckedChange={() => toggleRun(run.name)}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: getRunColor(index) }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium truncate">
                          {run.displayName}
                        </span>
                        {run.status === "running" && (
                          <Badge variant="outline" className="text-xs flex items-center gap-1 px-1.5 py-0">
                            <Clock className="h-2.5 w-2.5" />
                            Live
                          </Badge>
                        )}
                        {run.status === "failed" && (
                          <Badge variant="destructive" className="text-xs px-1.5 py-0">
                            Failed
                          </Badge>
                        )}
                        {run.status === "killed" && (
                          <Badge variant="outline" className="text-xs px-1.5 py-0">
                            Killed
                          </Badge>
                        )}
                      </div>
                      {run.currentLoss !== undefined && (
                        <div className="text-xs text-muted-foreground font-mono">
                          Loss: {run.currentLoss.toFixed(3)}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="px-3 py-2 text-xs text-muted-foreground">
                  No runs available
                </div>
              )}
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <ThemeSwitcher />
      </SidebarFooter>
    </Sidebar>
  )
}
