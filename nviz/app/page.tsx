import { AppSidebar } from "@/components/app-sidebar"
import { TrainingLossChart } from "@/components/training-loss-chart"
import { TrainingMetricsCards } from "@/components/training-metrics-cards"
import { RouterHealthHeatmap } from "@/components/router-health-heatmap"
import { RouterMetricsOverTime } from "@/components/router-metrics-over-time"
import { SiteHeader } from "@/components/site-header"
import { SummaryStats } from "@/components/summary-stats"
import { GpuMonitor } from "@/components/gpu-monitor"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"

export default function Home() {
  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": "calc(var(--spacing) * 72)",
          "--header-height": "calc(var(--spacing) * 12)",
        } as React.CSSProperties
      }
    >
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader />
        <div className="flex flex-1 flex-col">
          <div className="@container/main flex flex-1 flex-col gap-2">
            <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
              <div className="px-4 lg:px-6">
                <SummaryStats />
              </div>
              <div className="px-4 lg:px-6">
                <TrainingLossChart />
              </div>
              <div className="px-4 lg:px-6">
                <GpuMonitor />
              </div>
              <div className="px-4 lg:px-6">
                <RouterHealthHeatmap />
              </div>
              <div className="px-4 lg:px-6">
                <RouterMetricsOverTime />
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
