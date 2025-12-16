import { AppSidebar } from "@/components/app-sidebar"
import { SiteHeader } from "@/components/site-header"
import { TrainingLossChart } from "@/components/training-loss-chart"
import { RouterHealthHeatmap } from "@/components/router-health-heatmap"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"

export default function ProductionPage() {
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
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h1 className="text-3xl font-bold">Production Monitoring</h1>
                    <p className="text-muted-foreground">Real-time health monitoring with alerts</p>
                  </div>
                </div>
              </div>

              {/* TODO: Add health status panel */}
              {/* TODO: Add baseline comparison */}
              {/* TODO: Add alert configuration */}

              <div className="px-4 lg:px-6">
                <TrainingLossChart />
              </div>

              <div className="px-4 lg:px-6">
                <RouterHealthHeatmap />
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
