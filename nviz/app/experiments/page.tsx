import { AppSidebar } from "@/components/app-sidebar"
import { SiteHeader } from "@/components/site-header"
import { TrainingLossChartMulti } from "@/components/training-loss-chart-multi"
import { RunsSidebar } from "@/components/runs-sidebar"
import { ExperimentsRunsTable } from "@/components/experiments-runs-table"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"

export default function ExperimentsPage() {
  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": "calc(var(--spacing) * 72)",
          "--header-height": "calc(var(--spacing) * 12)",
        } as React.CSSProperties
      }
    >
      <RunsSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader mode="experiment" />
        <div className="flex flex-1 flex-col">
          <div className="@container/main flex flex-1 flex-col gap-2">
            <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
              <div className="px-4 lg:px-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h1 className="text-3xl font-bold">Experiments</h1>
                    <p className="text-muted-foreground">Compare multiple training runs</p>
                  </div>
                </div>
              </div>

              <div className="px-4 lg:px-6">
                <TrainingLossChartMulti />
              </div>

              <div className="px-4 lg:px-6">
                <ExperimentsRunsTable />
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
