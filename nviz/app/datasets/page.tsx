import { AppSidebar } from "@/components/app-sidebar"
import { SiteHeader } from "@/components/site-header"
import { DatasetsViewer } from "@/components/datasets-viewer"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"
import { Suspense } from "react"
import { Skeleton } from "@/components/ui/skeleton"

export default function DatasetsPage() {
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
        <SiteHeader mode="datasets" />
        <div className="flex flex-1 flex-col">
          <div className="@container/main flex flex-1 flex-col gap-2">
            <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
              <div className="px-4 lg:px-6">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h1 className="text-3xl font-bold">Datasets</h1>
                    <p className="text-muted-foreground">Browse shared `/data` artifacts</p>
                  </div>
                </div>
              </div>
              <div className="px-4 lg:px-6">
                <Suspense fallback={<Skeleton className="h-[600px] w-full" />}>
                  <DatasetsViewer />
                </Suspense>
              </div>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
