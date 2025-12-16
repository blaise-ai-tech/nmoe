"use client"

import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { useRuns } from "@/lib/run-context"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu"
import { ChevronDown } from "lucide-react"
import { formatRunLabel } from "@/lib/utils"

interface SiteHeaderProps {
  mode?: "single-run" | "experiment" | "datasets"
}

export function SiteHeader({ mode = "single-run" }: SiteHeaderProps) {
  const { currentRun, availableRuns, selectedRuns, setSelectedRuns, setCurrentRun, loading } = useRuns()

  // For experiment mode, show placeholder
  if (mode === "experiment") {
    return (
      <header className="flex h-(--header-height) shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)">
        <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
          <SidebarTrigger className="-ml-1" />
          <Separator
            orientation="vertical"
            className="mx-2 data-[orientation=vertical]:h-4"
          />
          <h1 className="text-base font-medium">Experiment Comparison</h1>
          <span className="text-xs text-muted-foreground ml-2">
            (Experiment naming coming soon)
          </span>
        </div>
      </header>
    )
  }

  if (mode === "datasets") {
    return (
      <header className="flex h-(--header-height) shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)">
        <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
          <SidebarTrigger className="-ml-1" />
          <Separator
            orientation="vertical"
            className="mx-2 data-[orientation=vertical]:h-4"
          />
          <h1 className="text-base font-medium">Datasets</h1>
          <span className="text-xs text-muted-foreground ml-2">
            (/data)
          </span>
        </div>
      </header>
    )
  }

  return (
    <header className="flex h-(--header-height) shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1" />
        <Separator
          orientation="vertical"
          className="mx-2 data-[orientation=vertical]:h-4"
        />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="gap-2">
              <h1 className="text-base font-medium">{loading ? "Loading..." : (currentRun ? formatRunLabel(currentRun, { compact: true }) : "No Run")}</h1>
              <ChevronDown className="h-4 w-4 opacity-50" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-64" style={{ backgroundColor: 'hsl(var(--background))' }}>
            <DropdownMenuLabel>Select Run</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {availableRuns.map((run) => (
              <DropdownMenuItem
                key={run}
                onClick={() => {
                  setCurrentRun(run)
                  setSelectedRuns([run])
                }}
                className={currentRun === run ? "bg-accent" : ""}
              >
                {formatRunLabel(run, { compact: true })}
              </DropdownMenuItem>
            ))}
            {availableRuns.length === 0 && (
              <DropdownMenuItem disabled>No runs available</DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
