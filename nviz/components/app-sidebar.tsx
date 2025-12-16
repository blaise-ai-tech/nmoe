"use client"

import * as React from "react"
import {
  Activity,
  BarChart3,
  Brain,
  Gauge,
  LineChart,
  Network,
  Settings,
  Zap,
  FlaskConical,
  Shield,
  Monitor,
  Database,
} from "lucide-react"

// Metrics nav omitted for minimalism; main nav only
import { NavMain } from "@/components/nav-main"
import { ThemeSwitcher } from "@/components/theme-switcher"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { Badge } from "@/components/ui/badge"

const staticNavMain = [
  {
    title: "Overview",
    url: "/",
    icon: Activity,
  },
  {
    title: "Datasets",
    url: "/datasets",
    icon: Database,
  },
  {
    title: "Experiments",
    url: "/experiments",
    icon: FlaskConical,
  },
  {
    title: "Production",
    url: "/production",
    icon: Shield,
  },
]

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {

  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
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
        <NavMain items={staticNavMain} />
      </SidebarContent>
      <SidebarFooter>
        <ThemeSwitcher />
      </SidebarFooter>
    </Sidebar>
  )
}
