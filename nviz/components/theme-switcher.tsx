"use client"

import { Moon, Sun, Palette } from "lucide-react"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // useEffect only runs on the client, so now we can safely show the UI
  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <div className="flex items-center gap-2">
        <Palette className="h-4 w-4 text-muted-foreground" />
        <div className="w-[180px] h-8" />
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <Palette className="h-4 w-4 text-muted-foreground" />
      <Select value={theme} onValueChange={setTheme}>
        <SelectTrigger className="w-[180px] h-8 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="rose-pine">
            <div className="flex items-center gap-2">
              <Moon className="h-3 w-3" />
              <span>Rosé Pine</span>
            </div>
          </SelectItem>
          <SelectItem value="tomorrow-night-bright">
            <div className="flex items-center gap-2">
              <Moon className="h-3 w-3" />
              <span>Tomorrow Night</span>
            </div>
          </SelectItem>
          <SelectItem value="light">
            <div className="flex items-center gap-2">
              <Sun className="h-3 w-3" />
              <span>Rosé Pine Light</span>
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}
