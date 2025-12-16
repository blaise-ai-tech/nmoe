"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { api } from "@/lib/api"
import { parseRunName } from "./utils"

interface RunContextType {
  availableRuns: string[]
  selectedRuns: string[]
  currentRun: string
  setSelectedRuns: (runs: string[]) => void
  setCurrentRun: (run: string) => void
  loading: boolean
}

const RunContext = createContext<RunContextType | undefined>(undefined)

export function RunProvider({ children }: { children: React.ReactNode }) {
  const [availableRuns, setAvailableRuns] = useState<string[]>([])
  const [selectedRuns, setSelectedRuns] = useState<string[]>([])
  const [currentRun, setCurrentRun] = useState<string>("")
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchRuns() {
      try {
        const list = await api.runs()
        const names = list.map(x => x.run)
        setAvailableRuns(names)
        // Default selection to latest if empty; otherwise retain if still present
        setCurrentRun(prev => (prev && names.includes(prev)) ? prev : (names[0] || ""))
        setSelectedRuns(prev => (prev.length ? prev.filter(r => names.includes(r)) : (names[0] ? [names[0]] : [])))
      } catch (error) {
        console.error("Failed to fetch runs:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchRuns()
    // Refresh every 30 seconds
    const interval = setInterval(fetchRuns, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <RunContext.Provider
      value={{
        availableRuns,
        selectedRuns,
        currentRun,
        setSelectedRuns,
        setCurrentRun,
        loading,
      }}
    >
      {children}
    </RunContext.Provider>
  )
}

export function useRuns() {
  const context = useContext(RunContext)
  if (context === undefined) {
    throw new Error("useRuns must be used within a RunProvider")
  }
  return context
}
