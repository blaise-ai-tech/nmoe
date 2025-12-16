import { describe, it, expect } from 'bun:test'
import React from 'react'
import { Window } from 'happy-dom'

// Ensure DOM globals exist before importing @testing-library/react.
const win = new Window()
Object.assign(globalThis, {
  window: win.window,
  document: win.document,
  navigator: win.navigator,
  HTMLElement: (win.window as any).HTMLElement,
})

const { render, screen, waitFor, cleanup } = await import('@testing-library/react')
const { TrainingMetricsCards } = await import('@/components/training-metrics-cards')

// Mock fetch for /api endpoints
function installFetchMocks() {
  const original = global.fetch
  const handler = async (input: RequestInfo, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input.toString()
    if (url.startsWith('/api/runs')) {
      return new Response(JSON.stringify({ runs: [{ run: 'runA', last_ts: Date.now(), last_step: 3 }] }), { status: 200 })
    }
    if (url.startsWith('/api/summary')) {
      return new Response(JSON.stringify({ run: 'runA', summary: {
        'train/loss': 6.2,
        'optimizer/lr': 3e-4,
        'throughput/tokens_per_s_gpu': 23500,
        'throughput/ms_per_step': 85.0,
      } }), { status: 200 })
    }
    if (url.startsWith('/api/series') && url.includes('tag=train/loss')) {
      const series = [
        { step: 2, ts_ms: 1, value: 7.5 },
        { step: 3, ts_ms: 2, value: 6.2 },
      ]
      return new Response(JSON.stringify({ run: 'runA', tag: 'train/loss', series }), { status: 200 })
    }
    return new Response(JSON.stringify({}), { status: 404 })
  }
  // @ts-ignore
  global.fetch = handler
  return () => { global.fetch = original }
}

describe('TrainingMetricsCards', () => {
  it('renders Loss and Throughput from /api responses', async () => {
    const restore = installFetchMocks()
    render(React.createElement(TrainingMetricsCards))
    // Wait for Loss value
    await waitFor(() => {
      expect(screen.getByText(/Training Loss/i)).toBeTruthy()
    })
    // Loss value formatted
    expect(screen.getByText('6.20')).toBeTruthy()
    // Throughput in k tok/s
    expect(screen.getByText(/23\.5k tok\/s/i)).toBeTruthy()
    restore()
    cleanup()
  })
})
