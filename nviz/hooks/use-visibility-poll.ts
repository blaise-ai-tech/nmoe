import * as React from "react"

export function useVisibilityPoll(cb: () => void | Promise<void>, intervalMs: number, enabled = true) {
  React.useEffect(() => {
    if (!enabled) return
    let disposed = false
    let timer: any
    const run = async () => {
      try { await cb() } catch {}
    }
    const start = () => { if (!timer) timer = setInterval(run, intervalMs) }
    const stop = () => { if (timer) { clearInterval(timer); timer = null } }
    run()
    start()
    const onVis = () => { if (document.hidden) stop(); else { run(); start() } }
    document.addEventListener('visibilitychange', onVis)
    return () => { disposed = true; stop(); document.removeEventListener('visibilitychange', onVis) }
  }, [enabled, intervalMs, cb])
}

