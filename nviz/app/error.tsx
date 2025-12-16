"use client"

export default function GlobalError({ error, reset }: { error: Error & { digest?: string }, reset: () => void }) {
  return (
    <html>
      <body>
        <div style={{ padding: 24 }}>
          <h1 style={{ fontSize: 18, marginBottom: 8 }}>Something went wrong</h1>
          <p style={{ opacity: 0.7, marginBottom: 16 }}>{error.message}</p>
          <button
            onClick={() => reset()}
            style={{
              padding: '6px 10px',
              borderRadius: 6,
              background: 'hsl(var(--accent))',
              color: 'hsl(var(--accent-foreground))',
            }}
          >Try again</button>
        </div>
      </body>
    </html>
  )
}

