interface CacheEntry<T> {
  data: T
  timestamp: number
}

class DataCache {
  private cache: Map<string, CacheEntry<any>> = new Map()
  private ttl: number = 10000 // 10 seconds default TTL
  private maxEntries = 500

  set<T>(key: string, data: T, ttl?: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
    })
    if (this.cache.size > this.maxEntries) {
      // delete oldest entries
      const entries = Array.from(this.cache.entries())
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp)
      const toDelete = entries.slice(0, this.cache.size - this.maxEntries)
      for (const [k] of toDelete) this.cache.delete(k)
    }
  }

  get<T>(key: string, ttl?: number): T | null {
    const entry = this.cache.get(key)
    if (!entry) return null

    const maxAge = ttl || this.ttl
    const age = Date.now() - entry.timestamp

    if (age > maxAge) {
      this.cache.delete(key)
      return null
    }

    return entry.data as T
  }

  has(key: string, ttl?: number): boolean {
    return this.get(key, ttl) !== null
  }

  clear(): void {
    this.cache.clear()
  }

  invalidate(pattern?: string): void {
    if (!pattern) {
      this.clear()
      return
    }

    // Remove entries matching pattern
    const regex = new RegExp(pattern)
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key)
      }
    }
  }

  // Get or fetch pattern
  async getOrFetch<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttl?: number
  ): Promise<T> {
    const cached = this.get<T>(key, ttl)
    if (cached !== null) {
      return cached
    }

    const data = await fetcher()
    this.set(key, data, ttl)
    return data
  }
}

export const dataCache = new DataCache()
