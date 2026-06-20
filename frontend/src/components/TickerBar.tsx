import { useState, useEffect, useRef } from 'react'
import api from '../api'


const TICKER_SYMBOLS = [
  'NSE:INFY', 'NSE:RELIANCE', 'NSE:TCS', 'NSE:HDFCBANK',
  'NSE:WIPRO', 'NSE:BAJFINANCE', 'NSE:TATAMOTORS', 'NSE:ICICIBANK',
  'NSE:AXISBANK', 'NSE:SBIN',
]

interface TickerItem {
  symbol:     string
  price:      number
  prevClose:  number
  changePct:  number
  change:     number
  source:     'live' | 'last_known' | 'simulated'
}

interface MarketStatus {
  is_open:          boolean
  current_time_ist: string
  day:              string
  next_open:        string | null
}

// ── IST market hours check (client-side) ─────────────────────────────────────
function useMarketStatus() {
  const [status, setStatus] = useState<MarketStatus | null>(null)

  // Fetch from backend (accurate server-side check)
  async function fetchStatus() {
    try {
      const res = await api.get('/market_status')
      setStatus(res.data)
    } catch {
      // Fallback: compute locally in IST
      const now = new Date()
      const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }))
      const h = ist.getHours(), m = ist.getMinutes(), d = ist.getDay()
      const minOfDay = h * 60 + m
      const is_open  = d >= 1 && d <= 5 && minOfDay >= 555 && minOfDay <= 930  // 9:15–15:30
      setStatus({
        is_open,
        current_time_ist: `${h.toString().padStart(2,'0')}:${m.toString().padStart(2,'0')}`,
        day: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d],
        next_open: is_open ? null : '09:15 IST',
      })
    }
  }

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 60_000)  // re-check every minute
    return () => clearInterval(id)
  }, [])

  return status
}

// ── Parse backend quote response ─────────────────────────────────────────────
function parseQuotes(raw: any, symbols: string[]): Record<string, Partial<TickerItem>> {
  if (!raw || raw.error) return {}
  const out: Record<string, Partial<TickerItem>> = {}
  const quotes = raw.quotes || {}

  if (raw.source === 'kite') {
    for (const sym of symbols) {
      const q = quotes[sym]
      if (!q) continue
      const price    = q.last_price ?? q.ohlc?.close ?? 0
      const prevClose = q.ohlc?.close ?? price
      out[sym] = { price, prevClose, source: 'live' }
    }
  } else {
    // Yahoo Finance shape
    for (const sym of symbols) {
      const q = quotes[sym]
      if (!q) continue
      const price     = q.price ?? q.regularMarketPrice ?? 0
      const prevClose = q.previousClose ?? q.regularMarketPreviousClose ?? price
      out[sym] = { price, prevClose, source: 'last_known' }
    }
  }
  return out
}

export default function TickerBar() {
  const marketStatus = useMarketStatus()
  const [tickers, setTickers]       = useState<TickerItem[]>([])
  const [lastFetch, setLastFetch]   = useState<Date | null>(null)
  const prevRef = useRef<Record<string, number>>({})

  // ── Fetch real prices ─────────────────────────────────────────────────────
  async function fetchPrices() {
    try {
      const res = await api.post('/quotes', {
        symbols: TICKER_SYMBOLS.join(','),
      })
      const parsed = parseQuotes(res.data, TICKER_SYMBOLS)
      setLastFetch(new Date())

      setTickers(prev => {
        const updated: TickerItem[] = TICKER_SYMBOLS.map(sym => {
          const q      = parsed[sym]
          const price  = q?.price ?? prev.find(p => p.symbol === sym)?.price ?? 0
          const prev0  = prevRef.current[sym] ?? price
          const prevClose = q?.prevClose ?? prev.find(p => p.symbol === sym)?.prevClose ?? price
          const change    = price - prevClose
          const changePct = prevClose !== 0 ? (change / prevClose) * 100 : 0
          prevRef.current[sym] = price
          return {
            symbol:    sym,
            price,
            prevClose,
            change,
            changePct,
            source: q?.source ?? 'last_known',
          }
        })
        return updated
      })
    } catch {
      // If backend unreachable, keep existing tickers and show simulated flicker only if market open
    }
  }

  // ── Simulated micro-tick when market is open ──────────────────────────────
  function applyMicroTick() {
    setTickers(prev =>
      prev.map(t => {
        if (!marketStatus?.is_open) return t  // freeze when closed
        const noise     = (Math.random() - 0.5) * t.price * 0.0008
        const newPrice  = Math.max(0.01, t.price + noise)
        const change    = newPrice - t.prevClose
        const changePct = t.prevClose !== 0 ? (change / t.prevClose) * 100 : 0
        return { ...t, price: newPrice, change, changePct, source: 'live' }
      })
    )
  }

  // Initial fetch + periodic real-data refresh
  useEffect(() => {
    fetchPrices()
    // Refresh every 15s if market open, every 5min if closed
    const interval = setInterval(() => {
      fetchPrices()
    }, marketStatus?.is_open ? 15_000 : 300_000)
    return () => clearInterval(interval)
  }, [marketStatus?.is_open])

  // Micro-tick visual animation — only during market hours
  useEffect(() => {
    if (!marketStatus?.is_open) return
    const id = setInterval(applyMicroTick, 1_500)
    return () => clearInterval(id)
  }, [marketStatus?.is_open, tickers.length])

  const isOpen = marketStatus?.is_open ?? false

  return (
    <div className="ticker-bar" style={{ userSelect: 'none', position: 'relative' }}>

      {/* Market status badge */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        paddingRight: 20, borderRight: '1px solid var(--border-subtle)',
        marginRight: 4, flexShrink: 0,
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: isOpen ? 'var(--success)' : 'var(--danger)',
          boxShadow: isOpen ? '0 0 6px var(--success)' : 'none',
          animation: isOpen ? 'pulse-dot 1.5s infinite' : 'none',
        }} />
        <span style={{ fontSize: 11, fontWeight: 700, color: isOpen ? 'var(--success)' : 'var(--text-muted)', letterSpacing: '0.04em' }}>
          NSE {isOpen ? 'LIVE' : 'CLOSED'}
        </span>
        {!isOpen && marketStatus && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {marketStatus.current_time_ist} IST · Opens {marketStatus.next_open}
          </span>
        )}
        {isOpen && lastFetch && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
            {lastFetch.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        )}
      </div>

      {/* Tickers */}
      {tickers.length === 0 ? (
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Loading prices…</span>
      ) : (
        tickers.map(t => {
          const up = t.changePct >= 0
          return (
            <div key={t.symbol} className="ticker-item">
              <span className="ticker-symbol">{t.symbol.replace('NSE:', '')}</span>
              <span className="ticker-price" style={{
                transition: 'color 0.3s',
              }}>
                ₹{t.price < 100
                  ? t.price.toFixed(3)
                  : t.price < 10000
                    ? t.price.toFixed(2)
                    : t.price.toFixed(1)}
              </span>
              <span className={`ticker-change ${up ? 'up' : 'down'}`}>
                {up ? '▲' : '▼'} {Math.abs(t.changePct).toFixed(2)}%
              </span>
              {/* Source indicator */}
              {!isOpen && (
                <span style={{
                  fontSize: 9, color: 'var(--text-muted)',
                  background: 'var(--bg-elevated)',
                  borderRadius: 4, padding: '1px 5px',
                  border: '1px solid var(--border-subtle)',
                }}>
                  PREV CLOSE
                </span>
              )}
            </div>
          )
        })
      )}
    </div>
  )
}
