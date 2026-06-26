import { useState, useEffect, useCallback } from 'react'
import api from '../api'

const MODE_SYMBOLS: Record<string, string[]> = {
  equity: [
    'NSE:NIFTY 50', 'NSE:NIFTY BANK',
    'NSE:RELIANCE', 'NSE:TCS', 'NSE:INFY',
    'NSE:HDFCBANK', 'NSE:WIPRO', 'NSE:BAJFINANCE',
    'NSE:TMCV', 'NSE:TMPV', 'NSE:ICICIBANK', 'NSE:SBIN',
  ],
  forex: ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X'],
  crypto: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'PAXGUSDT']
}

interface Tick { symbol: string; ltp: number | null; prev: number | null }

function fmt(n: number) {
  return n < 100 ? n.toFixed(3) : n < 10000 ? n.toFixed(2) : n.toFixed(1)
}

export default function TickerBar({ mode }: { mode: string }) {
  const currentSymbols = MODE_SYMBOLS[mode] || MODE_SYMBOLS.equity
  const [ticks, setTicks]         = useState<Tick[]>(() => currentSymbols.map(s => ({ symbol: s, ltp: null, prev: null })))
  const [isOpen, setIsOpen]       = useState(false)
  const [timeStr, setTimeStr]     = useState('')

  // Market status
  useEffect(() => {
    async function check() {
      try {
        const res = await api.get(`/market_status?mode=${mode}`)
        setIsOpen(res.data.is_open)
        setTimeStr(res.data.current_time_ist)
      } catch {
        const now = new Date()
        const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }))
        const h = ist.getHours(), m = ist.getMinutes(), d = ist.getDay()
        const mins = h * 60 + m
        if (mode === 'crypto') {
          setIsOpen(true)
        } else if (mode === 'forex') {
          // Forex is Sun 22:00 GMT to Fri 22:00 GMT
          setIsOpen(d >= 1 && d <= 5)
        } else {
          setIsOpen(d >= 1 && d <= 5 && mins >= 555 && mins <= 930)
        }
        setTimeStr(`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`)
      }
    }
    check()
    const id = setInterval(check, 60_000)
    return () => clearInterval(id)
  }, [mode])

  // Reset ticks state when mode/symbols change
  useEffect(() => {
    setTicks(currentSymbols.map(s => ({ symbol: s, ltp: null, prev: null })))
  }, [mode])

  // Fetch LTPs
  const fetchLtps = useCallback(async () => {
    try {
      const res = await api.post('/ltp', { symbols: currentSymbols })
      const q   = res.data.quotes || {}
      setTicks(prev => prev.map(t => {
        const hit = q[t.symbol]
        return { ...t, prev: t.ltp, ltp: hit?.ltp ?? t.ltp }
      }))
    } catch {}
  }, [currentSymbols])

  useEffect(() => {
    fetchLtps()
    const id = setInterval(fetchLtps, isOpen ? 15_000 : 300_000)
    return () => clearInterval(id)
  }, [isOpen, fetchLtps])

  const getModeLabel = () => {
    if (mode === 'crypto') return 'CRYPTO'
    if (mode === 'forex') return 'FOREX'
    return 'NSE'
  }

  const getCurrencyPrefix = () => {
    return mode === 'equity' ? '₹' : '$'
  }

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 0,
      background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-subtle)',
      height: 36, overflow: 'hidden', padding: '0 14px', flexShrink: 0,
    }}>
      {/* Status badge */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        paddingRight: 14, borderRight: '1px solid var(--border-subtle)',
        marginRight: 10, flexShrink: 0,
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: isOpen ? 'var(--success)' : 'var(--danger)',
          boxShadow: isOpen ? '0 0 5px var(--success)' : 'none',
          animation: isOpen ? 'pulse-dot 1.5s infinite' : 'none',
          flexShrink: 0,
        }} />
        <span style={{ fontSize: 10, fontWeight: 700, color: isOpen ? 'var(--success)' : 'var(--text-muted)', whiteSpace: 'nowrap' }}>
          {getModeLabel()} {isOpen ? 'LIVE' : 'CLOSED'} · {timeStr} IST
        </span>
      </div>

      {/* Scrolling tickers */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', gap: 20, alignItems: 'center' }}>
        {ticks.map(({ symbol, ltp, prev }) => {
          const name = symbol.replace('NSE:NIFTY 50', 'NIFTY').replace('NSE:NIFTY BANK', 'BNKNIFTY').replace('NSE:', '').replace('=X', '')
          const up   = ltp !== null && prev !== null ? ltp >= prev : true
          const pct  = ltp && prev && prev !== 0 ? ((ltp - prev) / prev) * 100 : null
          return (
            <div key={symbol} style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0, fontSize: 11 }}>
              <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>{name}</span>
              <span style={{ fontWeight: 700, color: ltp ? (up ? '#26a69a' : '#ef5350') : 'var(--text-muted)' }}>
                {ltp !== null ? `${getCurrencyPrefix()}${fmt(ltp)}` : '—'}
              </span>
              {pct !== null && (
                <span style={{ color: up ? '#26a69a' : '#ef5350', fontSize: 10 }}>
                  {up ? '▲' : '▼'}{Math.abs(pct).toFixed(2)}%
                </span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
