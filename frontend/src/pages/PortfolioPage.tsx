import { useState, useEffect } from 'react'
import api from '../api'

// SESSION_KEY is derived dynamically per mode inside the component

interface Stats {
  total_invested: number
  total_current:  number
  total_pnl:      number
  pnl_pct:        number
  num_holdings:   number
}

function fmt(n: number) {
  return new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(n)
}

export default function PortfolioPage({ mode }: { mode: string }) {
  const [loading,  setLoading]  = useState(false)
  const [output,   setOutput]   = useState('')
  const [chartUrl, setChartUrl] = useState('')
  const [error,    setError]    = useState('')
  const [loaded,   setLoaded]   = useState(false)
  const [stats,    setStats]    = useState<Stats | null>(null)

  const sessionKey = `portfolioPageState_${mode}`

  // Restore state from sessionStorage on mount
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(sessionKey)
      if (saved) {
        const s = JSON.parse(saved)
        setOutput(s.output   ?? '')
        setChartUrl(s.chartUrl ?? '')
        setStats(s.stats     ?? null)
        setLoaded(s.loaded   ?? false)
      }
    } catch {}
  }, [])

  // Auto-refresh when mode changes
  useEffect(() => {
    setLoaded(false)
    setStats(null)
    setOutput('')
    setChartUrl('')
    refresh()
  }, [mode])

  // Persist state to sessionStorage whenever it changes
  useEffect(() => {
    if (loaded) {
      sessionStorage.setItem(sessionKey, JSON.stringify({ output, chartUrl, stats, loaded }))
    }
  }, [output, chartUrl, stats, loaded, sessionKey])

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      // Fetch dashboard + stats concurrently
      const [dashRes, statsRes] = await Promise.all([
        api.post('/portfolio', { mode }),
        api.get(`/portfolio_stats?mode=${mode}`),
      ])
      if (dashRes.data.error) throw new Error(dashRes.data.error)
      setOutput(dashRes.data.output   || '')
      setChartUrl(dashRes.data.chart_url || '')
      setStats(statsRes.data.error ? null : statsRes.data)
      setLoaded(true)
    } catch (e: any) {
      setError(e.response?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  const pnlPositive = stats ? stats.total_pnl >= 0 : true
  const currencySymbol = mode === 'equity' ? '₹' : '$'

  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">💼 Live Portfolio</h1>
            <p className="page-subtitle">
              {mode === 'equity' 
                ? 'Holdings, P&L, and portfolio dashboard — powered by Zerodha Kite.' 
                : `Holdings, P&L, and portfolio dashboard — Mock paper-trading mode.`}
            </p>
          </div>
          <button className="btn btn-primary" onClick={refresh} disabled={loading}>
            {loading ? <><span className="spinner" /> Loading…</> : '🔄 Refresh Portfolio'}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error animate-fade-up">{error}</div>}

      {/* Aggregate stats bar */}
      {stats && !loading && (
        <div className="grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 18 }}>
          {[
            {
              label: '💰 Total Invested',
              value: `${currencySymbol}${fmt(stats.total_invested)}`,
              sub: `${stats.num_holdings} holdings`,
              color: 'var(--text-primary)',
            },
            {
              label: '📈 Current Value',
              value: `${currencySymbol}${fmt(stats.total_current)}`,
              sub: 'at market price',
              color: 'var(--accent-blue-bright)',
            },
            {
              label: pnlPositive ? '🟢 Total P&L' : '🔴 Total P&L',
              value: `${pnlPositive ? '+' : ''}${currencySymbol}${fmt(stats.total_pnl)}`,
              sub: `${pnlPositive ? '+' : ''}${stats.pnl_pct.toFixed(2)}% overall`,
              color: pnlPositive ? '#10b981' : '#ef4444',
            },
            {
              label: pnlPositive ? '📊 Return' : '📊 Return',
              value: `${pnlPositive ? '+' : ''}${stats.pnl_pct.toFixed(2)}%`,
              sub: 'unrealised return',
              color: pnlPositive ? '#10b981' : '#ef4444',
            },
          ].map(({ label, value, sub, color }) => (
            <div key={label} className="card" style={{ padding: '16px 20px' }}>
              <div className="text-sm text-muted" style={{ marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color }}>{value}</div>
              <div className="text-sm text-muted" style={{ marginTop: 4 }}>{sub}</div>
            </div>
          ))}
        </div>
      )}

      {!loaded && !loading && !error && (
        <div className="card" style={{ textAlign: 'center', padding: '60px 28px' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>💼</div>
          <h2 style={{ fontWeight: 700, marginBottom: 8 }}>Your Portfolio</h2>
          <p className="text-muted" style={{ maxWidth: 420, margin: '0 auto 24px' }}>
            {mode === 'equity' 
              ? 'Click Refresh to pull live holdings, P&L, and margin data from your Zerodha Kite account. Requires Kite authentication in the Configuration tab.'
              : 'Click Refresh to load mock portfolio holdings and paper trades from the database.'}
          </p>
          <button className="btn btn-primary btn-lg" onClick={refresh}>
            🔄 Load Portfolio
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-screen">
          <span className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <span>{mode === 'equity' ? 'Fetching live portfolio from Kite…' : 'Fetching mock portfolio from DB…'}</span>
        </div>
      )}

      {loaded && !loading && (
        <div className="grid" style={{ gridTemplateColumns: '1fr', gap: 18 }}>
          {output && (
            <div className="card animate-fade-up">
              <div className="card-title" style={{ marginBottom: 14 }}>📋 Portfolio Summary</div>
              <div className="md-output" dangerouslySetInnerHTML={{ __html: output }} />
            </div>
          )}
          {chartUrl && (
            <div className="chart-frame-wrap animate-fade-up">
              <iframe
                src={`http://localhost:5000${chartUrl}`}
                style={{ height: 850 }}
                title="Portfolio Dashboard"
              />
            </div>
          )}
        </div>
      )}

      {/* Feature Info Cards when idle */}
      {!loaded && !error && !loading && (
        <div className="grid grid-3 mt-6">
          {[
            { icon: '📊', title: 'Holdings Breakdown', desc: 'Real-time prices for every position, quantity, and cost basis.' },
            { icon: '💰', title: 'P&L Tracking', desc: 'Day P&L and overall P&L per holding with visual indicators.' },
            { icon: '🗂️', title: 'Allocation Dashboard', desc: 'Interactive Plotly charts showing portfolio allocation, invested vs. current value.' },
          ].map(({ icon, title, desc }) => (
            <div key={title} className="card" style={{ textAlign: 'center', padding: '24px 18px' }}>
              <div style={{ fontSize: 28, marginBottom: 10 }}>{icon}</div>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>{title}</div>
              <div className="text-sm text-muted">{desc}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
