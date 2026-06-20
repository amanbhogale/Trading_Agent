import { useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:5000/api'

const POPULAR_SYMBOLS = [
  'NSE:INFY', 'NSE:RELIANCE', 'NSE:TCS', 'NSE:HDFC',
  'NSE:WIPRO', 'NSE:BAJFINANCE', 'NSE:TATAMOTORS', 'NSE:ICICIBANK',
]

export default function AnalysisPage() {
  const [symbol, setSymbol]       = useState('NSE:INFY')
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState<{ output: string; chart_url: string } | null>(null)
  const [error, setError]         = useState('')

  async function runAnalysis() {
    if (!symbol.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await axios.post(`${API_BASE}/analysis`, { symbol: symbol.trim().toUpperCase() })
      if (res.data.error) throw new Error(res.data.error)
      setResult(res.data)
    } catch (e: any) {
      setError(e.response?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <h1 className="page-title">📊 Technical Analysis</h1>
        <p className="page-subtitle">Real-time RSI, MACD, Bollinger Bands, EMA crossovers, and candlestick chart.</p>
      </div>

      {/* Input Row */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-row" style={{ flex: 1, minWidth: 200, marginBottom: 0 }}>
            <label className="form-label">Symbol</label>
            <input className="form-input" value={symbol} onChange={e => setSymbol(e.target.value)}
              placeholder="e.g. NSE:INFY"
              onKeyDown={e => e.key === 'Enter' && runAnalysis()} />
          </div>
          <button className="btn btn-primary" onClick={runAnalysis} disabled={loading}>
            {loading ? <><span className="spinner" /> Analysing…</> : '🔍 Analyse'}
          </button>
        </div>

        {/* Quick Symbols */}
        <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {POPULAR_SYMBOLS.map(sym => (
            <button key={sym} className="btn btn-secondary btn-sm"
              onClick={() => { setSymbol(sym); }}>
              {sym.replace('NSE:', '')}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="alert alert-error animate-fade-up" style={{ marginBottom: 18 }}>⚠️ {error}</div>
      )}

      {loading && (
        <div className="loading-screen">
          <span className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
          <span>Fetching market data & computing indicators…</span>
        </div>
      )}

      {result && !loading && (
        <div className="grid" style={{ gridTemplateColumns: '1fr', gap: 18 }}>
          {/* AI Analysis Output */}
          <div className="card animate-fade-up">
            <div className="card-title" style={{ marginBottom: 14 }}>📋 Analysis Report — {symbol.toUpperCase()}</div>
            <div className="md-output" dangerouslySetInnerHTML={{ __html: result.output }} />
          </div>

          {/* Chart */}
          <div className="chart-frame-wrap animate-fade-up">
            <iframe
              src={`http://localhost:5000${result.chart_url}`}
              style={{ height: 700 }}
              title="Chart"
            />
          </div>
        </div>
      )}

      {/* Info Cards when idle */}
      {!result && !loading && !error && (
        <div className="grid grid-3" style={{ marginTop: 8 }}>
          {[
            { icon: '📈', title: 'RSI', desc: 'Relative Strength Index to detect overbought/oversold conditions.' },
            { icon: '〽️', title: 'MACD', desc: 'Moving Average Convergence Divergence for trend momentum.' },
            { icon: '📉', title: 'Bollinger Bands', desc: 'Volatility bands for breakout and squeeze detection.' },
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
