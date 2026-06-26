import { useState, useEffect } from 'react'
import api from '../api'
import { TickerSearch } from '../components/TickerSearch'


const POPULAR_SYMBOLS_BY_MODE: Record<string, string[]> = {
  equity: [
    'NSE:INFY', 'NSE:RELIANCE', 'NSE:TCS', 'NSE:HDFCBANK',
    'NSE:WIPRO', 'NSE:BAJFINANCE', 'NSE:TATAMOTORS', 'NSE:ICICIBANK',
  ],
  forex: ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X'],
  crypto: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'PAXGUSDT']
}

// SESSION_KEY is derived dynamically per mode inside the component

export default function AnalysisPage({ mode }: { mode: string }) {
  const [symbol,       setSymbol]       = useState('NSE:INFY')
  const [loading,     setLoading]     = useState(false)
  const [result,       setResult]       = useState<{ output: string; chart_url: string } | null>(null)
  const [error,         setError]         = useState('')

  const sessionKey = `analysisPageState_${mode}`

  // Restore from sessionStorage on mount
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(sessionKey)
      if (saved) {
        const s = JSON.parse(saved)
        if (s.symbol) setSymbol(s.symbol)
        if (s.result) setResult(s.result)
      }
    } catch {}
  }, [sessionKey])

  // Auto-switch default symbol when mode changes
  useEffect(() => {
    const popular = POPULAR_SYMBOLS_BY_MODE[mode] || POPULAR_SYMBOLS_BY_MODE.equity
    const defaultSym = popular[0]
    
    const isCrypto = symbol.endsWith("USDT") || symbol.startsWith("P-") || symbol.startsWith("C-") || symbol.startsWith("F-")
    const isForex = symbol.endsWith("=X")
    
    if (mode === 'crypto' && !isCrypto) {
      setSymbol(defaultSym)
      setResult(null)
    } else if (mode === 'forex' && !isForex) {
      setSymbol(defaultSym)
      setResult(null)
    } else if (mode === 'equity' && (isCrypto || isForex)) {
      setSymbol(defaultSym)
      setResult(null)
    }
  }, [mode])

  // Persist to sessionStorage whenever result changes
  useEffect(() => {
    if (result) sessionStorage.setItem(sessionKey, JSON.stringify({ symbol, result }))
  }, [result, symbol, sessionKey])

  async function runAnalysis() {
    if (!symbol.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await api.post('/analysis', { symbol: symbol.trim().toUpperCase() })
      if (res.data.error) throw new Error(res.data.error)
      setResult(res.data)
    } catch (e: any) {
      setError(e.response?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  const popularSymbols = POPULAR_SYMBOLS_BY_MODE[mode] || POPULAR_SYMBOLS_BY_MODE.equity

  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <h1 className="page-title">📊 Technical Analysis</h1>
        <p className="page-subtitle">Real-time RSI, MACD, Bollinger Bands, EMA crossovers, and candlestick chart.</p>
      </div>

      {/* Input Row */}
      <div className="card" style={{ marginBottom: 18, overflow: 'visible' }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-row" style={{ flex: 1, minWidth: 200, marginBottom: 0 }}>
            <label className="form-label">Symbol</label>
            <TickerSearch 
              value={symbol} onChange={setSymbol}
              placeholder={mode === 'equity' ? 'NSE:INFY' : mode === 'forex' ? 'EURUSD=X' : 'BTCUSDT'}
              onSelect={() => runAnalysis()} 
              mode={mode}
            />
          </div>
          <button className="btn btn-primary" onClick={runAnalysis} disabled={loading}>
            {loading ? <><span className="spinner" /> Analysing…</> : '🔍 Analyse'}
          </button>
        </div>

        {/* Quick Symbols */}
        <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {popularSymbols.map(sym => (
            <button key={sym} className="btn btn-secondary btn-sm"
              onClick={() => { setSymbol(sym); }}>
              {sym.replace('NSE:', '').replace('=X', '')}
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
