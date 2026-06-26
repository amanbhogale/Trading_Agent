import { useState, useEffect } from 'react'
import api from '../api'
import { TickerSearch } from '../components/TickerSearch'


const STRATEGIES = [
  { value: 'sma_crossover',         label: 'SMA Crossover',          defaultParams: '{"fast": 20, "slow": 50}' },
  { value: 'rsi_mean_reversion',    label: 'RSI Mean Reversion',     defaultParams: '{"bb_window": 20}' },
  { value: 'macd_trend',            label: 'MACD Trend',             defaultParams: '{}' },
  { value: 'brownian_motion',       label: 'Brownian Motion (GBM)',  defaultParams: '{"window": 20}' },
  { value: 'market_making',         label: 'Market Making',          defaultParams: '{"window": 20, "spread_threshold": 0.001}' },
  { value: 'statistical_arbitrage', label: 'Statistical Arbitrage',  defaultParams: '{"window": 30, "z_score_threshold": 2.0}' },
  { value: 'momentum',              label: 'Momentum',               defaultParams: '{"lookback": 14}' },
  { value: 'mean_reversion',        label: 'Mean Reversion (BB)',    defaultParams: '{"bb_window": 20}' },
  { value: 'sentiment',             label: 'Sentiment Based',        defaultParams: '{"sentiment_threshold": 0.5}' },
]


// SESSION_KEY is derived dynamically per mode inside the component

export default function BacktestPage({ mode }: { mode: string }) {
  const [symbol,   setSymbol]   = useState('NSE:INFY')
  const [strategy, setStrategy] = useState('sma_crossover')
  const [params,   setParams]   = useState('{"fast": 20, "slow": 50}')
  const [days,     setDays]     = useState(365)
  const [loading,  setLoading]  = useState(false)
  const [chartUrl, setChartUrl] = useState('')
  const [output,   setOutput]   = useState('')
  const [error,    setError]    = useState('')

  const sessionKey = `backtestPageState_${mode}`

  // Restore from sessionStorage on mount
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(sessionKey)
      if (saved) {
        const s = JSON.parse(saved)
        if (s.symbol)   setSymbol(s.symbol)
        if (s.strategy) setStrategy(s.strategy)
        if (s.params)   setParams(s.params)
        if (s.days)     setDays(s.days)
        if (s.output)   setOutput(s.output)
        if (s.chartUrl) setChartUrl(s.chartUrl)
      }
    } catch {}
  }, [])

  // Auto-switch default symbol when mode changes
  useEffect(() => {
    const isCrypto = symbol.endsWith("USDT") || symbol.startsWith("P-") || symbol.startsWith("C-") || symbol.startsWith("F-")
    const isForex = symbol.endsWith("=X")
    
    if (mode === 'crypto' && !isCrypto) {
      setSymbol('BTCUSDT')
      setOutput('')
      setChartUrl('')
    } else if (mode === 'forex' && !isForex) {
      setSymbol('EURUSD=X')
      setOutput('')
      setChartUrl('')
    } else if (mode === 'equity' && (isCrypto || isForex)) {
      setSymbol('NSE:INFY')
      setOutput('')
      setChartUrl('')
    }
  }, [mode])

  // Persist to sessionStorage whenever results change
  useEffect(() => {
    if (output || chartUrl) {
      sessionStorage.setItem(sessionKey, JSON.stringify({ symbol, strategy, params, days, output, chartUrl }))
    }
  }, [output, chartUrl, symbol, strategy, params, days, sessionKey])

  function handleStrategyChange(val: string) {
    setStrategy(val)
    const s = STRATEGIES.find(s => s.value === val)
    if (s) setParams(s.defaultParams)
  }

  async function runBacktest() {
    setLoading(true)
    setError('')
    setOutput('')
    setChartUrl('')
    try {
      const res = await api.post('/backtest', {
        symbol: symbol.trim().toUpperCase(),
        strategy,
        params,
        days,
      })
      if (res.data.error) throw new Error(res.data.error)
      setOutput(res.data.output || '')
      setChartUrl(res.data.chart_url || '')
      // Try parse stats from output if possible
      // For now show what we have from backend
    } catch (e: any) {
      setError(e.response?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  const stratLabel = STRATEGIES.find(s => s.value === strategy)?.label ?? strategy

  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <h1 className="page-title">🧪 Strategy Backtest</h1>
        <p className="page-subtitle">
          Evaluate any of the {STRATEGIES.length} trading strategies against historical {mode === 'equity' ? 'NSE/BSE' : mode === 'forex' ? 'Forex' : 'Delta Exchange Crypto'} data with slippage & fee simulation.
        </p>
      </div>

      {/* Config Panel */}
      <div className="card" style={{ marginBottom: 20, overflow: 'visible' }}>
        <div className="grid" style={{ gridTemplateColumns: '1fr 2fr 1fr', gap: 14, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Symbol</label>
            <TickerSearch 
              value={symbol} onChange={setSymbol}
              placeholder={mode === 'equity' ? 'NSE:INFY' : mode === 'forex' ? 'EURUSD=X' : 'BTCUSDT'}
              style={{ width: '100%', boxSizing: 'border-box' }}
              mode={mode}
            />
          </div>

          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Strategy</label>
            <select className="form-select" value={strategy} onChange={e => handleStrategyChange(e.target.value)}>
              {STRATEGIES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>

          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Lookback Days</label>
            <input className="form-input" type="number" value={days} onChange={e => setDays(Number(e.target.value))} min={30} max={1825} />
          </div>
        </div>

        <div className="form-row" style={{ marginTop: 14 }}>
          <label className="form-label">Strategy Parameters (JSON)</label>
          <textarea className="form-textarea" value={params} onChange={e => setParams(e.target.value)} rows={2} />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-primary btn-lg" onClick={runBacktest} disabled={loading}>
            {loading ? <><span className="spinner" /> Running Backtest…</> : '▶ Run Backtest'}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error animate-fade-up" style={{ marginBottom: 18 }}>⚠️ {error}</div>}

      {loading && (
        <div className="loading-screen">
          <span className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <span>Running <strong>{stratLabel}</strong> on {symbol} for {days} days…</span>
          <span className="text-sm text-muted">This may take 10–30 seconds</span>
        </div>
      )}

      {(output || chartUrl) && !loading && (
        <div className="grid" style={{ gridTemplateColumns: '1fr', gap: 18 }}>
          {output && (
            <div className="card animate-fade-up">
              <div className="card-title" style={{ marginBottom: 12 }}>📋 Backtest Report</div>
              <div className="md-output" dangerouslySetInnerHTML={{ __html: output }} />
            </div>
          )}
          {chartUrl && (
            <div className="chart-frame-wrap animate-fade-up">
              <iframe
                src={`http://localhost:5000${chartUrl}`}
                style={{ height: 650 }}
                title="Backtest Equity Curve"
              />
            </div>
          )}
        </div>
      )}

      {/* Strategy Cards */}
      {!output && !loading && !error && (
        <>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Available Strategies
          </div>
          <div className="grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            {STRATEGIES.map(s => (
              <div key={s.value} className={`card ${strategy === s.value ? 'card-glow' : ''}`}
                style={{ cursor: 'pointer', padding: '14px 16px', borderColor: strategy === s.value ? 'var(--border-accent)' : undefined }}
                onClick={() => handleStrategyChange(s.value)}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: strategy === s.value ? 'var(--accent-blue-bright)' : 'var(--text-primary)' }}>
                  {s.label}
                </div>
                <div className="text-sm text-muted" style={{ fontFamily: 'monospace' }}>{s.defaultParams}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
