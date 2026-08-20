import { useState, useEffect } from 'react'
import api from '../api'
import Plot from 'react-plotly.js'
import { TickerSearch } from '../components/TickerSearch'

interface ChainRow {
  strike: number
  call_price: number
  put_price: number
  delta: number
  gamma: number
  theta: number
  vega: number
}

interface ChainResponse {
  symbol: string
  spot: number
  chain: ChainRow[]
  error?: string
}

interface RiskResponse {
  VaR_99: number
  CVaR_99: number
  simulated_worst_case: number
  current_portfolio_value?: number
  plot_paths?: number[][]
  error?: string
}

interface HedgingPosition {
  id: number
  symbol: string
  option_type: string
  strike: number
  expiry: string | null
  quantity: number
}

interface HedgeStatus {
  is_active: boolean
  net_delta: number
  recent_trades: any[]
}

interface OptionsPageProps {
  mode: string
}

export default function OptionsPage({ mode }: OptionsPageProps) {
  const [symbol, setSymbol] = useState(mode === 'equity' ? 'NSE:NIFTY 50' : mode === 'forex' ? 'EURUSD=X' : 'BTCUSDT')
  const [assetClass, setAssetClass] = useState('equity')
  const [chainData, setChainData] = useState<ChainResponse | null>(null)
  const [loadingChain, setLoadingChain] = useState(false)

  const [mu, setMu] = useState(0.1)
  const [sigma, setSigma] = useState(0.2)
  const [riskData, setRiskData] = useState<RiskResponse | null>(null)
  const [loadingRisk, setLoadingRisk] = useState(false)

  const [positions, setPositions] = useState<HedgingPosition[]>([])
  const [hedgeStatus, setHedgeStatus] = useState<HedgeStatus | null>(null)
  
  const [mockType, setMockType] = useState('call')
  const [mockStrike, setMockStrike] = useState(25000)
  const [mockQty, setMockQty] = useState(100)

  useEffect(() => {
    fetchPortfolio()
    const intv = setInterval(fetchStatus, 3000)
    return () => clearInterval(intv)
  }, [])

  const fetchPortfolio = async () => {
    try {
      const res = await api.get<HedgingPosition[]>('/options/portfolio')
      setPositions(res.data)
    } catch(e) {}
  }

  const fetchStatus = async () => {
    try {
      const res = await api.get<HedgeStatus>('/options/hedge_status')
      setHedgeStatus(res.data)
    } catch(e) {}
  }

  const addMockPosition = async () => {
    await api.post('/options/portfolio', { symbol, option_type: mockType, strike: mockStrike, quantity: mockQty })
    fetchPortfolio()
  }

  const removePosition = async (id: number) => {
    await api.delete('/options/portfolio', { data: { id } })
    fetchPortfolio()
  }

  const toggleEngine = async () => {
    if (!hedgeStatus) return
    await api.post('/options/hedge_status', { active: !hedgeStatus.is_active })
    fetchStatus()
  }

  const fetchChain = async () => {
    setLoadingChain(true)
    setChainData(null)
    try {
      const res = await api.post<ChainResponse>('/options/chain', {
        symbol,
        asset_class: assetClass
      })
      if (res.data.error) alert(res.data.error)
      else setChainData(res.data)
    } catch (err: any) {
      alert(err.friendlyMessage || "Failed to fetch chain data.")
    } finally {
      setLoadingChain(false)
    }
  }

  const fetchRisk = async () => {
    setLoadingRisk(true)
    setRiskData(null)
    try {
      const res = await api.post<RiskResponse>('/options/risk', {
        mu,
        sigma
      })
      if (res.data.error) alert(res.data.error)
      else setRiskData(res.data)
    } catch (err: any) {
      alert(err.friendlyMessage || "Failed to fetch risk data.")
    } finally {
      setLoadingRisk(false)
    }
  }

  return (
    <div className="page animate-fade-up" style={{ padding: '20px' }}>
      <div className="page-header">
        <h1 className="page-title">📈 Options & Derivatives Desk</h1>
        <p className="page-subtitle">
          Advanced Quantitative Models: Black-Scholes (Equity), Garman-Kohlhagen (Forex), Black-76 (Commodities), and Monte Carlo (VaR).
        </p>
      </div>

      <div className="card" style={{ marginBottom: 24, overflow: 'visible' }}>
        <h2 style={{ fontSize: '18px', marginTop: 0, marginBottom: '16px' }}>🔗 Options Chain (Theoretical Pricing)</h2>
        <div className="grid" style={{ gridTemplateColumns: '2fr 1fr auto', gap: 14, alignItems: 'flex-end', marginBottom: '16px' }}>
          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Underlying Asset</label>
            <TickerSearch
              value={symbol}
              onChange={setSymbol}
              mode={mode}
              placeholder="Search Underlying..."
              style={{ width: '100%' }}
            />
          </div>
          
          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Pricing Model</label>
            <select 
              value={assetClass} 
              onChange={(e) => setAssetClass(e.target.value)} 
              className="form-select"
            >
            <option value="equity">Equity (Black-Scholes)</option>
            <option value="forex">Forex (Garman-Kohlhagen)</option>
            <option value="commodity">Commodity (Black-76)</option>
          </select>
          </div>

          <div className="form-row" style={{ marginBottom: 0 }}>
            <button onClick={fetchChain} className="btn btn-primary" disabled={loadingChain} style={{ height: '36px' }}>
              {loadingChain ? <><span className="spinner" /> Running...</> : 'Calculate Greeks'}
            </button>
          </div>
        </div>

        {chainData && (
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table" style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '8px' }}>Strike</th>
                  <th style={{ padding: '8px' }}>Call Price</th>
                  <th style={{ padding: '8px' }}>Put Price</th>
                  <th style={{ padding: '8px' }}>Delta (Δ)</th>
                  <th style={{ padding: '8px' }}>Gamma (Γ)</th>
                  <th style={{ padding: '8px' }}>Theta (Θ)</th>
                  <th style={{ padding: '8px' }}>Vega (V)</th>
                </tr>
              </thead>
              <tbody>
                {chainData.chain.map((row, idx) => {
                  const isAtm = Math.abs(row.strike - chainData.spot) < (chainData.spot * 0.02)
                  return (
                    <tr key={idx} style={{ 
                      borderBottom: '1px solid var(--border-subtle)',
                      backgroundColor: isAtm ? 'rgba(59, 130, 246, 0.1)' : 'transparent' 
                    }}>
                      <td style={{ padding: '8px', fontWeight: 'bold' }}>{row.strike}</td>
                      <td style={{ padding: '8px', color: 'var(--text-positive)' }}>{row.call_price}</td>
                      <td style={{ padding: '8px', color: 'var(--text-negative)' }}>{row.put_price}</td>
                      <td style={{ padding: '8px' }}>{row.delta}</td>
                      <td style={{ padding: '8px' }}>{row.gamma}</td>
                      <td style={{ padding: '8px' }}>{row.theta}</td>
                      <td style={{ padding: '8px' }}>{row.vega}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card" style={{ padding: '20px' }}>
        <h2 style={{ fontSize: '18px', marginTop: 0, marginBottom: '8px' }}>🎲 True Portfolio Risk (Non-Linear VaR)</h2>
        <p className="page-subtitle" style={{ marginBottom: '16px' }}>
          Simulates 5,000 underlying price paths 30 days forward, re-pricing every option in your Mock Portfolio to find the 99% Value at Risk.
        </p>

        <div className="grid" style={{ gridTemplateColumns: '1fr 1fr auto', gap: 14, alignItems: 'flex-end', marginBottom: '16px' }}>
          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Expected Underlying Return (μ)</label>
            <input type="number" step="0.01" value={mu} onChange={(e) => setMu(Number(e.target.value))} className="form-input" />
          </div>
          
          <div className="form-row" style={{ marginBottom: 0 }}>
            <label className="form-label">Underlying Volatility (σ)</label>
            <input type="number" step="0.01" value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="form-input" />
          </div>
          
          <div className="form-row" style={{ marginBottom: 0 }}>
            <button onClick={fetchRisk} className="btn btn-primary" disabled={loadingRisk} style={{ height: '36px' }}>
              {loadingRisk ? <><span className="spinner" /> Simulating...</> : 'Run Portfolio Simulation'}
            </button>
          </div>
        </div>
        
        {riskData && (
          <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '14px', marginTop: 0, marginBottom: '12px' }}>Portfolio Risk Metrics (99% Confidence - 30 Days)</h3>
            <div className="grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', gap: 20 }}>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Current Value</div>
                <div style={{ fontSize: '20px', color: 'var(--text-primary)', fontWeight: 'bold' }}>₹{riskData.current_portfolio_value?.toLocaleString() || '0'}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Value at Risk (VaR)</div>
                <div style={{ fontSize: '20px', color: 'var(--text-negative)', fontWeight: 'bold' }}>₹{riskData.VaR_99.toLocaleString()}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Expected Shortfall (CVaR)</div>
                <div style={{ fontSize: '20px', color: 'var(--text-negative)', fontWeight: 'bold' }}>₹{riskData.CVaR_99.toLocaleString()}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Worst-case Value</div>
                <div style={{ fontSize: '20px', color: 'var(--warning)', fontWeight: 'bold' }}>₹{riskData.simulated_worst_case.toLocaleString()}</div>
              </div>
            </div>

            {riskData.plot_paths && riskData.plot_paths.length > 0 && (
              <div style={{ marginTop: '20px', borderTop: '1px solid var(--border-subtle)', paddingTop: '20px' }}>
                <h4 style={{ fontSize: '13px', margin: '0 0 12px 0', color: 'var(--text-muted)' }}>Monte Carlo Spaghetti Chart (Sample of 50 Paths)</h4>
                <Plot
                  data={riskData.plot_paths.map((path, idx) => ({
                    y: path,
                    type: 'scatter',
                    mode: 'lines',
                    line: { 
                      width: 1.5, 
                      color: `hsla(${(idx * 15) % 360}, 100%, 65%, 0.7)` 
                    },
                    hoverinfo: 'skip'
                  }))}
                  layout={{
                    autosize: true,
                    height: 300,
                    margin: { l: 40, r: 20, t: 10, b: 30 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    showlegend: false,
                    xaxis: { 
                      gridcolor: '#333',
                      tickfont: { color: '#bcc6e5', size: 11 }
                    },
                    yaxis: { 
                      gridcolor: '#333',
                      tickfont: { color: '#bcc6e5', size: 11 }
                    }
                  }}
                  useResizeHandler={true}
                  style={{ width: '100%', height: '100%' }}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Dynamic Hedging Dashboard */}
      <div className="card" style={{ padding: '20px', marginTop: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h2 style={{ fontSize: '18px', marginTop: 0, marginBottom: '8px' }}>🛡️ Dynamic Delta Hedging (Paper Trading)</h2>
            <p className="page-subtitle" style={{ margin: 0 }}>
              Automated algorithmic portfolio hedging. Engine automatically simulates buying/selling Futures to keep Net Delta near zero.
            </p>
          </div>
          <div>
            <button 
              onClick={toggleEngine} 
              className="btn" 
              style={{ 
                background: hedgeStatus?.is_active ? 'var(--danger)' : 'var(--success)', 
                color: '#fff', border: 'none', fontWeight: 'bold' 
              }}>
              {hedgeStatus?.is_active ? 'STOP ENGINE' : 'START ENGINE'}
            </button>
          </div>
        </div>

        <div className="grid grid-2" style={{ gap: 20 }}>
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '14px', marginTop: 0, marginBottom: '12px' }}>Current Portfolio Net Delta</h3>
            <div style={{ fontSize: '32px', fontWeight: 'bold', color: hedgeStatus && Math.abs(hedgeStatus.net_delta) > 0.5 ? 'var(--warning)' : 'var(--success)' }}>
              {hedgeStatus?.net_delta.toFixed(3) || '0.000'} Δ
            </div>
            
            <h3 style={{ fontSize: '14px', marginTop: '20px', marginBottom: '12px' }}>Recent Hedge Executions</h3>
            {hedgeStatus?.recent_trades && hedgeStatus.recent_trades.length > 0 ? (
              <table className="data-table" style={{ width: '100%', fontSize: '12px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <th>Time</th><th>Action</th><th>Symbol</th>
                  </tr>
                </thead>
                <tbody>
                  {hedgeStatus.recent_trades.map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '6px' }}>{new Date(t.executed_at).toLocaleTimeString()}</td>
                      <td style={{ padding: '6px', fontWeight: 'bold', color: t.trade_type === 'BUY' ? 'var(--success)' : 'var(--danger)' }}>{t.trade_type} {t.quantity}</td>
                      <td style={{ padding: '6px' }}>{t.symbol}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No recent trades executed.</p>}
          </div>

          <div>
            <h3 style={{ fontSize: '14px', marginTop: 0, marginBottom: '12px' }}>Mock Portfolio Positions</h3>
            <div className="grid" style={{ gridTemplateColumns: '1fr 1fr 1fr auto', gap: 10, alignItems: 'flex-end', marginBottom: '16px' }}>
              <div className="form-row" style={{ marginBottom: 0 }}>
                <label className="form-label">Type</label>
                <select className="form-select" value={mockType} onChange={e => setMockType(e.target.value)}>
                  <option value="call">Call</option>
                  <option value="put">Put</option>
                </select>
              </div>
              <div className="form-row" style={{ marginBottom: 0 }}>
                <label className="form-label">Strike</label>
                <input type="number" className="form-input" value={mockStrike} onChange={e => setMockStrike(Number(e.target.value))} />
              </div>
              <div className="form-row" style={{ marginBottom: 0 }}>
                <label className="form-label">Qty</label>
                <input type="number" className="form-input" value={mockQty} onChange={e => setMockQty(Number(e.target.value))} />
              </div>
              <div className="form-row" style={{ marginBottom: 0 }}>
                <button className="btn btn-primary" onClick={addMockPosition} style={{ height: '36px' }}>+ Add</button>
              </div>
            </div>

            <table className="data-table" style={{ width: '100%', fontSize: '12px', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '6px' }}>Symbol</th>
                  <th style={{ padding: '6px' }}>Type</th>
                  <th style={{ padding: '6px' }}>Strike</th>
                  <th style={{ padding: '6px' }}>Qty</th>
                  <th style={{ padding: '6px' }}></th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => (
                  <tr key={p.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '6px' }}>{p.symbol}</td>
                    <td style={{ padding: '6px' }}>{p.option_type.toUpperCase()}</td>
                    <td style={{ padding: '6px' }}>{p.strike || '-'}</td>
                    <td style={{ padding: '6px', fontWeight: 'bold', color: p.quantity > 0 ? 'var(--success)' : 'var(--danger)' }}>{p.quantity}</td>
                    <td style={{ padding: '6px', textAlign: 'right' }}>
                      <button onClick={() => removePosition(p.id)} style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer', fontSize: '14px' }}>✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
