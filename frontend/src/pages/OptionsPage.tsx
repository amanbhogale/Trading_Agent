import { useState } from 'react'
import api from '../api'
import Plot from 'react-plotly.js'

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
  plot_paths?: number[][]
  error?: string
}

interface OptionsPageProps {
  mode: string
}

export default function OptionsPage({ mode }: OptionsPageProps) {
  const [symbol, setSymbol] = useState(mode === 'equity' ? 'NSE:NIFTY 50' : mode === 'forex' ? 'EURUSD=X' : 'BTCUSDT')
  const [assetClass, setAssetClass] = useState(mode)
  const [chainData, setChainData] = useState<ChainResponse | null>(null)
  const [loadingChain, setLoadingChain] = useState(false)

  const [portVal, setPortVal] = useState(100000)
  const [mu, setMu] = useState(0.10)
  const [sigma, setSigma] = useState(0.20)
  const [riskData, setRiskData] = useState<RiskResponse | null>(null)
  const [loadingRisk, setLoadingRisk] = useState(false)

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
        portfolio_value: portVal,
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
    <div className="page-container" style={{ padding: '20px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ margin: 0, fontSize: '24px' }}>📈 Options & Derivatives Desk</h1>
        <p style={{ color: 'var(--text-muted)' }}>
          Advanced Quantitative Models: Black-Scholes (Equity), Garman-Kohlhagen (Forex), Black-76 (Commodities), and Monte Carlo (VaR).
        </p>
      </div>

      <div className="panel" style={{ marginBottom: '24px', padding: '20px', background: 'var(--bg-panel)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
        <h2 style={{ fontSize: '18px', marginTop: 0, marginBottom: '16px' }}>🔗 Options Chain (Theoretical Pricing)</h2>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="e.g. NSE:NIFTY 50"
            className="input-base"
            style={{ width: '200px' }}
          />
          <select 
            value={assetClass} 
            onChange={(e) => setAssetClass(e.target.value)} 
            className="input-base"
          >
            <option value="equity">Equity (Black-Scholes)</option>
            <option value="forex">Forex (Garman-Kohlhagen)</option>
            <option value="commodity">Commodity (Black-76)</option>
          </select>
          <button onClick={fetchChain} className="btn-primary" disabled={loadingChain}>
            {loadingChain ? 'Running Models...' : 'Calculate Prices & Greeks'}
          </button>
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

      <div className="panel" style={{ padding: '20px', background: 'var(--bg-panel)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
        <h2 style={{ fontSize: '18px', marginTop: 0, marginBottom: '16px' }}>🎲 Portfolio Risk (Monte Carlo VaR)</h2>
        <p style={{ color: 'var(--text-muted)' }}>
          Simulate 5,000 paths over 1 year (252 trading days) using Geometric Brownian Motion to find the 99% Value at Risk.
        </p>
        <div style={{ display: 'flex', gap: '10px', marginBottom: '16px', alignItems: 'center' }}>
          <label style={{ fontSize: '12px', fontWeight: 600 }}>Portfolio Value:</label>
          <input type="number" value={portVal} onChange={(e) => setPortVal(Number(e.target.value))} className="input-base" style={{ width: '120px' }} />
          
          <label style={{ fontSize: '12px', fontWeight: 600 }}>Return (μ):</label>
          <input type="number" step="0.01" value={mu} onChange={(e) => setMu(Number(e.target.value))} className="input-base" style={{ width: '80px' }} />
          
          <label style={{ fontSize: '12px', fontWeight: 600 }}>Volatility (σ):</label>
          <input type="number" step="0.01" value={sigma} onChange={(e) => setSigma(Number(e.target.value))} className="input-base" style={{ width: '80px' }} />
          
          <button onClick={fetchRisk} className="btn-primary" disabled={loadingRisk}>
            {loadingRisk ? 'Simulating 5,000 paths...' : 'Run Simulation'}
          </button>
        </div>
        
        {riskData && (
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ marginTop: 0, marginBottom: '12px', fontSize: '14px' }}>Risk Metrics (99% Confidence)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
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
                <div style={{ fontSize: '20px', color: 'var(--text-positive)', fontWeight: 'bold' }}>₹{riskData.simulated_worst_case.toLocaleString()}</div>
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
                    line: { width: 1, color: 'rgba(59, 130, 246, 0.3)' },
                    hoverinfo: 'skip'
                  }))}
                  layout={{
                    autosize: true,
                    height: 300,
                    margin: { l: 40, r: 20, t: 10, b: 30 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    showlegend: false,
                    xaxis: { gridcolor: '#333' },
                    yaxis: { gridcolor: '#333' }
                  }}
                  useResizeHandler={true}
                  style={{ width: '100%', height: '100%' }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
