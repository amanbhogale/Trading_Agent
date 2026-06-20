import { useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:5000/api'

export default function PortfolioPage() {
  const [loading,  setLoading]  = useState(false)
  const [output,   setOutput]   = useState('')
  const [chartUrl, setChartUrl] = useState('')
  const [error,    setError]    = useState('')
  const [loaded,   setLoaded]   = useState(false)

  async function refresh() {
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${API_BASE}/portfolio`)
      if (res.data.error) throw new Error(res.data.error)
      setOutput(res.data.output || '')
      setChartUrl(res.data.chart_url || '')
      setLoaded(true)
    } catch (e: any) {
      setError(e.response?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">💼 Live Portfolio</h1>
            <p className="page-subtitle">Holdings, P&amp;L, and portfolio dashboard — powered by Zerodha Kite.</p>
          </div>
          <button className="btn btn-primary" onClick={refresh} disabled={loading}>
            {loading ? <><span className="spinner" /> Loading…</> : '🔄 Refresh Portfolio'}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error animate-fade-up">{error}</div>}

      {!loaded && !loading && !error && (
        <div className="card" style={{ textAlign: 'center', padding: '60px 28px' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>💼</div>
          <h2 style={{ fontWeight: 700, marginBottom: 8 }}>Your Portfolio</h2>
          <p className="text-muted" style={{ maxWidth: 420, margin: '0 auto 24px' }}>
            Click Refresh to pull live holdings, P&amp;L, and margin data from your Zerodha Kite account.
            Requires Kite authentication in the Configuration tab.
          </p>
          <button className="btn btn-primary btn-lg" onClick={refresh}>
            🔄 Load Portfolio
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-screen">
          <span className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <span>Fetching live portfolio from Kite…</span>
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

      {/* Feature Info Cards */}
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
