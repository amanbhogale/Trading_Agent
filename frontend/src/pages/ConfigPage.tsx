import { useState, useEffect } from 'react'
import api from '../api'


const PROVIDERS = [
  { value: 'openrouter',    label: 'OpenRouter',   baseUrl: 'https://openrouter.ai/api/v1' },
  { value: 'openai',        label: 'OpenAI',        baseUrl: '' },
  { value: 'anthropic',     label: 'Anthropic',     baseUrl: '' },
  { value: 'google-gemini', label: 'Google Gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai' },
]

// Verified valid model IDs per provider
const PROVIDER_MODELS: Record<string, { id: string; label: string; free?: boolean }[]> = {
  openrouter: [
    { id: 'nvidia/nemotron-3-ultra-550b-a55b:free',   label: 'Nvidia Nemotron 550B',      free: true },
    { id: 'deepseek/deepseek-r1:free',                label: 'DeepSeek R1',              free: true },
    { id: 'deepseek/deepseek-chat-v3-0324:free',      label: 'DeepSeek Chat v3',         free: true },
    { id: 'meta-llama/llama-3.3-70b-instruct:free',   label: 'Llama 3.3 70B',            free: true },
    { id: 'google/gemma-3-27b-it:free',               label: 'Gemma 3 27B',              free: true },
    { id: 'mistralai/mistral-7b-instruct:free',       label: 'Mistral 7B',               free: true },
    { id: 'qwen/qwen3-235b-a22b:free',                label: 'Qwen3 235B',               free: true },
    { id: 'openai/gpt-4o',                            label: 'GPT-4o (paid)',            free: false },
    { id: 'anthropic/claude-sonnet-4-5',              label: 'Claude Sonnet 4.5 (paid)', free: false },
    { id: 'google/gemini-2.5-pro',                    label: 'Gemini 2.5 Pro (paid)',    free: false },
  ],
  openai: [
    { id: 'gpt-4o',          label: 'GPT-4o' },
    { id: 'gpt-4o-mini',     label: 'GPT-4o Mini' },
    { id: 'gpt-4-turbo',     label: 'GPT-4 Turbo' },
    { id: 'o1-mini',         label: 'o1 Mini' },
  ],
  anthropic: [
    { id: 'claude-sonnet-4-5',      label: 'Claude Sonnet 4.5' },
    { id: 'claude-3-5-haiku-latest',label: 'Claude 3.5 Haiku' },
    { id: 'claude-opus-4-5',        label: 'Claude Opus 4.5' },
  ],
  'google-gemini': [
    { id: 'gemini-2.5-pro',   label: 'Gemini 2.5 Pro' },
    { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
    { id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  ],
}

interface Props {
  setConnected:     (v: boolean) => void
  setKiteConnected: (v: boolean) => void
}

/* ── LLM Connection Panel ───────────────────── */
function LLMPanel({ setConnected }: { setConnected: (v: boolean) => void }) {
  const [provider,    setProvider]    = useState('openrouter')
  const [model,       setModel]       = useState('nvidia/nemotron-3-ultra-550b-a55b:free')
  const [apiKey,      setApiKey]      = useState('')
  const [baseUrl,     setBaseUrl]     = useState('https://openrouter.ai/api/v1')
  const [loading,     setLoading]     = useState(false)
  const [status,      setStatus]      = useState('')
  const [ok,          setOk]          = useState(false)

  function handleProvider(val: string) {
    setProvider(val)
    const p = PROVIDERS.find(p => p.value === val)
    if (p) setBaseUrl(p.baseUrl)
    // Auto-select first model for new provider
    const models = PROVIDER_MODELS[val]
    if (models?.length) setModel(models[0].id)
  }

  async function connect() {
    setLoading(true)
    setStatus('')
    try {
      const res = await api.post('/connect_llm', {
        provider, model, api_key: apiKey, base_url: baseUrl,
      })
      const success = res.data.connected === true
      setOk(success)
      setStatus(res.data.message || res.data.error)
      setConnected(success)
    } catch (e: any) {
      const msg = e.response?.data?.error || e.message
      setStatus('❌ ' + msg)
      setOk(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card card-glow" style={{ position: 'relative' }}>
      {/* Status badge top-right */}
      <div style={{ position: 'absolute', top: 18, right: 18 }}>
        <span className={`status-pill ${ok ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          {ok ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 20, marginBottom: 4 }}>🤖</div>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>LLM / Orchestrator</h2>
        <p className="text-sm text-muted">Powers the AI agent, analysis, strategy decisions, and chat.</p>
      </div>

      <div className="form-row">
        <label className="form-label">Provider</label>
        <select className="form-select" value={provider} onChange={e => handleProvider(e.target.value)}>
          {PROVIDERS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
        </select>
      </div>

      <div className="form-row">
        <label className="form-label">Model</label>
        <input className="form-input" value={model} onChange={e => setModel(e.target.value)}
          placeholder="e.g. deepseek/deepseek-r1:free" />
        {/* Quick-select chips */}
        {PROVIDER_MODELS[provider] && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {PROVIDER_MODELS[provider].map(m => (
              <button
                key={m.id}
                type="button"
                onClick={() => setModel(m.id)}
                className="btn btn-sm"
                style={{
                  background: model === m.id ? 'var(--accent-blue)' : 'var(--bg-elevated)',
                  border: `1px solid ${model === m.id ? 'var(--accent-blue)' : 'var(--border-mid)'}`,
                  color: model === m.id ? 'white' : 'var(--text-secondary)',
                  borderRadius: 99,
                  fontSize: 11,
                  padding: '3px 10px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                {m.free && <span style={{ color: '#10b981', fontWeight: 800 }}>FREE</span>}
                {m.label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="form-row">
        <label className="form-label">API Key</label>
        <input className="form-input" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
          placeholder="sk-..." />
      </div>

      <div className="form-row">
        <label className="form-label">Base URL</label>
        <input className="form-input" value={baseUrl} onChange={e => setBaseUrl(e.target.value)}
          placeholder="https://openrouter.ai/api/v1" />
      </div>

      <button className="btn btn-primary w-full" style={{ marginTop: 8 }} onClick={connect} disabled={loading}>
        {loading ? <><span className="spinner" /> Connecting…</> : '🔌 Connect LLM'}
      </button>

      {status && (
        <div className={`alert ${ok ? 'alert-success' : 'alert-error'} animate-fade-up`}
          style={{ marginTop: 12, fontSize: 12 }}>
          {status}
        </div>
      )}
    </div>
  )
}

/* ── Kite Connection Panel ──────────────────── */
function KitePanel({ setKiteConnected }: { setKiteConnected: (v: boolean) => void }) {
  const [kiteKey,   setKiteKey]   = useState('')
  const [kiteToken, setKiteToken] = useState('')
  const [loading,   setLoading]   = useState(false)
  const [testing,   setTesting]   = useState(false)
  const [status,    setStatus]    = useState('')
  const [ok,        setOk]        = useState(false)
  const [userName,  setUserName]  = useState('')

  async function connect() {
    setLoading(true)
    setStatus('')
    try {
      const res = await api.post('/connect_kite', {
        kite_key: kiteKey, kite_token: kiteToken,
      })
      const success = res.data.connected === true
      setOk(success)
      setStatus(res.data.message || res.data.error)
      setUserName(res.data.user || '')
      setKiteConnected(success)
    } catch (e: any) {
      const msg = e.response?.data?.error || e.message
      setStatus('❌ ' + msg)
      setOk(false)
    } finally {
      setLoading(false)
    }
  }

  async function testConnection() {
    setTesting(true)
    setStatus('')
    try {
      const res = await api.post('/test_kite', {
        kite_key: kiteKey, kite_token: kiteToken,
      })
      const msg = res.data.message || ''
      setStatus(msg.replace(/\*\*/g, ''))
      setOk(msg.includes('True'))
      setKiteConnected(msg.includes('True'))
    } catch (e: any) {
      setStatus('❌ ' + e.message)
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="card" style={{ position: 'relative' }}>
      {/* Status badge top-right */}
      <div style={{ position: 'absolute', top: 18, right: 18 }}>
        <span className={`status-pill ${ok ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          {ok ? (userName || 'Connected') : 'Disconnected'}
        </span>
      </div>

      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 20, marginBottom: 4 }}>🪁</div>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>Zerodha Kite</h2>
        <p className="text-sm text-muted">Live market data, order placement, portfolio, and margin data.</p>
      </div>

      <div className="form-row">
        <label className="form-label">Kite API Key</label>
        <input className="form-input" type="password" value={kiteKey} onChange={e => setKiteKey(e.target.value)}
          placeholder="Your Kite API key" />
      </div>

      <div className="form-row">
        <label className="form-label">Access Token</label>
        <input className="form-input" type="password" value={kiteToken} onChange={e => setKiteToken(e.target.value)}
          placeholder="Daily access token (from Kite login)" />
      </div>

      <p className="text-sm text-muted" style={{ marginBottom: 14, lineHeight: 1.5 }}>
        The access token changes every day. Generate it at{' '}
        <a href="https://developers.zerodha.com/api_login" target="_blank" rel="noreferrer"
          style={{ color: 'var(--accent-blue-bright)' }}>
          developers.zerodha.com
        </a>.
      </p>

      <div style={{ display: 'flex', gap: 10 }}>
        <button className="btn btn-secondary" style={{ flex: 1 }} onClick={testConnection} disabled={testing || loading}>
          {testing ? <><span className="spinner" /> Testing…</> : '🧪 Test'}
        </button>
        <button className="btn btn-primary" style={{ flex: 2 }} onClick={connect} disabled={loading || testing}>
          {loading ? <><span className="spinner" /> Connecting…</> : '🔌 Connect Kite'}
        </button>
      </div>

      {status && (
        <div className={`alert ${ok ? 'alert-success' : 'alert-error'} animate-fade-up`}
          style={{ marginTop: 12, fontSize: 12 }}>
          {status}
        </div>
      )}
    </div>
  )
}

/* ── Delta Exchange Connection Panel ────────── */
function DeltaPanel() {
  const [apiKey,    setApiKey]    = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [loading,   setLoading]   = useState(false)
  const [status,    setStatus]    = useState('')
  const [ok,        setOk]        = useState(false)
  const [userName,  setUserName]  = useState('')

  async function connect() {
    setLoading(true)
    setStatus('')
    try {
      const res = await api.post('/connect_delta', {
        api_key: apiKey, api_secret: apiSecret,
      })
      const success = res.data.connected === true
      setOk(success)
      setStatus(res.data.message || res.data.error)
      setUserName(res.data.user || '')
    } catch (e: any) {
      const msg = e.response?.data?.error || e.message
      setStatus('❌ ' + msg)
      setOk(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card" style={{ position: 'relative' }}>
      {/* Status badge top-right */}
      <div style={{ position: 'absolute', top: 18, right: 18 }}>
        <span className={`status-pill ${ok ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          {ok ? (userName || 'Connected') : 'Disconnected'}
        </span>
      </div>

      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 20, marginBottom: 4 }}>🪙</div>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>Delta Exchange</h2>
        <p className="text-sm text-muted">API credentials setup for live mock trading and portfolios on Delta.</p>
      </div>

      <div className="form-row">
        <label className="form-label">Delta API Key</label>
        <input className="form-input" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
          placeholder="Your Delta API Key" />
      </div>

      <div className="form-row">
        <label className="form-label">API Secret</label>
        <input className="form-input" type="password" value={apiSecret} onChange={e => setApiSecret(e.target.value)}
          placeholder="Your Delta API Secret" />
      </div>

      <button className="btn btn-primary w-full" style={{ marginTop: 8 }} onClick={connect} disabled={loading}>
        {loading ? <><span className="spinner" /> Connecting…</> : '🔌 Connect Delta'}
      </button>

      {status && (
        <div className={`alert ${ok ? 'alert-success' : 'alert-error'} animate-fade-up`}
          style={{ marginTop: 12, fontSize: 12 }}>
          {status}
        </div>
      )}
    </div>
  )
}

/* ── Finnhub Connection Panel ─────────────── */
function FinnhubPanel() {
  const [apiKey,    setApiKey]    = useState('')
  const [loading,   setLoading]   = useState(false)
  const [status,    setStatus]    = useState('')
  const [ok,        setOk]        = useState(false)

  async function connect() {
    setLoading(true)
    setStatus('')
    try {
      const res = await api.post('/connect_finnhub', { api_key: apiKey })
      const success = res.data.connected === true
      setOk(success)
      setStatus(res.data.message || res.data.error)
    } catch (e: any) {
      setStatus('❌ ' + (e.response?.data?.error || e.message))
      setOk(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card" style={{ position: 'relative' }}>
      <div style={{ position: 'absolute', top: 18, right: 18 }}>
        <span className={`status-pill ${ok ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />{ok ? 'Connected' : 'Disconnected'}
        </span>
      </div>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 20, marginBottom: 4 }}>📈</div>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>Finnhub</h2>
        <p className="text-sm text-muted">API credentials for global market OHLCV data.</p>
      </div>
      <div className="form-row"><label className="form-label">Finnhub API Key</label>
        <input className="form-input" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Your Finnhub API Key" />
      </div>
      <button className="btn btn-primary w-full" style={{ marginTop: 8 }} onClick={connect} disabled={loading}>
        {loading ? <><span className="spinner" /> Connecting…</> : '🔌 Connect Finnhub'}
      </button>
      {status && <div className={`alert ${ok ? 'alert-success' : 'alert-error'} animate-fade-up`} style={{ marginTop: 12, fontSize: 12 }}>{status}</div>}
    </div>
  )
}

/* ── API Status & Routing Strategy Panel ─────── */
function APIStatusPanel() {
  const [status, setStatus] = useState<Record<string, {
    connected: boolean;
    calls_last_minute: number;
    limit: number;
    rate_limited: boolean;
  }>>({})
  const [loading, setLoading] = useState(true)

  async function fetchStatus() {
    try {
      const res = await api.get('/api_status')
      setStatus(res.data)
    } catch (e) {
      console.error('Failed to fetch API status:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const apiNames: Record<string, string> = {
    kite: 'Zerodha Kite',
    delta: 'Delta Exchange',
    finnhub: 'Finnhub API',
    yfinance: 'Yahoo Finance'
  }

  return (
    <div className="card" style={{ gridColumn: 'span 2', marginTop: 12 }}>
      <div style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 20, marginBottom: 4 }}>📊</div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>API Routing & Rate Limits</h2>
          <p className="text-sm text-muted">
            Status and live metrics of backend data providers. Workload is dynamically routed based on rates and chart period.
          </p>
        </div>
        <button className="btn btn-secondary text-sm" onClick={fetchStatus} disabled={loading} style={{ padding: '6px 12px' }}>
          🔄 Refresh
        </button>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-secondary)', fontSize: 13 }}>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>API Provider</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Status</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Calls (60s window)</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Rate Limit</th>
              <th style={{ padding: '10px 12px', fontWeight: 600 }}>Load/Health</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(status).map(([key, value]) => {
              const capUsed = value.calls_last_minute
              const maxCap = value.limit
              const ratio = maxCap > 0 ? (capUsed / maxCap) * 100 : 0
              let loadStatus = 'Optimal ✅'
              let loadColor = 'var(--success)'
              
              if (value.rate_limited) {
                loadStatus = 'Rate Limited ⚠️'
                loadColor = 'var(--error)'
              } else if (ratio > 70) {
                loadStatus = 'High Load ⚡'
                loadColor = 'var(--warning)'
              }

              return (
                <tr key={key} style={{ borderBottom: '1px solid var(--border-color)', fontSize: 14 }}>
                  <td style={{ padding: '12px 12px', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {apiNames[key] || key}
                  </td>
                  <td style={{ padding: '12px 12px' }}>
                    <span className={`status-pill ${value.connected ? 'connected' : 'disconnected'}`}>
                      <span className="status-dot" />
                      {value.connected ? 'Connected' : 'Disconnected'}
                    </span>
                  </td>
                  <td style={{ padding: '12px 12px', color: 'var(--text-primary)' }}>
                    {capUsed}
                  </td>
                  <td style={{ padding: '12px 12px', color: 'var(--text-secondary)' }}>
                    {maxCap >= 9999 ? 'Unlimited' : `${maxCap} / min`}
                  </td>
                  <td style={{ padding: '12px 12px', color: loadColor, fontWeight: 600 }}>
                    {value.connected ? loadStatus : 'Unavailable'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 18, padding: 14, backgroundColor: 'var(--bg-muted, #f8f9fa)', borderRadius: 8, border: '1px solid var(--border-color)' }}>
        <h4 style={{ margin: '0 0 6px 0', fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>🔄 Routing Strategy Summary:</h4>
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <li><strong>Longer Charts (&gt;30 Days Lookback)</strong>: Always routed exclusively to <strong>Yahoo Finance API</strong> to conserve API rate limits on premium connections.</li>
          <li><strong>Shorter Charts (&le;30 Days Lookback)</strong>: Work is divided dynamically:
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              <li><strong>Crypto Assets</strong>: Handled by <strong>Delta Exchange API</strong> (falls back to Yahoo Finance if limited).</li>
              <li><strong>Indian Equities (NSE/BSE)</strong>: Handled by <strong>Zerodha Kite API</strong> or <strong>Finnhub</strong>, falling back to Yahoo Finance if rate limits are reached.</li>
              <li><strong>Global Equities</strong>: Handled by <strong>Finnhub API</strong>, falling back to Yahoo Finance if rate limits are reached.</li>
            </ul>
          </li>
        </ul>
      </div>
    </div>
  )
}

/* ── Main Config Page ───────────────────────── */
export default function ConfigPage({ setConnected, setKiteConnected }: Props) {
  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <h1 className="page-title">⚙️ Configuration</h1>
        <p className="page-subtitle">
          Connect your LLM agent, Zerodha Kite broker, and Delta Exchange API independently.
        </p>
      </div>

      {/* Independent connection panels */}
      <div className="grid grid-2" style={{ gap: 20, marginBottom: 24 }}>
        <LLMPanel setConnected={setConnected} />
        <KitePanel setKiteConnected={setKiteConnected} />
        <DeltaPanel />
        <FinnhubPanel />
        <APIStatusPanel />
      </div>

      {/* Info row */}
      <div className="grid grid-3" style={{ gap: 14 }}>
        {[
          { icon: '🌐', title: 'Yahoo Finance Fallback',
            desc: 'When Kite is disconnected, market data is fetched from Yahoo Finance automatically.' },
          { icon: '🧠', title: 'Multi-Agent Orchestrator',
            desc: 'LangGraph powers 5 sub-agents: Data, Analysis, Strategy, Execution, and Visualization.' },
          { icon: '🔐', title: 'Zero Persistence',
            desc: 'No keys are ever written to disk. They live only in the current session memory.' },
        ].map(({ icon, title, desc }) => (
          <div key={title} className="card" style={{ textAlign: 'center', padding: '22px 16px' }}>
            <div style={{ fontSize: 26, marginBottom: 10 }}>{icon}</div>
            <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 13 }}>{title}</div>
            <div className="text-sm text-muted">{desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
