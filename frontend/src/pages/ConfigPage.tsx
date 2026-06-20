import { useState } from 'react'
import api from '../api'


const PROVIDERS = [
  { value: 'openrouter',    label: 'OpenRouter',   baseUrl: 'https://openrouter.ai/api/v1' },
  { value: 'openai',        label: 'OpenAI',        baseUrl: '' },
  { value: 'anthropic',     label: 'Anthropic',     baseUrl: '' },
  { value: 'google-gemini', label: 'Google Gemini', baseUrl: '' },
]

interface Props {
  setConnected:     (v: boolean) => void
  setKiteConnected: (v: boolean) => void
}

/* ── LLM Connection Panel ───────────────────── */
function LLMPanel({ setConnected }: { setConnected: (v: boolean) => void }) {
  const [provider,    setProvider]    = useState('openrouter')
  const [model,       setModel]       = useState('nvidia/nemotron-super-49b-v1:free')
  const [apiKey,      setApiKey]      = useState('')
  const [baseUrl,     setBaseUrl]     = useState('https://openrouter.ai/api/v1')
  const [loading,     setLoading]     = useState(false)
  const [status,      setStatus]      = useState('')
  const [ok,          setOk]          = useState(false)

  function handleProvider(val: string) {
    setProvider(val)
    const p = PROVIDERS.find(p => p.value === val)
    if (p) setBaseUrl(p.baseUrl)
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
          placeholder="e.g. gpt-4o, claude-3-5-sonnet, gemini-pro" />
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

/* ── Main Config Page ───────────────────────── */
export default function ConfigPage({ setConnected, setKiteConnected }: Props) {
  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <h1 className="page-title">⚙️ Configuration</h1>
        <p className="page-subtitle">
          Connect your LLM agent and Zerodha Kite broker independently — each has its own status.
        </p>
      </div>

      {/* Two independent connection panels */}
      <div className="grid grid-6-6" style={{ gap: 20, marginBottom: 24 }}>
        <LLMPanel setConnected={setConnected} />
        <KitePanel setKiteConnected={setKiteConnected} />
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
