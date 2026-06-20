import { useState } from 'react'
import axios from 'axios'

const API_BASE = 'http://localhost:5000/api'

const PROVIDERS = [
  { value: 'openrouter',    label: 'OpenRouter',    baseUrl: 'https://openrouter.ai/api/v1' },
  { value: 'openai',        label: 'OpenAI',         baseUrl: '' },
  { value: 'anthropic',     label: 'Anthropic',      baseUrl: '' },
  { value: 'google-gemini', label: 'Google Gemini',  baseUrl: '' },
]

interface Props {
  setConnected: (v: boolean) => void
  setKiteConnected: (v: boolean) => void
}

export default function ConfigPage({ setConnected, setKiteConnected }: Props) {
  const [provider, setProvider] = useState('openrouter')
  const [model, setModel]       = useState('nvidia/nemotron-super-49b-v1:free')
  const [apiKey, setApiKey]     = useState('')
  const [baseUrl, setBaseUrl]   = useState('https://openrouter.ai/api/v1')
  const [kiteKey, setKiteKey]   = useState('')
  const [kiteToken, setKiteToken] = useState('')

  const [connectStatus, setConnectStatus] = useState('')
  const [kiteStatus, setKiteStatus]       = useState('')
  const [connecting, setConnecting]       = useState(false)
  const [testingKite, setTestingKite]     = useState(false)

  function handleProviderChange(val: string) {
    setProvider(val)
    const p = PROVIDERS.find(p => p.value === val)
    if (p) setBaseUrl(p.baseUrl)
  }

  async function handleConnect() {
    setConnecting(true)
    setConnectStatus('')
    try {
      const res = await axios.post(`${API_BASE}/connect`, {
        provider, model, api_key: apiKey, base_url: baseUrl, kite_key: kiteKey, kite_token: kiteToken
      })
      setConnectStatus(res.data.message || res.data.error)
      const success = !res.data.error
      setConnected(success)
    } catch (e: any) {
      setConnectStatus('❌ ' + (e.response?.data?.error || e.message))
    } finally {
      setConnecting(false)
    }
  }

  async function handleTestKite() {
    setTestingKite(true)
    setKiteStatus('')
    try {
      const res = await axios.post(`${API_BASE}/test_kite`, { kite_key: kiteKey, kite_token: kiteToken })
      const msg = res.data.message || ''
      setKiteStatus(msg)
      setKiteConnected(msg.includes('True'))
    } catch (e: any) {
      setKiteStatus('❌ ' + e.message)
    } finally {
      setTestingKite(false)
    }
  }

  const isSuccess = (s: string) => s.includes('✅') || s.includes('Connected')
  const isError   = (s: string) => s.includes('❌') || s.includes('Error') || s.includes('False')

  return (
    <div className="page animate-fade-up">
      <div className="page-header">
        <h1 className="page-title">⚙️ Configuration</h1>
        <p className="page-subtitle">Connect your LLM provider and Zerodha Kite broker to power the trading agent.</p>
      </div>

      <div className="grid grid-6-6" style={{ gap: '20px' }}>

        {/* ── LLM Provider ── */}
        <div className="card card-glow">
          <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 18, color: 'var(--accent-blue-bright)' }}>
            🤖 LLM Provider
          </h2>

          <div className="form-row">
            <label className="form-label">Provider</label>
            <select className="form-select" value={provider} onChange={e => handleProviderChange(e.target.value)}>
              {PROVIDERS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
            </select>
          </div>

          <div className="form-row">
            <label className="form-label">Model</label>
            <input className="form-input" value={model} onChange={e => setModel(e.target.value)}
              placeholder="e.g. gpt-4o or nvidia/nemotron-super-49b-v1:free" />
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
        </div>

        {/* ── Kite Broker ── */}
        <div className="card">
          <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 18, color: 'var(--accent-blue-bright)' }}>
            🪁 Zerodha Kite Broker
          </h2>

          <div className="form-row">
            <label className="form-label">Kite API Key</label>
            <input className="form-input" type="password" value={kiteKey} onChange={e => setKiteKey(e.target.value)}
              placeholder="Your Kite API key" />
          </div>

          <div className="form-row">
            <label className="form-label">Access Token</label>
            <input className="form-input" type="password" value={kiteToken} onChange={e => setKiteToken(e.target.value)}
              placeholder="Daily access token" />
          </div>

          <button className="btn btn-secondary w-full mt-4" onClick={handleTestKite} disabled={testingKite}>
            {testingKite ? <><span className="spinner" /> Testing...</> : '🧪 Test Kite Connection'}
          </button>

          {kiteStatus && (
            <div className={`alert ${isError(kiteStatus) ? 'alert-error' : 'alert-success'}`}>
              {kiteStatus.replace(/\*\*/g, '')}
            </div>
          )}
        </div>
      </div>

      {/* ── Connect Button ── */}
      <div className="card mt-4" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
            Ready to connect?
          </div>
          <div className="text-sm text-muted">
            This will initialize the LangGraph orchestrator and all sub-agents.
          </div>
        </div>
        <button className="btn btn-primary btn-lg" onClick={handleConnect} disabled={connecting}>
          {connecting ? <><span className="spinner" /> Connecting...</> : '🔌 Connect Agent'}
        </button>
      </div>

      {connectStatus && (
        <div className={`alert ${isError(connectStatus) ? 'alert-error' : 'alert-success'} animate-fade-up`}
          style={{ whiteSpace: 'pre-wrap', marginTop: 12 }}>
          {connectStatus}
        </div>
      )}

      {/* ── Info Cards ── */}
      <div className="grid grid-3 mt-6">
        {[
          { icon: '🌐', title: 'Market Data', desc: 'Yahoo Finance fallback + Kite live feed for Indian markets.' },
          { icon: '🧠', title: 'Multi-Agent AI', desc: 'LangGraph orchestrator with specialized sub-agents for analysis, execution, and strategy.' },
          { icon: '🔐', title: 'Secure Credentials', desc: 'All API keys stay in memory — never persisted to disk.' },
        ].map(({ icon, title, desc }) => (
          <div key={title} className="card" style={{ textAlign: 'center', padding: '24px 18px' }}>
            <div style={{ fontSize: 28, marginBottom: 10 }}>{icon}</div>
            <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 14 }}>{title}</div>
            <div className="text-sm text-muted">{desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
