import { useState, useRef, useEffect } from 'react'
import api from '../api'


interface Message {
  role: 'user' | 'assistant'
  content: string
  html?: string
  time: string
}

const QUICK_PROMPTS = [
  'Analyse NSE:INFY and suggest a trade',
  'What is the current market sentiment for RELIANCE?',
  'Backtest SMA crossover on NSE:TCS for 365 days',
  'Show me the RSI and MACD for NSE:HDFC',
  'Compare momentum vs mean reversion strategy on WIPRO',
]

export default function ChatPage() {
  const [messages, setMessages]   = useState<Message[]>([])
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const historyRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight
    }
  }

  useEffect(scrollToBottom, [messages])

  const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  async function send(msg?: string) {
    const text = (msg ?? input).trim()
    if (!text || loading) return
    setInput('')
    setLoading(true)

    const userMsg: Message = { role: 'user', content: text, time: now() }
    setMessages(prev => [...prev, userMsg])

    try {
      const res = await api.post('/chat', { message: text })
      const aiMsg: Message = {
        role: 'assistant',
        content: '',
        html: res.data.response,
        time: now(),
      }
      setMessages(prev => [...prev, aiMsg])
    } catch (e: any) {
      const err = e.response?.data?.response || e.message
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `❌ Error: ${err}`, time: now() },
      ])
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="page" style={{ padding: 0, display: 'flex', flexDirection: 'column', flex: 1, height: '100%' }}>

      {/* Quick Prompts */}
      {messages.length === 0 && (
        <div style={{ padding: '28px 28px 0' }} className="animate-fade-up">
          <div className="page-header">
            <h1 className="page-title">💬 AI Agent Chat</h1>
            <p className="page-subtitle">Interact with the Orchestrator agent. It can analyse markets, backtest strategies, and execute trades.</p>
          </div>
          <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 20 }}>
            {QUICK_PROMPTS.map(q => (
              <div key={q} className="card" style={{ cursor: 'pointer', padding: '12px 16px', fontSize: 13 }}
                onClick={() => send(q)}>
                <span style={{ color: 'var(--accent-blue-bright)', marginRight: 8 }}>→</span>
                {q}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="chat-history" ref={historyRef} style={{ flex: 1, padding: '20px 28px' }}>
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {m.html
              ? <div className="md-output" dangerouslySetInnerHTML={{ __html: m.html }} />
              : <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            }
            <div style={{ fontSize: 10, opacity: 0.5, marginTop: 6, textAlign: m.role === 'user' ? 'right' : 'left' }}>
              {m.time}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '14px 16px' }}>
            <span className="spinner" />
            <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>Agent is thinking...</span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="chat-input-area" style={{ padding: '14px 28px' }}>
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
          placeholder="e.g. Analyse NSE:INFY and suggest a trade…"
        />
        <button className="btn btn-primary" onClick={() => send()} disabled={loading || !input.trim()}>
          {loading ? <span className="spinner" /> : '➤ Send'}
        </button>
      </div>
    </div>
  )
}
