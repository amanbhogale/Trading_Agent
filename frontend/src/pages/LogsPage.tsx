import { useState, useEffect, useRef } from 'react'
import api from '../api'

interface LogRecord {
  time:    string
  level:   string
  name:    string
  message: string
}

const LEVEL_COLORS: Record<string, string> = {
  DEBUG:    '#6b7280',
  INFO:     '#60a5fa',
  WARNING:  '#f59e0b',
  ERROR:    '#f43f5e',
  CRITICAL: '#dc2626',
}

const LEVEL_BG: Record<string, string> = {
  DEBUG:    'rgba(107,114,128,0.08)',
  INFO:     'rgba(96,165,250,0.06)',
  WARNING:  'rgba(245,158,11,0.08)',
  ERROR:    'rgba(244,63,94,0.10)',
  CRITICAL: 'rgba(220,38,38,0.15)',
}

export default function LogsPage() {
  const [logs,       setLogs]       = useState<LogRecord[]>([])
  const [filter,     setFilter]     = useState('')
  const [levelFilter,setLevelFilter]= useState('ALL')
  const [autoScroll, setAutoScroll] = useState(true)
  const [polling,    setPolling]    = useState(true)
  const [lastCount,  setLastCount]  = useState(0)
  const bottomRef = useRef<HTMLDivElement>(null)

  async function fetchLogs() {
    try {
      const res = await api.get('/logs?n=300')
      const records: LogRecord[] = res.data.logs || []
      setLogs(records)
      if (records.length !== lastCount) {
        setLastCount(records.length)
        if (autoScroll) {
          setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
        }
      }
    } catch {
      // backend may be restarting
    }
  }

  useEffect(() => {
    fetchLogs()
    if (!polling) return
    const id = setInterval(fetchLogs, 2000)
    return () => clearInterval(id)
  }, [polling, autoScroll, lastCount])

  const filtered = logs.filter(l => {
    const matchLevel = levelFilter === 'ALL' || l.level === levelFilter
    const matchText  = !filter || l.message.toLowerCase().includes(filter.toLowerCase()) ||
                       l.name.toLowerCase().includes(filter.toLowerCase())
    return matchLevel && matchText
  })

  function levelCount(lvl: string) {
    return logs.filter(l => l.level === lvl).length
  }

  return (
    <div className="page animate-fade-up" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: 0 }}>

      {/* Header */}
      <div style={{ padding: '20px 28px 0' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <div>
            <h1 className="page-title">🖥️ Gateway Logs</h1>
            <p className="page-subtitle">Live Flask backend logs — OpenRouter requests, Kite API calls, errors.</p>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className={`btn btn-sm ${polling ? 'btn-success' : 'btn-secondary'}`}
              onClick={() => setPolling(p => !p)}
            >
              {polling ? '⏸ Pause' : '▶ Resume'} Live
            </button>
            <button className="btn btn-sm btn-secondary" onClick={() => setLogs([])}>
              🗑 Clear
            </button>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-muted)', cursor: 'pointer' }}>
              <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
              Auto-scroll
            </label>
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <input
            className="form-input"
            style={{ flex: 1, minWidth: 200, padding: '7px 12px', fontSize: 12 }}
            placeholder="Filter by message or logger name…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
          {['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].map(lvl => (
            <button
              key={lvl}
              className="btn btn-sm"
              onClick={() => setLevelFilter(lvl)}
              style={{
                background: levelFilter === lvl ? (LEVEL_COLORS[lvl] || 'var(--accent-blue)') : 'var(--bg-elevated)',
                border: `1px solid ${levelFilter === lvl ? (LEVEL_COLORS[lvl] || 'var(--accent-blue)') : 'var(--border-mid)'}`,
                color: levelFilter === lvl ? 'white' : 'var(--text-muted)',
                borderRadius: 99,
                fontSize: 11,
                padding: '3px 10px',
              }}
            >
              {lvl} {lvl !== 'ALL' && levelCount(lvl) > 0 && (
                <span style={{
                  background: 'rgba(255,255,255,0.2)',
                  borderRadius: 99,
                  padding: '0 5px',
                  marginLeft: 3,
                  fontSize: 10,
                }}>
                  {levelCount(lvl)}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Stats row */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 14, fontSize: 11, color: 'var(--text-muted)' }}>
          <span>Total: <strong style={{ color: 'var(--text-primary)' }}>{logs.length}</strong></span>
          <span>Shown: <strong style={{ color: 'var(--text-primary)' }}>{filtered.length}</strong></span>
          <span style={{ color: levelCount('ERROR') > 0 ? 'var(--danger)' : 'var(--text-muted)' }}>
            Errors: <strong>{levelCount('ERROR')}</strong>
          </span>
          <span style={{ color: levelCount('WARNING') > 0 ? 'var(--warning)' : 'var(--text-muted)' }}>
            Warnings: <strong>{levelCount('WARNING')}</strong>
          </span>
          {polling && (
            <span style={{ color: 'var(--success)' }}>● Polling every 2s</span>
          )}
        </div>
      </div>

      {/* Log pane */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0 28px 20px',
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        fontSize: 12,
      }}>
        {filtered.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
            {logs.length === 0
              ? '⏳ Waiting for log events… Make sure Flask is running.'
              : '🔍 No logs match the current filter.'}
          </div>
        ) : (
          filtered.map((l, i) => (
            <div
              key={i}
              style={{
                display: 'grid',
                gridTemplateColumns: '60px 64px 180px 1fr',
                gap: '0 12px',
                padding: '4px 10px',
                borderRadius: 4,
                background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)',
                alignItems: 'flex-start',
                lineHeight: 1.5,
                background: LEVEL_BG[l.level] || 'transparent',
                borderLeft: l.level === 'ERROR' || l.level === 'CRITICAL'
                  ? `2px solid ${LEVEL_COLORS[l.level]}`
                  : '2px solid transparent',
              }}
            >
              <span style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{l.time}</span>
              <span style={{
                color: LEVEL_COLORS[l.level] || 'var(--text-muted)',
                fontWeight: 700,
                fontSize: 10,
                paddingTop: 2,
                textTransform: 'uppercase',
              }}>
                {l.level}
              </span>
              <span style={{
                color: 'var(--text-muted)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                fontSize: 10,
                paddingTop: 2,
              }}>
                {l.name}
              </span>
              <span style={{
                color: l.level === 'ERROR' || l.level === 'CRITICAL'
                  ? LEVEL_COLORS[l.level]
                  : l.level === 'WARNING'
                    ? LEVEL_COLORS[l.level]
                    : 'var(--text-primary)',
                wordBreak: 'break-all',
                whiteSpace: 'pre-wrap',
              }}>
                {l.message}
              </span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
