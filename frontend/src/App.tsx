import { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import ConfigPage from './pages/ConfigPage'
import ChatPage from './pages/ChatPage'
import AnalysisPage from './pages/AnalysisPage'
import BacktestPage from './pages/BacktestPage'
import PortfolioPage from './pages/PortfolioPage'
import LogsPage from './pages/LogsPage'
import ChartPage from './pages/ChartPage'
import NewsPage from './pages/NewsPage'
import TickerBar from './components/TickerBar'

const NAV_ITEMS = [
  { path: '/',          icon: '⚙️',  label: 'Configuration' },
  { path: '/chat',      icon: '💬',  label: 'AI Agent Chat'  },
  { path: '/chart',     icon: '📉',  label: 'Charts'         },
  { path: '/analysis',  icon: '📊',  label: 'Analysis'       },
  { path: '/backtest',  icon: '🧪',  label: 'Backtest'       },
  { path: '/portfolio', icon: '💼',  label: 'Portfolio'      },
  { path: '/news',      icon: '📰',  label: 'News Feed'      },
  { path: '/logs',      icon: '🖥️',  label: 'Gateway Logs'  },
]

function Sidebar({ connected }: { connected: boolean }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <img src="/logo.png" alt="Auxesis Logo" style={{ width: 80, height: 'auto', objectFit: 'contain' }} />
        <div>
          <div className="sidebar-logo-name">Auxesis</div>
          <div className="sidebar-logo-sub">Trade Platform</div>
        </div>
      </div>

      <div className="sidebar-section-label">Navigation</div>

      <nav style={{ flex: 1 }}>
        {NAV_ITEMS.map(({ path, icon, label }) => (
          <NavLink
            key={path}
            to={path}
            end={path === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className={`status-pill ${connected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          {connected ? 'Agent Connected' : 'Not Connected'}
        </div>
      </div>
    </aside>
  )
}

const PAGE_TITLES: Record<string, string> = {
  '/':          'Configuration',
  '/chat':      'AI Agent Chat',
  '/chart':     'TradingView Charts',
  '/analysis':  'Market Analysis',
  '/backtest':  'Strategy Backtest',
  '/portfolio': 'Live Portfolio',
  '/news':      'Market News Feed',
  '/logs':      'Gateway Logs',
}

interface TopbarProps {
  connected: boolean
  kiteConnected: boolean
  mode: string
  onModeChange: (m: string) => void
}

function Topbar({ connected, kiteConnected, mode, onModeChange }: TopbarProps) {
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] ?? 'Trading Platform'
  return (
    <div className="topbar">
      <div className="topbar-left">
        <div className="topbar-title">{title}</div>
      </div>
      <div className="topbar-right">
        {/* Mode Selector Toggle Pills */}
        <div className="mode-selector" style={{ display: 'flex', alignItems: 'center', gap: '6px', marginRight: '16px' }}>
          <span style={{ fontSize: '10px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', marginRight: '4px' }}>Market:</span>
          <div style={{ display: 'flex', background: 'var(--bg-panel)', borderRadius: '99px', padding: '3px', border: '1px solid var(--border-subtle)' }}>
            {[
              { id: 'equity', label: '🇮🇳 Equity' },
              { id: 'forex', label: '💱 Forex' },
              { id: 'crypto', label: '🪙 Crypto' }
            ].map(item => {
              const isActive = mode === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onModeChange(item.id)}
                  style={{
                    background: isActive ? 'var(--accent-blue-bright, #3b82f6)' : 'transparent',
                    border: 'none',
                    borderRadius: '99px',
                    color: isActive ? '#ffffff' : 'var(--text-muted)',
                    padding: '5px 12px',
                    fontSize: '11px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: isActive ? '0 0 12px rgba(59, 130, 246, 0.45)' : 'none',
                  }}
                >
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>

        <span className={`status-pill ${kiteConnected ? 'connected' : 'disconnected'}`}
          title={kiteConnected ? 'Kite broker is connected — live data & trading enabled' : 'Kite not connected — using Yahoo Finance fallback'}>
          <span className="status-dot" />
          🪁 Kite {kiteConnected ? 'Live' : 'Off'}
        </span>
        <span className={`status-pill ${connected ? 'connected' : 'disconnected'}`}
          title={connected ? 'LLM agent is ready' : 'Connect LLM in Configuration'}>
          <span className="status-dot" />
          🤖 LLM {connected ? 'Ready' : 'Off'}
        </span>
        <div className="avatar" title="User">Z</div>
      </div>
    </div>
  )
}

export default function App() {
  const [connected, setConnected] = useState(false)
  const [kiteConnected, setKiteConnected] = useState(false)
  const [mode, setMode] = useState<string>(() => localStorage.getItem('trading_mode') || 'equity')

  const handleModeChange = (newMode: string) => {
    setMode(newMode)
    localStorage.setItem('trading_mode', newMode)
    window.dispatchEvent(new CustomEvent('tradingModeChanged', { detail: { mode: newMode } }))
  }

  return (
    <BrowserRouter>
      <div className="shell">
        <Sidebar connected={connected} />
        <div className="main">
          <Topbar connected={connected} kiteConnected={kiteConnected} mode={mode} onModeChange={handleModeChange} />
          <TickerBar mode={mode} />
          <Routes>
            <Route path="/"          element={<ConfigPage   setConnected={setConnected} setKiteConnected={setKiteConnected} />} />
            <Route path="/chat"      element={<ChatPage key={mode} mode={mode} />} />
            <Route path="/chart"     element={<ChartPage key={mode} mode={mode} />} />
            <Route path="/analysis"  element={<AnalysisPage key={mode} mode={mode} />} />
            <Route path="/backtest"  element={<BacktestPage key={mode} mode={mode} />} />
            <Route path="/portfolio" element={<PortfolioPage key={mode} mode={mode} />} />
            <Route path="/news"      element={<NewsPage key={mode} mode={mode} />} />
            <Route path="/logs"      element={<LogsPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
