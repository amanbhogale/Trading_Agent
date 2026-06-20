import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import ConfigPage from './pages/ConfigPage'
import ChatPage from './pages/ChatPage'
import AnalysisPage from './pages/AnalysisPage'
import BacktestPage from './pages/BacktestPage'
import PortfolioPage from './pages/PortfolioPage'
import TickerBar from './components/TickerBar'

const API_BASE = 'http://localhost:5000'

const NAV_ITEMS = [
  { path: '/',          icon: '⚙️',  label: 'Configuration' },
  { path: '/chat',      icon: '💬',  label: 'AI Agent Chat'  },
  { path: '/analysis',  icon: '📊',  label: 'Analysis'       },
  { path: '/backtest',  icon: '🧪',  label: 'Backtest'       },
  { path: '/portfolio', icon: '💼',  label: 'Portfolio'      },
]

function Sidebar({ connected }: { connected: boolean }) {
  const location = useLocation()
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">🪐</div>
        <div>
          <div className="sidebar-logo-name">Antigravity</div>
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
  '/analysis':  'Market Analysis',
  '/backtest':  'Strategy Backtest',
  '/portfolio': 'Live Portfolio',
}

function Topbar({ connected, kiteConnected }: { connected: boolean; kiteConnected: boolean }) {
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] ?? 'Trading Platform'
  return (
    <div className="topbar">
      <div className="topbar-left">
        <div className="topbar-title">{title}</div>
      </div>
      <div className="topbar-right">
        <span className={`status-pill ${kiteConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          Kite {kiteConnected ? 'Connected' : 'Disconnected'}
        </span>
        <span className={`status-pill ${connected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          LLM {connected ? 'Ready' : 'Not Set'}
        </span>
        <div className="avatar">Z</div>
      </div>
    </div>
  )
}

export default function App() {
  const [connected, setConnected] = useState(false)
  const [kiteConnected, setKiteConnected] = useState(false)

  return (
    <BrowserRouter>
      <div className="shell">
        <Sidebar connected={connected} />
        <div className="main">
          <Topbar connected={connected} kiteConnected={kiteConnected} />
          <TickerBar />
          <Routes>
            <Route path="/"          element={<ConfigPage   setConnected={setConnected} setKiteConnected={setKiteConnected} />} />
            <Route path="/chat"      element={<ChatPage />} />
            <Route path="/analysis"  element={<AnalysisPage />} />
            <Route path="/backtest"  element={<BacktestPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
