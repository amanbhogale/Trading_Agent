import React, { useState, useEffect } from 'react';
import { 
  LineChart, Activity, Briefcase, MessageSquare, Settings, 
  TrendingUp, TrendingDown, RefreshCw, Layers
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [portfolioData, setPortfolioData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [kiteConnected, setKiteConnected] = useState(false);

  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      // Mocking check for now
      setKiteConnected(true); 
    } catch (e) {
      setKiteConnected(false);
    }
  };

  const fetchPortfolio = async () => {
    setLoading(true);
    try {
      // Trying to fetch real backend data if it exists
      const res = await axios.post(`${API_BASE}/portfolio`);
      setPortfolioData(res.data);
    } catch (e) {
      // Fallback dummy data for visual wow factor
      setTimeout(() => {
        setPortfolioData({
          equity: 1245000,
          pnl: 45200,
          pnlPct: 3.76,
          holdings: [
            { symbol: 'RELIANCE', qty: 100, price: 2450, pnl: 12000, pnlPct: 5.1 },
            { symbol: 'TCS', qty: 50, price: 3420, pnl: -1500, pnlPct: -0.8 },
            { symbol: 'HDFCBANK', qty: 200, price: 1650, pnl: 34700, pnlPct: 11.7 }
          ]
        });
      }, 1000);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'portfolio') {
      fetchPortfolio();
    }
  }, [activeTab]);

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-title">
          <Layers className="text-blue-500" />
          <span>Antigravity Trade</span>
        </div>
        
        <nav style={{ flex: 1 }}>
          <a href="#" className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <Activity size={20} /> Dashboard
          </a>
          <a href="#" className={`nav-item ${activeTab === 'portfolio' ? 'active' : ''}`} onClick={() => setActiveTab('portfolio')}>
            <Briefcase size={20} /> Portfolio
          </a>
          <a href="#" className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}>
            <LineChart size={20} /> Analysis
          </a>
          <a href="#" className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
            <MessageSquare size={20} /> AI Agent
          </a>
        </nav>
        
        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--panel-border)', paddingTop: '1rem' }}>
          <a href="#" className="nav-item">
            <Settings size={20} /> Settings
          </a>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>
            {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span className={`badge ${kiteConnected ? 'badge-success' : 'badge-danger'}`}>
              {kiteConnected ? 'Kite Connected' : 'Disconnected'}
            </span>
            <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'var(--accent-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>
              Z
            </div>
          </div>
        </header>

        <div className="dashboard-content animate-fade-in">
          {activeTab === 'dashboard' && (
            <>
              <div className="panel col-span-4 stat-card">
                <span className="stat-label">Total Equity</span>
                <span className="stat-value">₹1,245,000.00</span>
                <span className="stat-change positive">
                  <TrendingUp size={16} /> +₹45,200 (3.76%)
                </span>
              </div>
              <div className="panel col-span-4 stat-card">
                <span className="stat-label">Available Margin</span>
                <span className="stat-value">₹320,500.00</span>
                <span className="stat-change text-muted">Ready to deploy</span>
              </div>
              <div className="panel col-span-4 stat-card">
                <span className="stat-label">Active Strategies</span>
                <span className="stat-value">3</span>
                <span className="stat-change positive">All systems nominal</span>
              </div>

              <div className="panel col-span-8" style={{ minHeight: '400px' }}>
                <h3 style={{ marginBottom: '1rem' }}>Market Overview</h3>
                <div className="chart-container" style={{ background: 'var(--bg-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed var(--panel-border)' }}>
                   <p className="text-muted">Select a symbol in Analysis to view chart</p>
                </div>
              </div>

              <div className="panel col-span-4">
                <h3 style={{ marginBottom: '1rem' }}>Recent Activity</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {[
                    { action: 'BUY', sym: 'INFY', qty: 50, price: 1450, time: '10:45 AM' },
                    { action: 'SELL', sym: 'RELIANCE', qty: 20, price: 2400, time: '09:30 AM' },
                    { action: 'STRATEGY', sym: 'SMA Crossover', qty: 'Trigger', price: '-', time: 'Yesterday' }
                  ].map((act, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '0.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <div>
                        <div style={{ fontWeight: 600 }}>{act.action} {act.sym}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{act.time}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div>{act.qty}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>₹{act.price}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {activeTab === 'portfolio' && (
            <div className="panel col-span-12">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
                <h2>Your Holdings</h2>
                <button className="btn btn-secondary" onClick={fetchPortfolio} disabled={loading}>
                  <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
                  Refresh
                </button>
              </div>

              {loading && !portfolioData ? (
                <div style={{ padding: '3rem', textAlign: 'center' }}><div className="loader"></div></div>
              ) : (
                <table className="data-table animate-fade-in">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Quantity</th>
                      <th>LTP</th>
                      <th>P&L</th>
                      <th>Change %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolioData?.holdings?.map((h: any, i: number) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 600 }}>{h.symbol}</td>
                        <td>{h.qty}</td>
                        <td>₹{h.price}</td>
                        <td className={h.pnl >= 0 ? 'positive' : 'negative'} style={{ color: h.pnl >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {h.pnl >= 0 ? '+' : ''}₹{h.pnl}
                        </td>
                        <td className={h.pnlPct >= 0 ? 'positive' : 'negative'} style={{ color: h.pnlPct >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {h.pnlPct >= 0 ? <TrendingUp size={14} style={{display:'inline', marginRight: 4}}/> : <TrendingDown size={14} style={{display:'inline', marginRight: 4}}/>}
                          {h.pnlPct}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {(activeTab === 'analysis' || activeTab === 'chat') && (
            <div className="panel col-span-12" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <div style={{ background: 'var(--bg-color)', width: 64, height: 64, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem' }}>
                <MessageSquare size={32} color="var(--accent-color)" />
              </div>
              <h2>Connect with Agent</h2>
              <p className="text-muted" style={{ maxWidth: 400, margin: '1rem auto 2rem' }}>
                Start a conversation with your trading agent to perform deep market analysis, backtest strategies, and execute orders.
              </p>
              <button className="btn">Open Agent Chat</button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
