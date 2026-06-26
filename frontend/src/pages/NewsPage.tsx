import { useState, useEffect } from 'react'
import api from '../api'

interface NewsItem {
  title: string
  link: string
  pub_date: string
  description: string
  source?: string
  region?: string
  method?: string
  confidence?: number
}

const getSourceStyles = (source: string) => {
  const src = (source || '').toLowerCase();
  if (src.includes('bloomberg')) {
    return { bg: '#ff500015', color: '#ff5000', label: 'Bloomberg' };
  }
  if (src.includes('reuters')) {
    return { bg: '#ff800015', color: '#ff8000', label: 'Reuters' };
  }
  if (src.includes('financial times') || src.includes('ft')) {
    return { bg: '#fff1e0', color: '#9e2f50', label: 'Financial Times' };
  }
  if (src.includes('journal') || src.includes('wsj') || src.includes('street')) {
    return { bg: '#e5e7eb', color: '#111827', label: 'WSJ' };
  }
  if (src.includes('cnbc')) {
    return { bg: '#0c1c2a', color: '#3b82f6', label: 'CNBC' };
  }
  if (src.includes('cointelegraph')) {
    return { bg: '#fab81415', color: '#fab814', label: 'Cointelegraph' };
  }
  if (src.includes('coindesk')) {
    return { bg: '#ffe60015', color: '#d97706', label: 'Coindesk' };
  }
  if (src.includes('investing')) {
    return { bg: '#1d4ed815', color: '#2563eb', label: 'Investing' };
  }
  if (src.includes('dailyfx')) {
    return { bg: '#0f172a', color: '#38bdf8', label: 'DailyFX' };
  }
  return { bg: 'var(--bg-elevated)', color: 'var(--text-secondary)', label: source || 'News' };
}

interface NewsPageProps {
  mode: string
}

const getSentimentDetails = (label: string) => {
  switch ((label || 'NEUTRAL').toUpperCase()) {
    case 'BULLISH':
      return {
        color: '#10b981', // green
        glow: 'rgba(16, 185, 129, 0.25)',
        text: 'Bullish Sentiment',
        emoji: '📈',
        desc: 'Positive news flow dominates. Buyers are in control.'
      }
    case 'BEARISH':
      return {
        color: '#f43f5e', // red
        glow: 'rgba(244, 63, 94, 0.25)',
        text: 'Bearish Sentiment',
        emoji: '📉',
        desc: 'Negative news flow dominates. Selling pressure persists.'
      }
    case 'NEUTRAL':
    default:
      return {
        color: '#f59e0b', // yellow
        glow: 'rgba(245, 158, 11, 0.25)',
        text: 'Neutral Sentiment',
        emoji: '⚖️',
        desc: 'Balanced news flow. Market searching for direction.'
      }
  }
}

const getVixDetails = (score: number) => {
  if (score < 15) {
    return {
      rating: 'Low Volatility',
      color: '#10b981', // green
      glow: 'rgba(16, 185, 129, 0.15)',
      desc: 'Market is calm. Complacency or stable trend expected.'
    }
  } else if (score < 22) {
    return {
      rating: 'Moderate Volatility',
      color: '#8ea6ba', // blue-grey accent
      glow: 'rgba(142, 166, 186, 0.15)',
      desc: 'Normal market movement. Standard fluctuation levels.'
    }
  } else if (score < 30) {
    return {
      rating: 'Elevated Panic',
      color: '#f59e0b', // yellow
      glow: 'rgba(245, 158, 11, 0.15)',
      desc: 'Increased caution. Market showing signs of fear.'
    }
  } else {
    return {
      rating: 'Extreme Stress',
      color: '#f43f5e', // red
      glow: 'rgba(244, 63, 94, 0.15)',
      desc: 'High panic! Market under significant distress.'
    }
  }
}

export default function NewsPage({ mode }: NewsPageProps) {
  const [news, setNews] = useState<NewsItem[]>([])
  const [vixScore, setVixScore] = useState<number>(15.0)
  const [sentiment, setSentiment] = useState<{ score: number; label: string }>({ score: 0.0, label: 'NEUTRAL' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [regionFilter, setRegionFilter] = useState<'ALL' | 'INDIAN' | 'GLOBAL'>('ALL')

  const filteredNews = news.filter(item => {
    if (regionFilter === 'ALL') return true;
    return (item.region || 'Global').toUpperCase() === regionFilter;
  });

  async function fetchNews() {
    setLoading(true)
    setError('')
    try {
      const res = await api.get(`/news?mode=${mode}`)
      if (res.data) {
        if (res.data.news) {
          setNews(res.data.news)
        } else {
          setNews([])
        }
        if (res.data.vix_score !== undefined) {
          setVixScore(res.data.vix_score)
        }
        if (res.data.sentiment) {
          setSentiment(res.data.sentiment)
        }
      }
    } catch (err: any) {
      setError(err.friendlyMessage || err.message || 'Failed to fetch news feed.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchNews()
  }, [mode])

  const getModeLabel = () => {
    if (mode === 'crypto') return '🪙 Crypto News'
    if (mode === 'forex') return '💱 Forex News'
    return '🇮🇳 Equity News'
  }


  return (
    <div className="page animate-fade-up">
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '20px' }}>
        <div>
          <h1 className="page-title">📰 News Feed</h1>
          <p className="page-subtitle">
            Stay updated with real-time financial RSS feeds for <strong>{getModeLabel()}</strong>.
          </p>
        </div>
        <button className="btn btn-primary" onClick={fetchNews} disabled={loading}>
          {loading ? <><span className="spinner" /> Loading…</> : '🔄 Refresh News'}
        </button>
      </div>

      {error && (
        <div className="alert alert-error animate-fade-up" style={{ marginBottom: 18 }}>
          ⚠️ {error}
        </div>
      )}

      {/* Region filter tabs & Engine Status */}
      {!error && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[
              { key: 'ALL', label: '🌍 All News' },
              { key: 'INDIAN', label: '🇮🇳 Indian Markets' },
              { key: 'GLOBAL', label: '🌐 Global Markets' }
            ].map(tab => (
              <button
                key={tab.key}
                className={`btn ${regionFilter === tab.key ? 'btn-primary' : 'btn-secondary'}`}
                style={{
                  padding: '6px 16px',
                  borderRadius: '99px',
                  fontSize: '12px',
                  fontWeight: 700,
                  boxShadow: regionFilter === tab.key ? '0 0 12px var(--accent-glow)' : 'none'
                }}
                onClick={() => setRegionFilter(tab.key as any)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(99, 102, 241, 0.08)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            borderRadius: '8px',
            padding: '6px 12px',
            fontSize: '11px',
            fontWeight: 700,
            color: '#a5b4fc'
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#10b981', boxShadow: '0 0 6px #10b981' }} />
            Classification Engine: <strong>Naive Bayes ML Active</strong>
          </div>
        </div>
      )}

      {/* Market Sentiment Dashboard */}
      {!error && (
        <div className="grid grid-2" style={{ marginBottom: '24px', gap: '20px' }}>
          {/* Sentiment Gauge Card */}
          <div className="card card-glow" style={{
            position: 'relative',
            padding: '24px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '24px',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: '-50px',
              right: '-50px',
              width: '120px',
              height: '120px',
              borderRadius: '50%',
              background: loading ? 'rgba(255, 255, 255, 0.03)' : getSentimentDetails(sentiment.label).glow,
              filter: 'blur(35px)',
              pointerEvents: 'none',
              transition: 'background 0.5s ease'
            }} />
            
            <div style={{ position: 'relative', width: '80px', height: '80px', flexShrink: 0 }}>
              <svg width="80" height="80" viewBox="0 0 80 80">
                <circle
                  cx="40"
                  cy="40"
                  r="34"
                  fill="none"
                  stroke="var(--bg-elevated)"
                  strokeWidth="6"
                />
                <circle
                  cx="40"
                  cy="40"
                  r="34"
                  fill="none"
                  stroke={loading ? 'var(--border-subtle)' : getSentimentDetails(sentiment.label).color}
                  strokeWidth="6"
                  strokeDasharray="213.6"
                  strokeDashoffset={loading ? 213.6 : 213.6 * (1 - (sentiment.score + 1) / 2)}
                  strokeLinecap="round"
                  transform="rotate(-90 40 40)"
                  style={{
                    transition: 'stroke-dashoffset 0.8s ease, stroke 0.5s ease',
                    filter: loading ? 'none' : `drop-shadow(0 0 4px ${getSentimentDetails(sentiment.label).color})`
                  }}
                />
              </svg>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '80px',
                height: '80px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1
              }}>
                <span style={{ fontSize: '15px', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {loading ? '...' : (sentiment.score > 0 ? `+${sentiment.score.toFixed(2)}` : sentiment.score.toFixed(2))}
                </span>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: '-2px', fontWeight: 700 }}>
                  Score
                </div>
              </div>
            </div>

            <div style={{ flex: 1, zIndex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '18px' }}>{loading ? '⏳' : getSentimentDetails(sentiment.label).emoji}</span>
                <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
                  {loading ? 'Analyzing Sentiment...' : getSentimentDetails(sentiment.label).text}
                </h3>
              </div>
              <p className="text-sm text-muted" style={{ lineHeight: '1.4', margin: 0 }}>
                {loading ? 'Evaluating the tone of recent news headlines...' : getSentimentDetails(sentiment.label).desc}
              </p>
              {!loading && (
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px',
                  marginTop: '10px',
                  padding: '3px 10px',
                  borderRadius: '99px',
                  fontSize: '11px',
                  fontWeight: 800,
                  backgroundColor: getSentimentDetails(sentiment.label).glow,
                  color: getSentimentDetails(sentiment.label).color,
                  border: `1px solid ${getSentimentDetails(sentiment.label).color}40`,
                  boxShadow: `0 0 8px ${getSentimentDetails(sentiment.label).glow}`
                }}>
                  <span style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    backgroundColor: getSentimentDetails(sentiment.label).color,
                    boxShadow: `0 0 8px ${getSentimentDetails(sentiment.label).color}`
                  }} />
                  {sentiment.label}
                </span>
              )}
            </div>
          </div>

          {/* Volatility Index (VIX) Card */}
          <div className="card card-glow" style={{
            position: 'relative',
            padding: '24px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '24px',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: '-50px',
              right: '-50px',
              width: '120px',
              height: '120px',
              borderRadius: '50%',
              background: loading ? 'rgba(255, 255, 255, 0.03)' : getVixDetails(vixScore).glow,
              filter: 'blur(35px)',
              pointerEvents: 'none',
              transition: 'background 0.5s ease'
            }} />
            
            <div style={{ position: 'relative', width: '80px', height: '80px', flexShrink: 0 }}>
              <svg width="80" height="80" viewBox="0 0 80 80">
                <circle
                  cx="40"
                  cy="40"
                  r="34"
                  fill="none"
                  stroke="var(--bg-elevated)"
                  strokeWidth="6"
                />
                <circle
                  cx="40"
                  cy="40"
                  r="34"
                  fill="none"
                  stroke={loading ? 'var(--border-subtle)' : getVixDetails(vixScore).color}
                  strokeWidth="6"
                  strokeDasharray="213.6"
                  strokeDashoffset={loading ? 213.6 : 213.6 * (1 - Math.min(vixScore, 50) / 50)}
                  strokeLinecap="round"
                  transform="rotate(-90 40 40)"
                  style={{
                    transition: 'stroke-dashoffset 0.8s ease, stroke 0.5s ease',
                    filter: loading ? 'none' : `drop-shadow(0 0 4px ${getVixDetails(vixScore).color})`
                  }}
                />
              </svg>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '80px',
                height: '80px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1
              }}>
                <span style={{ fontSize: '15px', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {loading ? '...' : vixScore.toFixed(2)}
                </span>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: '-2px', fontWeight: 700 }}>
                  {mode === 'equity' ? 'INDIA VIX' : 'VIX INDEX'}
                </div>
              </div>
            </div>

            <div style={{ flex: 1, zIndex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '18px' }}>⚡</span>
                <h3 style={{ fontSize: '16px', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>
                  {loading ? 'Fetching volatility...' : getVixDetails(vixScore).rating}
                </h3>
              </div>
              <p className="text-sm text-muted" style={{ lineHeight: '1.4', margin: 0 }}>
                {loading ? 'Reading volatility index data from market data feeds...' : getVixDetails(vixScore).desc}
              </p>
              {!loading && (
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px',
                  marginTop: '10px',
                  padding: '3px 10px',
                  borderRadius: '99px',
                  fontSize: '11px',
                  fontWeight: 800,
                  backgroundColor: `${getVixDetails(vixScore).color}15`,
                  color: getVixDetails(vixScore).color,
                  border: `1px solid ${getVixDetails(vixScore).color}30`
                }}>
                  Status: {getVixDetails(vixScore).rating}
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-screen" style={{ minHeight: '300px' }}>
          <span className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
          <span>Fetching fresh market news...</span>
        </div>
      )}

      {!loading && !error && filteredNews.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '48px 24px', color: 'var(--text-muted)' }}>
          <div>📰</div>
          <h3 style={{ fontWeight: 600, marginTop: 12 }}>No articles found</h3>
          <p className="text-sm">Try resetting your filter or refreshing the feed.</p>
        </div>
      )}

      {!loading && !error && filteredNews.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
          
          {/* 1. Featured Top Story */}
          <div>
            <h2 style={{ fontSize: '13px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '12px' }}>
              🔥 Top Headline
            </h2>
            {(() => {
              const item = filteredNews[0];
              const srcDetails = getSourceStyles(item.source || '');
              return (
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="card card-hover"
                  style={{
                    display: 'flex',
                    flexDirection: 'row',
                    alignItems: 'center',
                    gap: '24px',
                    textDecoration: 'none',
                    color: 'inherit',
                    padding: '24px',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%)',
                    cursor: 'pointer',
                    minHeight: '130px'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', flexWrap: 'wrap', width: '100%' }}>
                      <span style={{
                        padding: '4px 10px',
                        borderRadius: '99px',
                        fontSize: '11px',
                        fontWeight: 800,
                        backgroundColor: srcDetails.bg,
                        color: srcDetails.color,
                        border: `1px solid ${srcDetails.color}30`
                      }}>
                        {srcDetails.label}
                      </span>
                      {(() => {
                        const rLabel = item.region || 'Global';
                        const rColor = rLabel === 'Indian' ? '#f97316' : '#3b82f6';
                        const rBg = rLabel === 'Indian' ? 'rgba(249, 115, 22, 0.1)' : 'rgba(59, 130, 246, 0.1)';
                        const rBorder = rLabel === 'Indian' ? 'rgba(249, 115, 22, 0.25)' : 'rgba(59, 130, 246, 0.25)';
                        return (
                          <span style={{
                            padding: '4px 10px',
                            borderRadius: '99px',
                            fontSize: '11px',
                            fontWeight: 800,
                            backgroundColor: rBg,
                            color: rColor,
                            border: `1px solid ${rBorder}`
                          }}>
                            {rLabel}
                          </span>
                        );
                      })()}
                      {(() => {
                        const method = item.method || 'Default Source';
                        const conf = Math.round((item.confidence || 1.0) * 100);
                        const text = method === 'Rule Override' ? 'Rule Override' : `ML (${conf}%)`;
                        const color = method === 'Rule Override' ? '#10b981' : '#a855f7';
                        const bg = method === 'Rule Override' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(168, 85, 247, 0.08)';
                        const border = method === 'Rule Override' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(168, 85, 247, 0.2)';
                        return (
                          <span style={{
                            padding: '4px 10px',
                            borderRadius: '99px',
                            fontSize: '11px',
                            fontWeight: 700,
                            backgroundColor: bg,
                            color: color,
                            border: `1px solid ${border}`
                          }}>
                            {text}
                          </span>
                        );
                      })()}
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                        {item.pub_date ? new Date(item.pub_date).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : ''}
                      </span>
                    </div>
                    <h2 style={{ fontSize: '18px', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '8px', lineHeight: '1.4' }}>
                      {item.title}
                    </h2>
                    <p className="text-sm text-muted" style={{ lineHeight: '1.6', margin: 0, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {item.description}
                    </p>
                  </div>
                  <div style={{ fontSize: '24px', color: 'var(--accent-blue-bright)', paddingRight: '12px' }}>
                    ➔
                  </div>
                </a>
              )
            })()}
          </div>

          {/* 2. Premium Media Carousel */}
          {filteredNews.length > 1 && (
            <div>
              <h2 style={{ fontSize: '13px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                📰 Premium Media Stream
              </h2>
              <div style={{ display: 'flex', gap: '16px', overflowX: 'auto', paddingBottom: '8px', scrollbarWidth: 'thin' }}>
                {filteredNews.slice(1, 7).map((item, idx) => {
                  const srcDetails = getSourceStyles(item.source || '');
                  return (
                    <a
                      key={idx}
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="card card-hover"
                      style={{
                        flex: '0 0 280px',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        textDecoration: 'none',
                        color: 'inherit',
                        padding: '16px',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: '10px',
                        minHeight: '180px',
                        cursor: 'pointer'
                      }}
                    >
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap', gap: '4px' }}>
                          <span style={{
                            padding: '2px 8px',
                            borderRadius: '99px',
                            fontSize: '10px',
                            fontWeight: 800,
                            backgroundColor: srcDetails.bg,
                            color: srcDetails.color,
                            border: `1px solid ${srcDetails.color}20`
                          }}>
                            {srcDetails.label}
                          </span>
                          {(() => {
                            const rLabel = item.region || 'Global';
                            const method = item.method || 'Default Source';
                            const conf = Math.round((item.confidence || 1.0) * 100);
                            const text = method === 'Rule Override' ? 'Rule' : `ML (${conf}%)`;
                            const rColor = rLabel === 'Indian' ? '#f97316' : '#3b82f6';
                            return (
                              <span style={{ fontSize: '9px', fontWeight: 750, color: rColor }}>
                                {rLabel} • {text}
                              </span>
                            );
                          })()}
                        </div>
                        <h4 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--text-primary)', lineHeight: '1.4', display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                          {item.title}
                        </h4>
                      </div>
                      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--accent-blue-bright)', marginTop: '12px' }}>
                        Read Article →
                      </div>
                    </a>
                  )
                })}
              </div>
            </div>
          )}

          {/* 3. General Stream List */}
          {filteredNews.length > 7 && (
            <div>
              <h2 style={{ fontSize: '13px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                🗂️ Market Feed
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {filteredNews.slice(7).map((item, idx) => {
                  const srcDetails = getSourceStyles(item.source || '');
                  return (
                    <a
                      key={idx}
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="card card-hover"
                      style={{
                        display: 'flex',
                        flexDirection: 'row',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        textDecoration: 'none',
                        color: 'inherit',
                        padding: '16px 20px',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: '8px',
                        cursor: 'pointer'
                      }}
                    >
                      <div style={{ flex: 1, paddingRight: '16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px', flexWrap: 'wrap' }}>
                          <span style={{
                            padding: '2px 8px',
                            borderRadius: '99px',
                            fontSize: '10px',
                            fontWeight: 800,
                            backgroundColor: srcDetails.bg,
                            color: srcDetails.color,
                          }}>
                            {srcDetails.label}
                          </span>
                          {(() => {
                            const rLabel = item.region || 'Global';
                            const rColor = rLabel === 'Indian' ? '#f97316' : '#3b82f6';
                            const rBg = rLabel === 'Indian' ? 'rgba(249, 115, 22, 0.1)' : 'rgba(59, 130, 246, 0.1)';
                            const rBorder = rLabel === 'Indian' ? 'rgba(249, 115, 22, 0.25)' : 'rgba(59, 130, 246, 0.25)';
                            return (
                              <span style={{
                                padding: '2px 8px',
                                borderRadius: '99px',
                                fontSize: '10px',
                                fontWeight: 800,
                                backgroundColor: rBg,
                                color: rColor,
                                border: `1px solid ${rBorder}`
                              }}>
                                {rLabel}
                              </span>
                            );
                          })()}
                          {(() => {
                            const method = item.method || 'Default Source';
                            const conf = Math.round((item.confidence || 1.0) * 100);
                            const text = method === 'Rule Override' ? 'Rule Override' : `ML (${conf}%)`;
                            const color = method === 'Rule Override' ? '#10b981' : '#a855f7';
                            const bg = method === 'Rule Override' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(168, 85, 247, 0.08)';
                            const border = method === 'Rule Override' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(168, 85, 247, 0.2)';
                            return (
                              <span style={{
                                padding: '2px 8px',
                                borderRadius: '99px',
                                fontSize: '10px',
                                fontWeight: 700,
                                backgroundColor: bg,
                                color: color,
                                border: `1px solid ${border}`
                              }}>
                                {text}
                              </span>
                            );
                          })()}
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                            {item.pub_date ? new Date(item.pub_date).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : ''}
                          </span>
                        </div>
                        <h4 style={{ fontSize: '14px', fontWeight: 700, margin: 0, color: 'var(--text-primary)', lineHeight: '1.4' }}>
                          {item.title}
                        </h4>
                        <p className="text-xs text-muted" style={{ margin: '4px 0 0 0', display: '-webkit-box', WebkitLineClamp: 1, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                          {item.description}
                        </p>
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>➔</div>
                    </a>
                  )
                })}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}
