import { useState, useEffect } from 'react'

const TICKERS = [
  { symbol: 'NSE:INFY',     base: 1510 },
  { symbol: 'NSE:RELIANCE', base: 2430 },
  { symbol: 'NSE:TCS',      base: 3890 },
  { symbol: 'NSE:HDFC',     base: 1720 },
  { symbol: 'NSE:WIPRO',    base: 480  },
  { symbol: 'NSE:BAJFINANCE', base: 6400 },
  { symbol: 'NSE:TATAMOTORS', base: 920 },
  { symbol: 'NSE:ICICIBANK',  base: 1200 },
]

function randomDelta(base: number) {
  return (Math.random() - 0.48) * base * 0.012
}

export default function TickerBar() {
  const [prices, setPrices] = useState(() =>
    TICKERS.map(t => ({ ...t, price: t.base, change: 0, changePct: 0 }))
  )

  useEffect(() => {
    const interval = setInterval(() => {
      setPrices(prev =>
        prev.map(t => {
          const delta  = randomDelta(t.price)
          const newPrice = Math.max(1, t.price + delta)
          const change    = newPrice - t.base
          const changePct = (change / t.base) * 100
          return { ...t, price: newPrice, change, changePct }
        })
      )
    }, 1800)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="ticker-bar">
      {prices.map(t => (
        <div key={t.symbol} className="ticker-item">
          <span className="ticker-symbol">{t.symbol.replace('NSE:', '')}</span>
          <span className="ticker-price">₹{t.price.toFixed(2)}</span>
          <span className={`ticker-change ${t.changePct >= 0 ? 'up' : 'down'}`}>
            {t.changePct >= 0 ? '▲' : '▼'} {Math.abs(t.changePct).toFixed(2)}%
          </span>
        </div>
      ))}
    </div>
  )
}
