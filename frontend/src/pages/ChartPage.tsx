import { useState, useEffect, useRef, useCallback } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  createSeriesMarkers,
} from 'lightweight-charts'
import type { IChartApi, UTCTimestamp, ISeriesMarkersPluginApi } from 'lightweight-charts'
import api from '../api'
import { TickerSearch } from '../components/TickerSearch'

// ─── Types ────────────────────────────────────────────────────────────────────
interface Candle {
  time:    UTCTimestamp
  open:    number
  high:    number
  low:     number
  close:   number
  volume:  number
  ema9?:   number | null
  ema21?:  number | null
  ema50?:  number | null
  ema200?: number | null
  rsi14?:  number | null
}
interface WatchItem { symbol: string; name: string; ltp: number | null; prev: number | null; sector?: string }

// ─── Constants ────────────────────────────────────────────────────────────────
const TIMEFRAMES = [
  { label:'1m',  kite:'1minute',  mongo:'1m',  days:7    },
  { label:'5m',  kite:'5minute',  mongo:'5m',  days:14   },
  { label:'15m', kite:'15minute', mongo:'15m', days:30   },
  { label:'1h',  kite:'60minute', mongo:'1h',  days:90   },
  { label:'1D',  kite:'day',      mongo:'1D',  days:365  },
  { label:'1W',  kite:'week',     mongo:'1W',  days:1825 },
  { label:'1M',  kite:'month',    mongo:'1M',  days:3650 },
]

const INDICATORS = ['EMA9','EMA21','EMA50','EMA200','Volume','RSI']

const C = {
  bg:'#18171c', bgPanel:'#201f24', bgCard:'#201f24',
  border:'#383d47', text:'#bcc6e5', muted:'#6b868e',
  up:'#26a69a', down:'#ef5350', blue:'#516565',
  ema9:'#f59e0b', ema21:'#8b5cf6', ema50:'#06b6d4', ema200:'#f43f5e',
  vol:'rgba(81, 101, 101, 0.35)', rsi:'#fbbf24',
}

// SESSION_KEY is now derived dynamically per mode inside the component

// ─── Watchlist ────────────────────────────────────────────────────────────────
function Watchlist({ items, active, onSelect, mode }: {
  items:WatchItem[]; active:string; onSelect:(s:string)=>void; mode:string
}) {
  const [query, setQuery] = useState('')
  
  const filteredItems = items.filter(item => 
    item.symbol.toLowerCase().includes(query.toLowerCase()) || 
    (item.name && item.name.toLowerCase().includes(query.toLowerCase())) ||
    (item.sector && item.sector.toLowerCase().includes(query.toLowerCase()))
  )

  const getWatchlistHeader = () => {
    if (mode === 'crypto') return 'Crypto Watchlist'
    if (mode === 'forex') return 'Forex Watchlist'
    return 'Equity Watchlist'
  }

  return (
    <aside style={{
      width:240, minWidth:240, background:C.bgPanel,
      borderRight:`1px solid ${C.border}`,
      display:'flex', flexDirection:'column', overflow:'hidden',
    }}>
      <div style={{
        padding:'10px 12px 8px', fontSize:10, fontWeight:700,
        color:C.muted, textTransform:'uppercase', letterSpacing:'0.08em',
        borderBottom:`1px solid ${C.border}`,
      }}>
        {getWatchlistHeader()}
      </div>
      <div style={{ padding: '8px 12px', borderBottom:`1px solid ${C.border}` }}>
        <input 
          type="text" 
          placeholder="Search Local Database..." 
          value={query}
          onChange={e => setQuery(e.target.value)}
          style={{
            width: '100%', padding: '6px 10px', background: C.bg, color: C.text,
            border: `1px solid ${C.border}`, borderRadius: 4, fontSize: 11,
            outline: 'none'
          }}
        />
      </div>
      <div style={{ flex:1, overflowY:'auto' }}>
        {filteredItems.map(({ symbol, name, ltp, prev, sector }) => {
          const cleanName = name ? name.replace('NSE:', '') : symbol
            .replace('NSE:NIFTY 50','NIFTY')
            .replace('NSE:NIFTY BANK','BNKNIFTY')
            .replace('NSE:','')
          const isActive = symbol === active
          const up   = ltp !== null && prev !== null ? ltp >= prev : true
          const pct  = ltp && prev && prev !== 0 ? ((ltp-prev)/prev)*100 : null
          
          return (
            <div key={symbol} onClick={() => onSelect(symbol)} style={{
              display:'flex', alignItems:'center', justifyContent:'space-between',
              padding:'9px 12px', cursor:'pointer',
              background: isActive ? 'rgba(59,130,246,0.10)' : 'transparent',
              borderLeft:`3px solid ${isActive ? C.blue : 'transparent'}`,
              transition:'background 0.12s',
              borderBottom:`1px solid rgba(99,179,237,0.03)`
            }}>
              <div style={{ flex: 1, minWidth: 0, paddingRight: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontWeight:600, fontSize:12,
                    color: isActive ? '#60a5fa' : C.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {cleanName}
                  </span>
                  {(symbol === 'NSE:TMCV' || symbol === 'NSE:TMPV') && (
                    <span style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b', fontSize: 8, padding: '1px 4px', borderRadius: 4, fontWeight: 700 }}>
                      DEMERGER
                    </span>
                  )}
                </div>
                <div style={{ fontSize:9, color:C.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {sector || (symbol.includes('NIFTY') ? 'Index' : 'Equity')}
                </div>
              </div>
              <div style={{ textAlign:'right', flexShrink: 0 }}>
                <div style={{ fontSize:12, fontWeight:600,
                  color: ltp != null ? (up ? C.up : C.down) : C.muted }}>
                  {ltp != null ? `${mode === 'equity' ? '₹' : '$'}${ltp.toFixed(ltp<100?3:2)}` : '—'}
                </div>
                {pct != null && (
                  <div style={{ fontSize:10, color: up ? C.up : C.down }}>
                    {up?'▲':'▼'} {Math.abs(pct).toFixed(2)}%
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </aside>
  )
}

// ─── OrderBook Visualizer Component ──────────────────────────────────────────
function OrderBookVisualizer({ orderbook }: { orderbook: any }) {
  const bids = orderbook?.bids || []
  const asks = orderbook?.asks || []
  
  const bestBid = bids.length > 0 ? bids[0].price : 0
  const bestAsk = asks.length > 0 ? asks[0].price : 0
  const midPrice = (bestBid > 0 && bestAsk > 0) ? ((bestBid + bestAsk) / 2) : 0
  
  const maxBidQty = Math.max(...bids.map((b: any) => b.qty), 1)
  const maxAskQty = Math.max(...asks.map((a: any) => a.qty), 1)
  const maxQty = Math.max(maxBidQty, maxAskQty)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '0 4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
        <span style={{ color: C.muted }}>Mid Price:</span>
        <span style={{ fontWeight: 600, color: C.text }}>
          {midPrice > 0 ? `₹${midPrice.toFixed(2)}` : 'N/A'}
        </span>
      </div>
      
      {/* Asks (Sells) - Render reversed so lowest ask is at bottom */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', marginTop: '4px' }}>
        {asks.slice().reverse().map((ask: any, i: number) => {
           const pct = (ask.qty / maxQty) * 100
           return (
             <div key={`ask-${i}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, position: 'relative', height: 18, alignItems: 'center' }}>
               <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: `${pct}%`, background: 'rgba(239,83,80,0.15)', zIndex: 0, borderRadius: 2 }} />
               <span style={{ color: C.down, zIndex: 1, paddingLeft: 4, fontFamily: 'monospace' }}>{ask.price.toFixed(2)}</span>
               <span style={{ color: C.text, zIndex: 1, paddingRight: 4, fontFamily: 'monospace' }}>{ask.qty}</span>
             </div>
           )
        })}
      </div>
      
      {/* Spread / Mid separator */}
      <div style={{ textAlign: 'center', fontSize: 10, color: C.muted, margin: '6px 0', borderTop: `1px dashed ${C.border}`, borderBottom: `1px dashed ${C.border}`, padding: '4px 0', background: 'rgba(255,255,255,0.02)' }}>
        Spread: { (bestAsk > 0 && bestBid > 0) ? (bestAsk - bestBid).toFixed(2) : 'N/A' }
      </div>
      
      {/* Bids (Buys) - Highest bid at top */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {bids.map((bid: any, i: number) => {
           const pct = (bid.qty / maxQty) * 100
           return (
             <div key={`bid-${i}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, position: 'relative', height: 18, alignItems: 'center' }}>
               <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${pct}%`, background: 'rgba(38,166,154,0.15)', zIndex: 0, borderRadius: 2 }} />
               <span style={{ color: C.text, zIndex: 1, paddingLeft: 4, fontFamily: 'monospace' }}>{bid.qty}</span>
               <span style={{ color: C.up, zIndex: 1, paddingRight: 4, fontFamily: 'monospace' }}>{bid.price.toFixed(2)}</span>
             </div>
           )
        })}
      </div>
    </div>
  )
}

// ─── Main Chart ───────────────────────────────────────────────────────────────
export default function ChartPage({ mode }: { mode: string }) {
  const mainRef   = useRef<HTMLDivElement>(null)
  const rsiRef    = useRef<HTMLDivElement>(null)

  // chart and series instances stored in refs
  const mainChart = useRef<IChartApi | null>(null)
  const rsiChart  = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<any>(null)
  // LWC v5: markers are managed via createSeriesMarkers() plugin, not series.setMarkers()
  const markersPluginRef = useRef<ISeriesMarkersPluginApi<any> | null>(null)
  const stopLossLineRef = useRef<any>(null)
  const takeProfitLineRef = useRef<any>(null)

  const sessionKey = `lwcChartState_${mode}`

  const [symbol, setSymbol] = useState(() => {
    try { 
      const parsed = JSON.parse(sessionStorage.getItem(sessionKey)||'{}')
      return (parsed && typeof parsed.symbol === 'string') ? parsed.symbol : 'NSE:INFY'
    }
    catch { return 'NSE:INFY' }
  })

  // Auto-switch symbol when mode changes
  useEffect(() => {
    const isCrypto = symbol.endsWith("USDT") || symbol.startsWith("P-") || symbol.startsWith("C-") || symbol.startsWith("F-")
    const isForex = symbol.endsWith("=X")
    
    if (mode === 'crypto' && !isCrypto) {
      setSymbol('BTCUSDT')
      setSearch('BTCUSDT')
    } else if (mode === 'forex' && !isForex) {
      setSymbol('EURUSD=X')
      setSearch('EURUSD=X')
    } else if (mode === 'equity' && (isCrypto || isForex)) {
      setSymbol('NSE:INFY')
      setSearch('NSE:INFY')
    }
  }, [mode])

  const [tfIdx, setTfIdx] = useState(() => {
    try { 
      const parsed = JSON.parse(sessionStorage.getItem(sessionKey)||'{}')
      const idx = parsed?.tfIdx
      return (typeof idx === 'number' && idx >= 0 && idx < TIMEFRAMES.length) ? idx : 5
    }
    catch { return 5 }
  })
  const [inds, setInds] = useState<Record<string,boolean>>({
    EMA9:true, EMA21:true, EMA50:true, EMA200:false, Volume:true, RSI:false,
  })
  const [candles,   setCandles]   = useState<Candle[]>([])
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState('')
  const [lastC,     setLastC]     = useState<Candle|null>(null)
  const [watchItems,setWatchItems] = useState<WatchItem[]>([])
  const [search, setSearch] = useState('')

  // ─── Execution Panel States ───────────────────────────────────────────────
  const [action, setAction] = useState<'BUY' | 'SELL'>('BUY')
  const [variety, setVariety] = useState<'regular' | 'intraday' | 'gtt'>('regular')
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('MARKET')
  const [qty, setQty] = useState<number>(10)
  const [price, setPrice] = useState<string>('0')
  const [triggerPrice, setTriggerPrice] = useState<string>('0')
  
  // Options State
  const [isOption, setIsOption] = useState<boolean>(false)
  const [optionType, setOptionType] = useState<'CE' | 'PE'>('CE')
  const [strikePrice, setStrikePrice] = useState<string>('')
  const [expiry, setExpiry] = useState<string>('26JUN') // Default next standard expiry monthly

  const [confirmTrade, setConfirmTrade] = useState<boolean>(false)
  const [tradeLoading, setTradeLoading] = useState<boolean>(false)
  const [tradeResult, setTradeResult] = useState<{ status: string; order_id?: string; message: string; error?: string } | null>(null)

  const [, setPredictions] = useState<any[]>([])
  const [predictLoading, setPredictLoading] = useState<boolean>(false)
  const [modelOverlayActive, setModelOverlayActive] = useState<boolean>(false)
  const [nextDayPred, setNextDayPred] = useState<any>(null)
  const [selectedModel, setSelectedModel] = useState<string>('tft')
  const [modelBacktestResults, setModelBacktestResults] = useState<any | null>(null)

  // ─── Pine Script Backtester States ─────────────────────────────────────────
  const [pineScript, setPineScript] = useState<string>(`//@version=5
strategy("EMA Crossover", overlay=true)

fastEMA = ta.ema(close, 9)
slowEMA = ta.ema(close, 21)

longCondition = ta.crossover(fastEMA, slowEMA)
shortCondition = ta.crossunder(fastEMA, slowEMA)

if (longCondition)
    strategy.entry("Long", strategy.long)

if (shortCondition)
    strategy.close("Long")`)
  const [pineBacktestLoading, setPineBacktestLoading] = useState<boolean>(false)
  const [pineResults, setPineResults] = useState<{
    trades: any[]
    summary: {
      total_trades: number
      net_profit: number
      net_profit_pct: number
      win_rate: number
      max_drawdown: number
      profit_factor: number
      avg_trade: number
      sharpe: number
      reward_risk_ratio?: number
    }
    strategy_type: string
  } | null>(null)
  const [pineError, setPineError] = useState<string>('')
  const [showPineTester, setShowPineTester] = useState<boolean>(false)

  // ─── Order Book States ───────────────────────────────────────────────────
  const [orderbook, setOrderbook] = useState<{bids:any[], asks:any[], timestamp:string} | null>(null)
  const [orderbookLoading, setOrderbookLoading] = useState<boolean>(false)

  const fetchOrderbook = useCallback(async (sym: string) => {
    setOrderbookLoading(true)
    try {
      let exch = 'NSE'
      let cleanSym = sym
      if (sym.includes(':')) {
        const parts = sym.split(':')
        exch = parts[0]
        cleanSym = parts[1]
      } else if (sym.endsWith('=X')) {
        exch = 'forex'
        cleanSym = sym
      } else if (sym.endsWith('USDT') || sym.endsWith('USD')) {
        exch = 'crypto'
        cleanSym = sym
      }
      
      const res = await api.get(`/orderbook?symbol=${cleanSym}&exchange=${exch}&limit=1`)
      if (res.data.snapshots && res.data.snapshots.length > 0) {
        setOrderbook(res.data.snapshots[0])
      } else {
        setOrderbook(null)
      }
    } catch (e) {
      console.error("Orderbook fetch failed", e)
      setOrderbook(null)
    } finally {
      setOrderbookLoading(false)
    }
  }, [])


  useEffect(() => {
    sessionStorage.setItem(sessionKey, JSON.stringify({ symbol, tfIdx }))
  }, [symbol, tfIdx, sessionKey])

  // ── Fetch Tickers Classification ─────────────────────────────────────────────
  const fetchTickers = useCallback(async () => {
    try {
      const res = await api.get('/tickers')
      if (res.data.tickers) {
        const dbTickers: any[] = res.data.tickers
        setWatchItems(prev => {
          // Merge existing with new DB tickers
          const newItems = [...prev]
          for (const dbT of dbTickers) {
            const idx = newItems.findIndex(i => i.symbol === dbT.symbol)
            if (idx >= 0) {
              newItems[idx] = { ...newItems[idx], name: dbT.name, sector: dbT.sector }
            } else {
              newItems.push({ symbol: dbT.symbol, name: dbT.name, sector: dbT.sector, ltp: null, prev: null })
            }
          }
          return newItems
        })
      }
    } catch {}
  }, [])

  useEffect(() => { fetchTickers() }, [fetchTickers])

  // ── Fetch OHLCV ─────────────────────────────────────────────────────────────
  const fetchCandles = useCallback(async (sym:string, tf:typeof TIMEFRAMES[0]) => {
    setLoading(true); setError(''); setNextDayPred(null); setPredictions([])
    try {
      const res = await api.post('/ohlcv', { symbol:sym, interval:tf.kite, days:tf.days })
      if (res.data.error) throw new Error(res.data.error)
      setCandles(res.data.candles)
      if (res.data.candles.length) {
        const lastCandle = res.data.candles.at(-1)
        setLastC(lastCandle)
        setPrice(lastCandle.close.toFixed(2))
        setTriggerPrice((lastCandle.close * 0.99).toFixed(2)) // default trigger slightly below
      }
    } catch(e:any) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { 
    fetchCandles(symbol, TIMEFRAMES[tfIdx]) 
    fetchOrderbook(symbol)
  }, [symbol, tfIdx, fetchCandles, fetchOrderbook])

  // ── Auto-poll Orderbook / Live WebSocket ──────────────────────────────────
  useEffect(() => {
    let ws: WebSocket | null = null
    const cleanSym = symbol.includes(':') ? symbol.split(':')[1] : symbol

    if (symbol.endsWith('USDT') || symbol.endsWith('USD')) {
      // Connect directly to Binance WebSocket for instant live orderbook depth!
      const binanceSym = cleanSym.toLowerCase().replace('usd', 'usdt')
      const wsUrl = `wss://stream.binance.com:9443/ws/${binanceSym}@depth20@100ms`
      console.log(`Connecting to Binance WS: ${wsUrl}`)
      ws = new WebSocket(wsUrl)
      
      ws.onopen = () => console.log(`Binance WS Connected: ${binanceSym}`)
      ws.onerror = (e) => console.error(`Binance WS Error:`, e)
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.bids && data.asks) {
           const bids = data.bids.map((b:any) => ({ price: parseFloat(b[0]), qty: parseFloat(b[1]) }))
           const asks = data.asks.map((a:any) => ({ price: parseFloat(a[0]), qty: parseFloat(a[1]) }))
           setOrderbook({ bids, asks, timestamp: new Date().toISOString() })
           setOrderbookLoading(false)
        }
      }
      return () => {
        console.log(`Closing Binance WS for ${binanceSym}`)
        ws?.close()
      }
    } else {
      // For non-crypto (NSE), fallback to 1.5s DB polling (or backend WS relay)
      const id = setInterval(() => fetchOrderbook(symbol), 1500)
      return () => clearInterval(id)
    }
  }, [symbol, fetchOrderbook])

  // ── Fetch watchlist ─────────────────────────────────────────────────────────
  const fetchWatchlist = useCallback(async () => {
    try {
      const res = await api.get(`/watchlist?mode=${mode}`)
      if (res.data && res.data.watchlist) {
        setWatchItems(prev => {
          const prevLtpMap = new Map<string, number | null>()
          prev.forEach(x => prevLtpMap.set(x.symbol, x.ltp))
          
          return res.data.watchlist.map((item: any) => ({
            symbol: item.symbol,
            name: item.name || item.symbol,
            ltp: item.ltp ?? prevLtpMap.get(item.symbol) ?? null,
            prev: prevLtpMap.get(item.symbol) ?? item.ltp ?? null,
            sector: item.sector
          }))
        })
      }
    } catch {}
  }, [mode])

  useEffect(() => {
    fetchWatchlist()
    const id = setInterval(fetchWatchlist, 30_000)
    return () => clearInterval(id)
  }, [fetchWatchlist])

  // ── Real-time chart updates via WebSocket ──────────────────────────────────
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765")

    ws.onopen = () => {
      console.log("WebSocket chart feed connected ✅")
    }

    ws.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data)
        if (tick.symbol.toUpperCase() !== symbol.toUpperCase()) return

        const price = parseFloat(tick.price)
        const timestamp = tick.timestamp
        const tickTime = new Date(timestamp * (timestamp < 1e11 ? 1000 : 1))
        
        // Round to start of minute for 1-minute chart timeframe
        const minuteTime = new Date(
          tickTime.getFullYear(),
          tickTime.getMonth(),
          tickTime.getDate(),
          tickTime.getHours(),
          tickTime.getMinutes(),
          0,
          0
        )
        const minuteEpoch = Math.floor(minuteTime.getTime() / 1000) as UTCTimestamp

        setCandles(prev => {
          if (!prev.length) return prev
          const newCandles = [...prev]
          const last = newCandles[newCandles.length - 1]

          if (last.time === minuteEpoch) {
            // Update the last candle
            const updated = {
              ...last,
              close: price,
              high: Math.max(last.high, price),
              low: Math.min(last.low, price),
            }
            newCandles[newCandles.length - 1] = updated
            setLastC(updated)
            setPrice(price.toFixed(2))
            return newCandles
          } else if (minuteEpoch > last.time) {
            // Append a new candle
            const newCandle: Candle = {
              time: minuteEpoch,
              open: price,
              high: price,
              low: price,
              close: price,
              volume: parseFloat(tick.volume || 1),
            }
            newCandles.push(newCandle)
            setLastC(newCandle)
            setPrice(price.toFixed(2))
            return newCandles
          }
          return prev
        })
      } catch (err) {
        console.error("Error handling tick payload:", err)
      }
    }

    return () => {
      ws.close()
    }
  }, [symbol])


  // ── Build charts whenever candles or indicator flags change ─────────────────
  useEffect(() => {
    if (!mainRef.current || !candles.length) return

    // ── Destroy old charts ─────────────────────────────────────────────────
    mainChart.current?.remove(); mainChart.current = null
    rsiChart.current?.remove();  rsiChart.current  = null

    const chartOpts = (el:HTMLElement, height?:number) => ({
      layout: {
        background: { type: ColorType.Solid as const, color: C.bg },
        textColor: C.text,
      },
      grid: { vertLines:{ color:C.border }, horzLines:{ color:C.border } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: C.border },
      timeScale: { borderColor:C.border, timeVisible:true, secondsVisible:false },
      width:  el.clientWidth,
      height: height ?? el.clientHeight,
    })

    // ── Main chart ─────────────────────────────────────────────────────────
    const mc = createChart(mainRef.current, {
      ...chartOpts(mainRef.current),
      rightPriceScale: {
        borderColor: C.border,
        scaleMargins: { top:0.07, bottom: inds.Volume ? 0.22 : 0.05 },
      },
    })
    mainChart.current = mc

    // Candlestick
    const cs = mc.addSeries(CandlestickSeries, {
      upColor:COLORS.up, downColor:COLORS.down,
      borderUpColor:COLORS.up, borderDownColor:COLORS.down,
      wickUpColor:COLORS.up, wickDownColor:COLORS.down,
    } as any)
    cs.setData(candles.map(c=>({ time:c.time, open:c.open, high:c.high, low:c.low, close:c.close })))
    candleSeriesRef.current = cs
    // Destroy any previous markers plugin when chart rebuilds
    markersPluginRef.current = null

    // Volume
    if (inds.Volume) {
      const vs = mc.addSeries(HistogramSeries, {
        priceFormat: { type:'volume' as const },
        priceScaleId:'vol',
        lastValueVisible:false,
        priceLineVisible:false,
      } as any)
      mc.priceScale('vol').applyOptions({ scaleMargins:{ top:0.80, bottom:0 } })
      vs.setData(candles.map(c=>({
        time:c.time, value:c.volume,
        color: c.close >= c.open ? 'rgba(38,166,154,0.4)':'rgba(239,83,80,0.4)',
      })))
    }

    // EMAs
    const EMA_CFG: [string, keyof Candle, string][] = [
      ['EMA9','ema9',C.ema9],['EMA21','ema21',C.ema21],
      ['EMA50','ema50',C.ema50],['EMA200','ema200',C.ema200],
    ]
    for (const [name, key, color] of EMA_CFG) {
      if (!inds[name]) continue
      const es = mc.addSeries(LineSeries, {
        color, lineWidth:1 as const,
        priceLineVisible:false, lastValueVisible:false, crosshairMarkerVisible:false,
      } as any)
      es.setData(
        candles.filter(c=>c[key]!=null).map(c=>({ time:c.time, value:c[key] as number }))
      )
    }

    mc.timeScale().fitContent()

    // ResizeObserver for main chart
    const ro = new ResizeObserver(() => {
      if (mainRef.current && mainChart.current) {
        mainChart.current.applyOptions({
          width:  mainRef.current.clientWidth,
          height: mainRef.current.clientHeight,
        })
      }
    })
    if (mainRef.current) ro.observe(mainRef.current)

    // ── RSI pane ───────────────────────────────────────────────────────────
    if (inds.RSI && rsiRef.current) {
      const rc = createChart(rsiRef.current, {
        ...chartOpts(rsiRef.current, 120),
        rightPriceScale: { borderColor:C.border, scaleMargins:{ top:0.05, bottom:0.05 } },
      })
      rsiChart.current = rc

      const rsiLine = rc.addSeries(LineSeries, {
        color:C.rsi, lineWidth:1 as const,
        priceLineVisible:false, lastValueVisible:true, crosshairMarkerVisible:false,
      } as any)
      const rsiData = candles.filter(c=>c.rsi14!=null).map(c=>({ time:c.time, value:c.rsi14 as number }))
      rsiLine.setData(rsiData)

      if (rsiData.length) {
        const times = rsiData.map(d=>d.time)
        const ob = rc.addSeries(LineSeries, { color:'rgba(239,68,68,0.5)', lineWidth:1 as const, lineStyle:2, priceLineVisible:false, lastValueVisible:false } as any)
        const os = rc.addSeries(LineSeries, { color:'rgba(16,185,129,0.5)', lineWidth:1 as const, lineStyle:2, priceLineVisible:false, lastValueVisible:false } as any)
        ob.setData(times.map(t=>({ time:t, value:70 })))
        os.setData(times.map(t=>({ time:t, value:30 })))
      }
      rc.timeScale().fitContent()

      const roRsi = new ResizeObserver(() => {
        if (rsiRef.current && rsiChart.current) {
          rsiChart.current.applyOptions({ width: rsiRef.current.clientWidth })
        }
      })
      roRsi.observe(rsiRef.current)
    }

    return () => {
      ro.disconnect()
      mainChart.current?.remove(); mainChart.current = null
      rsiChart.current?.remove();  rsiChart.current  = null
    }
  }, [candles, inds])

  const tf   = TIMEFRAMES[tfIdx]
  const chg  = lastC && candles.length > 1 ? lastC.close - candles[0].open : null
  const chgP = chg && candles[0]?.open ? (chg/candles[0].open)*100 : null

  function jumpSymbol(raw:string) {
    const sym = raw.trim().toUpperCase().includes(':')
      ? raw.trim().toUpperCase() : `NSE:${raw.trim().toUpperCase()}`
    setSymbol(sym); setSearch('')
  }

  const indColor = (n:string) => ({
    EMA9:C.ema9, EMA21:C.ema21, EMA50:C.ema50, EMA200:C.ema200,
  }[n] ?? C.blue)

  // ─── Fetch and Apply ML Predictions (LWC v5 — createSeriesMarkers plugin) ────
  const runPredictions = async () => {
    setPredictLoading(true)
    try {
      // Ensure we request enough history for the interval (at least tf.days)
      const res = await api.post('/predict_model', { 
        symbol, 
        interval: tf.kite, 
        days: Math.max(120, tf.days),
        model: selectedModel
      })
      if (res.data.predictions && res.data.predictions.length) {
        const preds = res.data.predictions
        setPredictions(preds)
        setNextDayPred(preds.at(-1))
        setModelBacktestResults(res.data.backtest)

        // LWC v5: use createSeriesMarkers() instead of series.setMarkers()
        if (candleSeriesRef.current && mainChart.current) {
          const markers = preds
            .filter((p: any) => p.signal === 'buy' || p.signal === 'sell')
            .map((p: any) => ({
              time: p.time as UTCTimestamp,
              position: p.signal === 'buy' ? 'belowBar' as const : 'aboveBar' as const,
              color: p.signal === 'buy' ? C.up : C.down,
              shape: p.signal === 'buy' ? 'arrowUp' as const : 'arrowDown' as const,
              text: p.signal === 'buy'
                ? `B ₹${p.predicted_price}`
                : `S ₹${p.predicted_price}`,
            }))

          // Remove old markers plugin if it exists
          if (markersPluginRef.current) {
            try { markersPluginRef.current.setMarkers([]) } catch {}
          }
          // Create new markers plugin (LWC v5 API)
          markersPluginRef.current = createSeriesMarkers(
            candleSeriesRef.current,
            markers,
          ) as ISeriesMarkersPluginApi<any>

          // Manage Stop Loss and Take Profit lines
          if (stopLossLineRef.current) {
            try { candleSeriesRef.current.removePriceLine(stopLossLineRef.current) } catch {}
            stopLossLineRef.current = null
          }
          if (takeProfitLineRef.current) {
            try { candleSeriesRef.current.removePriceLine(takeProfitLineRef.current) } catch {}
            takeProfitLineRef.current = null
          }

          const latestPred = preds.at(-1)
          if (latestPred && (latestPred.signal === 'buy' || latestPred.signal === 'sell')) {
            if (latestPred.stop_loss) {
              stopLossLineRef.current = candleSeriesRef.current.createPriceLine({
                price: latestPred.stop_loss,
                color: C.down,
                lineWidth: 2,
                lineStyle: 1, // Dashed
                axisLabelVisible: true,
                title: 'Stop Loss (SL)',
              })
            }
            if (latestPred.take_profit) {
              takeProfitLineRef.current = candleSeriesRef.current.createPriceLine({
                price: latestPred.take_profit,
                color: C.up,
                lineWidth: 2,
                lineStyle: 1, // Dashed
                axisLabelVisible: true,
                title: 'Take Profit (TP)',
              })
            }
          }

          setModelOverlayActive(true)
        }
      } else {
        alert('Insufficient historical data (requires ≥60 bars) to run LSTM+DQN predictions.')
      }
    } catch (e: any) {
      alert(`Model prediction error: ${e.friendlyMessage || e.message}`)
    } finally {
      setPredictLoading(false)
    }
  }

  const clearModelOverlay = () => {
    if (markersPluginRef.current) {
      try { markersPluginRef.current.setMarkers([]) } catch {}
      markersPluginRef.current = null
    }
    if (candleSeriesRef.current) {
      if (stopLossLineRef.current) {
        try { candleSeriesRef.current.removePriceLine(stopLossLineRef.current) } catch {}
        stopLossLineRef.current = null
      }
      if (takeProfitLineRef.current) {
        try { candleSeriesRef.current.removePriceLine(takeProfitLineRef.current) } catch {}
        takeProfitLineRef.current = null
      }
    }
    setModelOverlayActive(false)
    setNextDayPred(null)
    setModelBacktestResults(null)
  }

  // ─── Trade Placement Handler ───────────────────────────────────────────────
  const handlePlaceOrder = async () => {
    if (!confirmTrade) {
      alert("Please confirm trade terms before submitting.")
      return
    }
    setTradeLoading(true)
    setTradeResult(null)
    try {
      const payload = {
        symbol: symbol.replace('NSE:', ''),
        exchange: 'NSE',
        transaction: action,
        quantity: qty,
        order_type: orderType,
        price: orderType === 'LIMIT' ? parseFloat(price) : 0,
        product: variety === 'regular' ? 'CNC' : variety === 'intraday' ? 'MIS' : 'NRML',
        variety,
        trigger_price: variety === 'gtt' ? parseFloat(triggerPrice) : 0,
        ltp: lastC ? lastC.close : 0,
        is_option: isOption,
        option_type: optionType,
        strike_price: isOption ? strikePrice : null,
        expiry: isOption ? expiry : null,
      }
      const res = await api.post('/place_order', payload)
      setTradeResult(res.data)
      setConfirmTrade(false)
    } catch (e: any) {
      const errMsg = e.response?.data?.error || e.friendlyMessage || e.message || 'Execution error';
      setTradeResult({
        status: 'FAILED',
        message: errMsg
      })
    } finally {
      setTradeLoading(false)
    }
  }

  // ─── Pine Script Strategy Tester Handler ────────────────────────────────────
  const handlePineBacktest = async () => {
    setPineBacktestLoading(true)
    setPineError('')
    setPineResults(null)
    try {
      const res = await api.post('/pine_backtest', {
        script: pineScript,
        symbol: symbol,
        interval: TIMEFRAMES[tfIdx].kite,
        days: TIMEFRAMES[tfIdx].days,
        initial_capital: 100000,
      })
      setPineResults(res.data)

      // Clear take profit/stop loss price lines of the ML prediction model if any
      if (stopLossLineRef.current && candleSeriesRef.current) {
        try { candleSeriesRef.current.removePriceLine(stopLossLineRef.current) } catch {}
        stopLossLineRef.current = null
      }
      if (takeProfitLineRef.current && candleSeriesRef.current) {
        try { candleSeriesRef.current.removePriceLine(takeProfitLineRef.current) } catch {}
        takeProfitLineRef.current = null
      }
      setModelOverlayActive(false)

      // Apply backtest markers to the TradingView chart
      if (candleSeriesRef.current && res.data.trades && res.data.trades.length) {
        const markers: any[] = [];
        res.data.trades.forEach((t: any) => {
          // Entry Marker
          markers.push({
            time: t.entry_time,
            position: t.type === 'long' ? 'belowBar' : 'aboveBar',
            color: t.type === 'long' ? C.up : C.down,
            shape: t.type === 'long' ? 'arrowUp' : 'arrowDown',
            text: t.type === 'long' ? `Buy ₹${t.entry_price}` : `Sell ${t.entry_price}`,
          });
          // Exit Marker
          markers.push({
            time: t.exit_time,
            position: t.type === 'long' ? 'aboveBar' : 'belowBar',
            color: t.type === 'long' ? C.down : C.up,
            shape: t.type === 'long' ? 'arrowDown' : 'arrowUp',
            text: `Exit ₹${t.exit_price}`,
          });
        });

        // Sort markers chronologically
        markers.sort((a: any, b: any) => {
          return new Date(a.time).getTime() - new Date(b.time).getTime();
        });

        // Format times (handle timestamps vs date strings)
        const formattedMarkers = markers.map((m: any) => {
          let markerTime = m.time;
          if (typeof markerTime === 'string' && /^\d+$/.test(markerTime)) {
            markerTime = parseInt(markerTime, 10) as UTCTimestamp;
          } else if (typeof markerTime === 'string' && markerTime.includes(' ')) {
            markerTime = markerTime.split(' ')[0];
          }
          return {
            ...m,
            time: markerTime
          };
        });

        // Remove old markers plugin
        if (markersPluginRef.current) {
          try { markersPluginRef.current.setMarkers([]) } catch {}
        }

        // Create new markers plugin for the TV chart
        markersPluginRef.current = createSeriesMarkers(
          candleSeriesRef.current,
          formattedMarkers,
        ) as ISeriesMarkersPluginApi<any>;
      } else {
        // Clear markers if no trades executed
        if (markersPluginRef.current) {
          try { markersPluginRef.current.setMarkers([]) } catch {}
          markersPluginRef.current = null;
        }
      }
    } catch (e: any) {
      setPineError(e.response?.data?.error || e.friendlyMessage || e.message || 'Backtest error')
    } finally {
      setPineBacktestLoading(false)
    }
  }


  return (
    <div style={{ display:'flex', height:'calc(100vh - 130px)', overflow:'hidden' }}>

      <Watchlist items={watchItems} active={symbol} onSelect={setSymbol} mode={mode} />

      <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:C.bg }}>

        {/* ── Toolbar ───────────────────────────────────────────────────── */}
        <div style={{
          display:'flex', alignItems:'center', gap:8, padding:'7px 14px',
          background:C.bgPanel, borderBottom:`1px solid ${C.border}`,
          flexWrap:'wrap', flexShrink:0, minHeight:48,
          position: 'relative', zIndex: 100
        }}>
          {/* Symbol + OHLC */}
          <div style={{
            background:C.bgCard, border:`1px solid rgba(99,179,237,0.3)`,
            borderRadius:8, padding:'4px 14px', fontWeight:800, fontSize:14, color:'#60a5fa',
          }}>
            {symbol.replace('NSE:','').replace('=X','')}
          </div>

          {lastC && (
            <div style={{ display:'flex', gap:10, fontSize:12, alignItems:'center' }}>
              <span style={{ fontWeight:700, color: lastC.close>=lastC.open ? C.up : C.down }}>
                {mode === 'equity' ? '₹' : '$'}{lastC.close.toFixed(2)}
              </span>
              {chg !== null && (
                <span style={{ color: chg>=0 ? C.up : C.down }}>
                  {chg>=0?'+':''}{chg.toFixed(2)} ({chgP?.toFixed(2)}%)
                </span>
              )}
              <span style={{ color:C.muted, fontSize:11 }}>
                O:{lastC.open.toFixed(2)} H:{lastC.high.toFixed(2)} L:{lastC.low.toFixed(2)}
              </span>
            </div>
          )}

          {/* Timeframes */}
          <div style={{ display:'flex', gap:3, marginLeft:4 }}>
            {TIMEFRAMES.map((t,i) => (
              <button key={t.label} onClick={()=>setTfIdx(i)} style={{
                background: tfIdx===i ? C.blue : C.bgCard,
                border:`1px solid ${tfIdx===i ? C.blue : C.border}`,
                borderRadius:5, color: tfIdx===i ? '#fff' : '#8ba7c7',
                padding:'4px 9px', fontSize:11, fontWeight:600, cursor:'pointer',
                transition:'all 0.12s',
              }}>{t.label}</button>
            ))}
          </div>

          <div style={{ width:1, height:20, background:C.border, flexShrink:0 }} />

          {/* Indicator toggles */}
          <div style={{ display:'flex', gap:3 }}>
            {INDICATORS.map(ind => (
              <button key={ind}
                onClick={() => setInds(p=>({ ...p, [ind]:!p[ind] }))}
                style={{
                  background: inds[ind] ? 'rgba(59,130,246,0.15)' : C.bgCard,
                  border:`1px solid ${inds[ind] ? indColor(ind) : C.border}`,
                  borderRadius:5,
                  color: inds[ind] ? indColor(ind) : C.muted,
                  padding:'4px 8px', fontSize:10, fontWeight:700,
                  cursor:'pointer', transition:'all 0.12s',
                }}>{ind}</button>
            ))}
          </div>

          {/* Search */}
          <div style={{ marginLeft:'auto', display:'flex', gap:6 }}>
            <TickerSearch
              value={search} onChange={setSearch}
              onSelect={jumpSymbol}
              placeholder="Symbol… (Enter)"
              style={{
                background:C.bgCard, border:`1px solid ${C.border}`,
                borderRadius:6, padding:'5px 10px', fontSize:12,
                color:C.text, outline:'none', width:210,
              }}
              mode={mode}
            />
            <button onClick={()=>fetchCandles(symbol, TIMEFRAMES[tfIdx])} style={{
              background:C.bgCard, border:`1px solid ${C.border}`,
              borderRadius:6, color:C.muted, padding:'5px 10px', cursor:'pointer',
            }}>🔄</button>
          </div>
        </div>

        {error && (
          <div style={{ background:'rgba(244,63,94,0.10)', borderBottom:`1px solid #f43f5e`,
            padding:'6px 14px', fontSize:12, color:'#f87171' }}>⚠️ {error}</div>
        )}

        {/* Chart area */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden', position:'relative' }}>
          {loading && (
            <div style={{
              position:'absolute', inset:0, zIndex:10,
              display:'flex', alignItems:'center', justifyContent:'center',
              background:'rgba(6,11,20,0.75)', gap:10,
            }}>
              <span className="spinner" style={{ width:26, height:26, borderWidth:3 }} />
              <span style={{ color:C.muted, fontSize:13 }}>Loading {symbol.replace('NSE:','')}…</span>
            </div>
          )}

          {/* Main candlestick chart */}
          <div ref={mainRef} style={{ flex:1, width:'100%' }} />

          {/* RSI pane */}
          {inds.RSI && (
            <div style={{ borderTop:`1px solid ${C.border}`, background:C.bg, flexShrink:0 }}>
              <div style={{ fontSize:10, color:C.muted, padding:'3px 10px', fontWeight:700 }}>RSI (14)</div>
              <div ref={rsiRef} style={{ width:'100%' }} />
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding:'3px 14px', fontSize:10, color:C.muted,
          borderTop:`1px solid ${C.border}`, background:C.bgPanel,
          display:'flex', gap:16, flexShrink:0,
        }}>
          <span>📊 Lightweight Charts v5 · TradingView</span>
          <span>{tf.label} · {candles.length} bars</span>
          {lastC?.ema9  && inds.EMA9   && <span style={{color:C.ema9  }}>EMA9:{lastC.ema9.toFixed(2)}</span>}
          {lastC?.ema21 && inds.EMA21  && <span style={{color:C.ema21 }}>EMA21:{lastC.ema21.toFixed(2)}</span>}
          {lastC?.ema50 && inds.EMA50  && <span style={{color:C.ema50 }}>EMA50:{lastC.ema50.toFixed(2)}</span>}
          {lastC?.rsi14 && inds.RSI    && <span style={{color:C.rsi   }}>RSI:{lastC.rsi14.toFixed(1)}</span>}
        </div>
      </div>

      {/* ─── Right Panel: Execution & AI Model Predictions ──────────────────── */}
      <aside style={{
        width: 330, minWidth: 330, background: C.bgPanel,
        borderLeft: `1px solid ${C.border}`,
        display: 'flex', flexDirection: 'column', overflowY: 'auto'
      }}>
        
        {/* Order Book Section */}
        <div style={{ padding: '16px', borderBottom: `1px solid ${C.border}` }}>
          <h3 style={{ fontSize: 13, fontWeight: 700, color: '#f59e0b', margin: '0 0 10px 0', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', justifyContent: 'space-between' }}>
            <span>📊 L2 Order Book</span>
            <button onClick={() => fetchOrderbook(symbol)} style={{ background:'transparent', border:'none', cursor:'pointer', color:C.muted }}>🔄</button>
          </h3>
          
          <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px' }}>
            {orderbookLoading ? (
              <div style={{ fontSize: 12, color: C.text, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Loading depth...
              </div>
            ) : orderbook ? (
              <OrderBookVisualizer orderbook={orderbook} />
            ) : (
              <div style={{ fontSize: 11, color: C.muted, fontStyle: 'italic' }}>
                No order book data available.
              </div>
            )}
          </div>
        </div>

        {/* ML Model Section */}
        <div style={{ padding: '16px', borderBottom: `1px solid ${C.border}` }}>
          <h3 style={{ fontSize: 13, fontWeight: 700, color: '#60a5fa', margin: '0 0 10px 0', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            🔮 Deep Learning Predictor
          </h3>
          
          <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px', marginBottom: '12px' }}>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: '6px' }}>{selectedModel.toUpperCase()} Trading Model ({symbol})</div>
            {predictLoading ? (
              <div style={{ fontSize: 12, color: C.text, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /> Computing features & scaling...
              </div>
            ) : nextDayPred ? (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '4px 0' }}>
                  <span style={{ fontSize: 12, color: C.text }}>Signal Forecast:</span>
                  <span style={{
                    fontSize: 12, fontWeight: 800, padding: '2px 8px', borderRadius: 4,
                    background: nextDayPred.signal === 'buy' ? 'rgba(38,166,154,0.15)' : nextDayPred.signal === 'sell' ? 'rgba(239,83,80,0.15)' : 'rgba(77,106,138,0.15)',
                    color: nextDayPred.signal === 'buy' ? C.up : nextDayPred.signal === 'sell' ? C.down : C.muted,
                    textTransform: 'uppercase'
                  }}>
                    {nextDayPred.signal}
                  </span>
                </div>
                 <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '4px 0' }}>
                  <span style={{ fontSize: 11, color: C.muted }}>Target Price:</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#f59e0b' }}>
                    ₹{nextDayPred.predicted_price.toFixed(2)}
                  </span>
                </div>
                {nextDayPred.stop_loss && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '4px 0' }}>
                    <span style={{ fontSize: 11, color: C.muted }}>Stop Loss (SL):</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: C.down }}>
                      ₹{nextDayPred.stop_loss.toFixed(2)}
                    </span>
                  </div>
                )}
                {nextDayPred.take_profit && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '4px 0' }}>
                    <span style={{ fontSize: 11, color: C.muted }}>Take Profit (TP):</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: C.up }}>
                      ₹{nextDayPred.take_profit.toFixed(2)}
                    </span>
                  </div>
                )}
                <div style={{ fontSize: 10, color: C.muted, display: 'flex', gap: 6, marginTop: '8px' }}>
                  <span>Buy: {(nextDayPred.probs[1]*100).toFixed(0)}%</span>
                  <span>Sell: {(nextDayPred.probs[2]*100).toFixed(0)}%</span>
                  <span>Hold: {(nextDayPred.probs[0]*100).toFixed(0)}%</span>
                </div>

                {modelBacktestResults && modelBacktestResults.summary && (
                  <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: `1px solid ${C.border}` }}>
                    <div style={{ fontSize: 10, color: C.muted, fontWeight: 700, textTransform: 'uppercase', marginBottom: '6px', letterSpacing: '0.05em' }}>
                      Backtest Results
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>NET PROFIT</div>
                        <div style={{
                          fontSize: 11,
                          fontWeight: 700,
                          color: modelBacktestResults.summary.net_profit >= 0 ? C.up : C.down
                        }}>
                          ₹{modelBacktestResults.summary.net_profit.toLocaleString()} ({modelBacktestResults.summary.net_profit_pct}%)
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>WIN RATE</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                          {modelBacktestResults.summary.win_rate}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>TOTAL TRADES</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                          {modelBacktestResults.summary.total_trades}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>MAX DRAWDOWN</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: C.down }}>
                          {modelBacktestResults.summary.max_drawdown}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>PROFIT FACTOR</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                          {modelBacktestResults.summary.profit_factor}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>SHARPE RATIO</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                          {modelBacktestResults.summary.sharpe_ratio}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: C.muted }}>REWARD:RISK RATIO</div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: C.text }}>
                          {modelBacktestResults.summary.reward_risk_ratio ?? '0.0'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 11, color: C.muted, fontStyle: 'italic' }}>
                No active model overlays. Click compute below.
              </div>
            )}
          </div>

          <div style={{ marginBottom: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: 10, color: C.muted, fontWeight: 600, textTransform: 'uppercase' }}>Model Selector</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%', padding: '6px 10px', background: C.bg, color: C.text,
                border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 11,
                outline: 'none', cursor: 'pointer'
              }}
            >
              <option value="tft">🔮 Temporal Fusion Transformer (TFT)</option>
              <option value="lstm">📈 Long Short-Term Memory (LSTM)</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={runPredictions}
              disabled={predictLoading}
              style={{
                flex: 1, background: C.blue, border: 'none', borderRadius: 6,
                color: '#fff', padding: '8px 12px', fontSize: 11, fontWeight: 700,
                cursor: 'pointer', transition: 'opacity 0.12s'
              }}
            >
              {modelOverlayActive ? '🔄 Recompute Model' : `🔮 Run ${selectedModel.toUpperCase()} Model`}
            </button>
            {modelOverlayActive && (
              <button
                onClick={clearModelOverlay}
                style={{
                  background: 'transparent', border: `1px solid ${C.border}`, borderRadius: 6,
                  color: C.muted, padding: '8px 12px', fontSize: 11, fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Execution Engine Section */}
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <h3 style={{ fontSize: 13, fontWeight: 700, color: '#60a5fa', margin: '0', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            ⚡ Execution Engine
          </h3>

          {/* Buy/Sell tab */}
          <div style={{ display: 'flex', background: C.bgCard, padding: '3px', borderRadius: 8, border: `1px solid ${C.border}` }}>
            <button
              onClick={() => setAction('BUY')}
              style={{
                flex: 1, background: action === 'BUY' ? C.up : 'transparent',
                border: 'none', borderRadius: 6, color: '#fff', padding: '8px',
                fontSize: 11, fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s'
              }}
            >
              BUY
            </button>
            <button
              onClick={() => setAction('SELL')}
              style={{
                flex: 1, background: action === 'SELL' ? C.down : 'transparent',
                border: 'none', borderRadius: 6, color: '#fff', padding: '8px',
                fontSize: 11, fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s'
              }}
            >
              SELL
            </button>
          </div>

          {/* Variety Selector (Regular, Intraday, GTT) */}
          <div>
            <label style={{ fontSize: 10, color: C.muted, fontWeight: 700, display: 'block', marginBottom: '5px' }}>ORDER VARIETY</label>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['regular', 'intraday', 'gtt'] as const).map(v => (
                <button
                  key={v}
                  onClick={() => setVariety(v)}
                  style={{
                    flex: 1, background: variety === v ? 'rgba(59,130,246,0.15)' : C.bgCard,
                    border: `1px solid ${variety === v ? C.blue : C.border}`, borderRadius: 6,
                    color: variety === v ? '#60a5fa' : C.muted, padding: '6px',
                    fontSize: 10, fontWeight: 700, textTransform: 'uppercase', cursor: 'pointer'
                  }}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>

          {/* Options Contract Setup (Collapsible) */}
          {mode === 'equity' && (
            <div style={{ background: 'rgba(99,179,237,0.02)', border: `1px dashed ${C.border}`, borderRadius: 8, padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: C.text }}>Option F&O Contract</span>
                <input
                  type="checkbox"
                  checked={isOption}
                  onChange={e => setIsOption(e.target.checked)}
                  style={{ width: 14, height: 14, cursor: 'pointer' }}
                />
              </div>
              {isOption && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>STRIKE PRICE</label>
                      <input
                        type="text"
                        placeholder="e.g. 900"
                        value={strikePrice}
                        onChange={e => setStrikePrice(e.target.value)}
                        style={{
                          width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                          borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                        }}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>OPTION TYPE</label>
                      <select
                        value={optionType}
                        onChange={e => setOptionType(e.target.value as any)}
                        style={{
                          width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                          borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                        }}
                      >
                        <option value="CE">CALL (CE)</option>
                        <option value="PE">PUT (PE)</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>EXPIRY (e.g. Monthly/Weekly)</label>
                    <input
                      type="text"
                      placeholder="e.g. 26JUN"
                      value={expiry}
                      onChange={e => setExpiry(e.target.value)}
                      style={{
                        width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                        borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 10, color: '#f59e0b', fontWeight: 600, marginTop: '4px' }}>
                    Synthesized Contract: {symbol.replace('NSE:', '')}{expiry}{strikePrice}{optionType}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Form parameters */}
          <div style={{ display: 'flex', gap: 8 }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>QUANTITY</label>
              <input
                type="number"
                min="1"
                value={qty}
                onChange={e => setQty(Math.max(1, parseInt(e.target.value) || 1))}
                style={{
                  width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                  borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                }}
              />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>PRICE TYPE</label>
              <select
                value={orderType}
                onChange={e => setOrderType(e.target.value as any)}
                style={{
                  width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                  borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                }}
              >
                <option value="MARKET">MARKET</option>
                <option value="LIMIT">LIMIT</option>
              </select>
            </div>
          </div>

          {/* Price inputs based on selections */}
          {orderType === 'LIMIT' && (
            <div>
              <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>LIMIT PRICE (₹)</label>
              <input
                type="text"
                value={price}
                onChange={e => setPrice(e.target.value)}
                style={{
                  width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                  borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                }}
              />
            </div>
          )}

          {variety === 'gtt' && (
            <div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <div style={{ flex: 2 }}>
                  <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>GTT TRIGGER PRICE (₹)</label>
                  <input
                    type="text"
                    value={triggerPrice}
                    onChange={e => setTriggerPrice(e.target.value)}
                    style={{
                      width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                      borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                    }}
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ fontSize: 9, color: C.muted, display: 'block', marginBottom: '3px' }}>% FROM LTP</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="e.g. -5"
                    onChange={e => {
                      const pct = parseFloat(e.target.value);
                      if (!isNaN(pct) && lastC) {
                        const newPrice = lastC.close * (1 + pct / 100);
                        setTriggerPrice(newPrice.toFixed(2));
                        if (orderType === 'LIMIT') setPrice(newPrice.toFixed(2));
                      }
                    }}
                    style={{
                      width: '100%', background: C.bgCard, border: `1px solid ${C.border}`,
                      borderRadius: 6, padding: '6px', fontSize: 11, color: C.text, boxSizing: 'border-box'
                    }}
                  />
                </div>
              </div>
              <div style={{ fontSize: 9, color: C.muted, marginTop: '6px', lineHeight: '1.2' }}>
                Note: Trigger price must cross the current price (strictly above or below LTP). It cannot be exactly equal.
              </div>
            </div>
          )}

          {/* Confirmation Checklist */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: '6px' }}>
            <input
              type="checkbox"
              id="confirmCheck"
              checked={confirmTrade}
              onChange={e => setConfirmTrade(e.target.checked)}
              style={{ width: 14, height: 14, cursor: 'pointer' }}
            />
            <label htmlFor="confirmCheck" style={{ fontSize: 11, color: C.muted, cursor: 'pointer' }}>
              Confirm trade details are correct
            </label>
          </div>

          {/* Order Placement Action Button */}
          <button
            onClick={handlePlaceOrder}
            disabled={tradeLoading || !confirmTrade}
            style={{
              background: action === 'BUY' ? C.up : C.down,
              opacity: confirmTrade ? 1 : 0.4,
              border: 'none', borderRadius: 8, color: '#fff', padding: '10px 14px',
              fontSize: 12, fontWeight: 700, cursor: confirmTrade ? 'pointer' : 'not-allowed',
              transition: 'opacity 0.15s', width: '100%', marginTop: '6px'
            }}
          >
            {tradeLoading ? 'Placing Order...' : `${action} ${qty} SHARES`}
          </button>

          {/* Trade Execution Feedback Result */}
          {tradeResult && (
            <div style={{
              background: tradeResult.status === 'SUCCESS' ? 'rgba(38,166,154,0.08)' : 'rgba(239,83,80,0.08)',
              border: `1px solid ${tradeResult.status === 'SUCCESS' ? C.up : C.down}`,
              borderRadius: 8, padding: '10px', marginTop: '10px'
            }}>
              <div style={{
                fontSize: 11, fontWeight: 700,
                color: tradeResult.status === 'SUCCESS' ? C.up : C.down,
                textTransform: 'uppercase', marginBottom: '4px'
              }}>
                {tradeResult.status === 'SUCCESS' ? 'SUCCESS' : 'EXECUTION FAILED'}
              </div>
              <div style={{ fontSize: 11, color: C.text }}>
                {tradeResult.message}
              </div>
              {tradeResult.order_id && (
                <div style={{ fontSize: 9, color: C.muted, marginTop: '4px' }}>
                  Order ID: {tradeResult.order_id}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Pine Script Strategy Tester Section */}
        <div style={{ padding: '16px', borderTop: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div
            onClick={() => setShowPineTester(!showPineTester)}
            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
          >
            <h3 style={{ fontSize: 13, fontWeight: 700, color: '#60a5fa', margin: '0', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              🌲 Pine Script Strategy Tester
            </h3>
            <span style={{ color: C.muted, fontSize: 11 }}>{showPineTester ? '▼' : '▲'}</span>
          </div>
          
          {showPineTester && (
            <>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: 10, color: C.muted, fontWeight: 700 }}>PINE SCRIPT CODE</label>
                <textarea
                  value={pineScript}
                  onChange={e => setPineScript(e.target.value)}
                  rows={8}
                  style={{
                    width: '100%',
                    background: C.bg,
                    color: C.text,
                    border: `1px solid ${C.border}`,
                    borderRadius: 6,
                    padding: '8px',
                    fontSize: 11,
                    fontFamily: 'monospace',
                    outline: 'none',
                    resize: 'vertical',
                    boxSizing: 'border-box'
                  }}
                />
              </div>

              <button
                onClick={handlePineBacktest}
                disabled={pineBacktestLoading}
                style={{
                  background: C.blue,
                  border: 'none',
                  borderRadius: 6,
                  color: '#fff',
                  padding: '8px 12px',
                  fontSize: 11,
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'opacity 0.12s'
                }}
              >
                {pineBacktestLoading ? (
                  <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                    <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2 }} /> Running Backtest...
                  </span>
                ) : '🌲 Run Pine Backtest'}
              </button>

              {pineError && (
                <div style={{
                  background: 'rgba(239,83,80,0.08)',
                  border: `1px solid ${C.down}`,
                  borderRadius: 6,
                  padding: '8px',
                  fontSize: 11,
                  color: '#f87171'
                }}>
                  ⚠️ {pineError}
                </div>
              )}

              {pineResults && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ background: C.bgCard, border: `1px solid ${C.border}`, borderRadius: 8, padding: '12px' }}>
                    <div style={{ fontSize: 11, color: C.muted, marginBottom: '8px' }}>
                      Results for {symbol} ({pineResults.strategy_type})
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>NET PROFIT</div>
                        <div style={{
                          fontSize: 13,
                          fontWeight: 700,
                          color: pineResults.summary.net_profit >= 0 ? C.up : C.down
                        }}>
                          ₹{pineResults.summary.net_profit.toLocaleString()} ({pineResults.summary.net_profit_pct}%)
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>WIN RATE</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                          {pineResults.summary.win_rate}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>TOTAL TRADES</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                          {pineResults.summary.total_trades}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>MAX DRAWDOWN</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.down }}>
                          {pineResults.summary.max_drawdown}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>PROFIT FACTOR</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                          {pineResults.summary.profit_factor}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>SHARPE RATIO</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                          {pineResults.summary.sharpe}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: C.muted }}>REWARD:RISK RATIO</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: C.text }}>
                          {pineResults.summary.reward_risk_ratio ?? '0.0'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {pineResults.trades.length > 0 ? (
                    <div>
                      <div style={{ fontSize: 10, color: C.muted, fontWeight: 700, marginBottom: '6px', textTransform: 'uppercase' }}>
                        Recent Trades (Last 5)
                      </div>
                      <div style={{
                        background: C.bgCard,
                        border: `1px solid ${C.border}`,
                        borderRadius: 8,
                        overflow: 'hidden'
                      }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
                          <thead>
                            <tr style={{ background: 'rgba(255,255,255,0.02)', borderBottom: `1px solid ${C.border}` }}>
                              <th style={{ padding: '6px 8px', textAlign: 'left', color: C.muted }}>Time</th>
                              <th style={{ padding: '6px 8px', textAlign: 'right', color: C.muted }}>Entry/Exit</th>
                              <th style={{ padding: '6px 8px', textAlign: 'right', color: C.muted }}>P&L</th>
                            </tr>
                          </thead>
                          <tbody>
                            {pineResults.trades.slice(-5).map((t: any, idx: number) => (
                              <tr key={idx} style={{ borderBottom: idx < 4 ? `1px solid rgba(255,255,255,0.02)` : 'none' }}>
                                <td style={{ padding: '6px 8px', color: C.text }}>
                                  {t.exit_time.split(' ')[0]}
                                </td>
                                <td style={{ padding: '6px 8px', textAlign: 'right', color: C.text }}>
                                  ₹{t.entry_price} → ₹{t.exit_price}
                                </td>
                                <td style={{
                                  padding: '6px 8px',
                                  textAlign: 'right',
                                  fontWeight: 600,
                                  color: t.pnl >= 0 ? C.up : C.down
                                }}>
                                  {t.pnl >= 0 ? '+' : ''}{t.pnl}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: C.muted, fontStyle: 'italic', textAlign: 'center', padding: '10px' }}>
                      No trades executed by this strategy.
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  )
}

// keep TS happy — C is used in chart setup but TypeScript can't see the inline ref
const COLORS = C
