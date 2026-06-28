import logging
import threading
import time
import psycopg2
import datetime
from src.trading_system.memory import DB_CONFIG
from src.options_engine import GreeksCalculator

logger = logging.getLogger(__name__)

class DynamicHedgingEngine:
    """
    A daemon that continuously monitors a mock portfolio of options and automatically
    executes paper trades (logging to DB) to maintain Delta neutrality.
    """
    def __init__(self, delta_threshold=0.5, check_interval=60):
        self.delta_threshold = delta_threshold
        self.check_interval = check_interval
        self._running = False
        self._thread = None
        self.last_net_delta = 0.0
        self.is_active = False # Manual killswitch

    def start(self):
        if not self._running:
            self._running = True
            self.is_active = False # default to off until enabled in UI
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            logger.info("Dynamic Hedging Engine started (monitoring only).")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            logger.info("Dynamic Hedging Engine stopped.")

    def set_active(self, active: bool):
        self.is_active = active
        logger.info(f"Dynamic Hedging is now {'ACTIVE' if active else 'PAUSED'}")

    def _get_spot_price(self, symbol):
        # Stub for getting live spot price.
        # Fallback to yfinance if Kite fails.
        import yfinance as yf
        try:
            from src.trading_system.tools import get_kite
            kite = get_kite()
            if kite:
                quote = kite.quote([symbol])
                if symbol in quote:
                    return quote[symbol]['last_price']
        except Exception:
            pass
            
        try:
            yf_symbol = symbol
            if 'NIFTY' in symbol:
                yf_symbol = '^NSEI' if 'BANK' not in symbol else '^NSEBANK'
            ticker = yf.Ticker(yf_symbol)
            return ticker.fast_info.get('lastPrice', 25000) # Mock default
        except Exception:
            return 25000

    def _loop(self):
        while self._running:
            try:
                self._run_hedge_cycle()
            except Exception as e:
                logger.error(f"Hedging cycle error: {e}")
            time.sleep(self.check_interval)

    def _run_hedge_cycle(self):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            
            # Fetch all active positions
            cur.execute("SELECT id, symbol, option_type, strike, expiry, quantity FROM hedging_positions")
            positions = cur.fetchall()
            
            if not positions:
                self.last_net_delta = 0.0
                return
                
            portfolio_delta = 0.0
            symbol_spots = {}
            
            for pid, symbol, opt_type, strike, expiry, qty in positions:
                # If it's a future position used for hedging, delta is exactly 1 per unit
                if opt_type.lower() == 'future':
                    portfolio_delta += float(qty)
                    continue

                if symbol not in symbol_spots:
                    symbol_spots[symbol] = self._get_spot_price(symbol)
                
                spot = symbol_spots[symbol]
                
                # Calculate time to expiry in years
                if expiry:
                    tte = max((expiry - datetime.date.today()).days / 365.0, 1/365.0)
                else:
                    tte = 30 / 365.0 # default 30 days
                    
                iv = 0.15 
                r_d = 0.05
                
                greeks = GreeksCalculator.calculate_greeks(spot, float(strike), tte, r_d, iv, opt_type.lower())
                pos_delta = greeks['delta'] * qty
                portfolio_delta += pos_delta
                
            self.last_net_delta = portfolio_delta
            
            # Check if threshold is breached AND system is active
            if self.is_active and abs(portfolio_delta) > self.delta_threshold:
                hedge_qty = -int(round(portfolio_delta))
                
                if hedge_qty != 0:
                    trade_type = 'BUY' if hedge_qty > 0 else 'SELL'
                    hedge_symbol = list(symbol_spots.keys())[0] if symbol_spots else 'NSE:NIFTY 50'
                    hedge_price = symbol_spots.get(hedge_symbol, 25000)
                    
                    logger.info(f"Hedge triggered: Net Delta {portfolio_delta}. Executing {trade_type} {abs(hedge_qty)} {hedge_symbol} Futures @ {hedge_price}")
                    
                    cur.execute("""
                        INSERT INTO hedge_trades (symbol, trade_type, quantity, price, net_delta_before)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (f"{hedge_symbol}-FUT", trade_type, abs(hedge_qty), float(hedge_price), float(portfolio_delta)))
                    
                    cur.execute("""
                        INSERT INTO hedging_positions (symbol, option_type, strike, expiry, quantity)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (f"{hedge_symbol}-FUT", 'future', 0, None, hedge_qty))
                    
                    conn.commit()

            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB Error in hedging engine: {e}")

# Global singleton instance
hedger = DynamicHedgingEngine(check_interval=10) # 10 sec interval for testing
