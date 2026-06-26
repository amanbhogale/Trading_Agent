import time
from collections import deque
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class APIManager:
    def __init__(self):
        # API Keys and credentials
        self.finnhub_api_key: Optional[str] = None
        self.delta_api_key: Optional[str] = None
        self.delta_api_secret: Optional[str] = None
        
        # Call tracking: mapping of api_name -> deque of timestamps
        self.call_history: Dict[str, deque] = {
            'kite': deque(),
            'delta': deque(),
            'finnhub': deque(),
            'yfinance': deque()
        }
        
        # Rate limits (calls per minute)
        self.limits: Dict[str, int] = {
            'kite': 120,
            'delta': 120,
            'finnhub': 30,
            'yfinance': 9999
        }

        # Yahoo Finance WebSocket Cache
        self.yahoo_live_prices: Dict[str, Dict[str, Any]] = {}
        self._yahoo_ws_thread: Optional[Any] = None
        self._yahoo_subscribed_symbols = set()
        self._yahoo_ws_client: Optional[Any] = None

    def set_finnhub_key(self, key: str):
        self.finnhub_api_key = key
        logger.info("Finnhub key updated in API Manager")

    def set_delta_credentials(self, key: str, secret: str):
        self.delta_api_key = key
        self.delta_api_secret = secret
        logger.info("Delta Exchange credentials updated in API Manager")

    def track_call(self, api_name: str):
        if api_name in self.call_history:
            self.call_history[api_name].append(time.time())
            logger.debug(f"Tracked call for {api_name}. Total in last 60s: {self.get_calls_last_minute(api_name)}")

    def _clean_old_calls(self, api_name: str):
        now = time.time()
        window = 60.0  # 1 minute
        history = self.call_history.get(api_name)
        if history:
            while history and (now - history[0] > window):
                history.popleft()

    def get_calls_last_minute(self, api_name: str) -> int:
        self._clean_old_calls(api_name)
        return len(self.call_history.get(api_name, []))

    def get_limit(self, api_name: str) -> int:
        return self.limits.get(api_name, 0)

    def is_rate_limited(self, api_name: str) -> bool:
        calls = self.get_calls_last_minute(api_name)
        limit = self.get_limit(api_name)
        # Safety buffer of 2 calls to prevent hard lockouts
        return calls >= max(1, limit - 2)

    def is_available(self, api_name: str) -> bool:
        if api_name == 'finnhub':
            return bool(self.finnhub_api_key)
        elif api_name == 'delta':
            return bool(self.delta_api_key and self.delta_api_secret)
        elif api_name == 'kite':
            try:
                from src.trading_system.tools import kite_available
                return kite_available()
            except ImportError:
                return False
        elif api_name == 'yfinance':
            return True
        return False

    def get_api_status(self) -> Dict[str, Any]:
        status = {}
        for api in ['kite', 'delta', 'finnhub', 'yfinance']:
            status[api] = {
                'connected': self.is_available(api),
                'calls_last_minute': self.get_calls_last_minute(api),
                'limit': self.get_limit(api),
                'rate_limited': self.is_rate_limited(api)
            }
        return status

    def route_ohlcv_request(self, symbol: str, interval: str, days: int) -> str:
        """
        Routes the OHLCV request based on symbol, timeframe lookback, and API rate limits.
        """
        days = int(days)
        # Rule: For longer charts (days > 30), use Yahoo Finance only.
        if days > 30:
            logger.info(f"Routing {symbol} to Yahoo Finance (longer chart lookback: {days} days)")
            return 'yfinance'

        # Shorter charts: divide work based on call limits
        is_crypto = symbol.upper().strip().endswith("USDT") or symbol.upper().strip().startswith("P-") or symbol.upper().strip().startswith("C-") or symbol.upper().strip().startswith("F-")
        
        if is_crypto:
            if self.is_available('delta') and not self.is_rate_limited('delta'):
                return 'delta'
            else:
                logger.info("Delta is unavailable or rate limited. Routing crypto request to Yahoo Finance.")
                return 'yfinance'
                
        # Indian Equities vs US/Global Equities
        is_indian = symbol.upper().strip().startswith("NSE:") or symbol.upper().strip().startswith("BSE:")
        
        if is_indian:
            if self.is_available('kite') and not self.is_rate_limited('kite'):
                return 'kite'
            elif self.is_available('finnhub') and not self.is_rate_limited('finnhub'):
                return 'finnhub'
            else:
                logger.info("Kite & Finnhub are unavailable/rate limited for Indian asset. Routing to Yahoo Finance.")
                return 'yfinance'
        else:
            # US/Global equities
            if self.is_available('finnhub') and not self.is_rate_limited('finnhub'):
                return 'finnhub'
            else:
                logger.info("Finnhub is unavailable or rate limited for global asset. Routing to Yahoo Finance.")
                return 'yfinance'

    def start_yahoo_websocket(self, symbols_to_subscribe):
        """Starts or updates the Yahoo Finance WebSocket connection in a daemon thread."""
        import threading
        
        # Clean symbols
        new_syms = [s for s in symbols_to_subscribe if s and s not in self._yahoo_subscribed_symbols]
        if not new_syms and self._yahoo_ws_thread is not None:
            return
            
        self._yahoo_subscribed_symbols.update(new_syms)
        
        if self._yahoo_ws_thread is not None:
            logger.info(f"New symbols added to Yahoo WS. Restarting connection to subscribe to: {new_syms}")
            # If we already have a client, we can close it to force a reconnect with the updated list
            if self._yahoo_ws_client:
                try:
                    self._yahoo_ws_client.close()
                except Exception as e:
                    logger.debug(f"Error closing old Yahoo WS client: {e}")
            return

        def run_loop():
            import yfinance as yf
            while True:
                current_list = list(self._yahoo_subscribed_symbols)
                if not current_list:
                    time.sleep(2)
                    continue
                try:
                    logger.info(f"Connecting Yahoo Finance WebSocket for {len(current_list)} symbols...")
                    self._yahoo_ws_client = yf.WebSocket()
                    self._yahoo_ws_client.subscribe(current_list)
                    
                    def on_message(message):
                        sym = message.get('id')
                        price = message.get('price')
                        if sym and price is not None:
                            self.yahoo_live_prices[sym] = {
                                'price': float(price),
                                'change': float(message.get('change', 0)),
                                'change_percent': float(message.get('change_percent', 0)),
                                'updated_at': time.time()
                            }
                    
                    self._yahoo_ws_client.listen(on_message)
                except Exception as e:
                    logger.warning(f"Yahoo Finance WebSocket error: {e}. Reconnecting in 10s...")
                    time.sleep(10)
                finally:
                    self._yahoo_ws_client = None

        self._yahoo_ws_thread = threading.Thread(target=run_loop, daemon=True)
        self._yahoo_ws_thread.start()
        logger.info("Yahoo Finance WebSocket listener thread started successfully")

# Singleton instance
api_manager = APIManager()
