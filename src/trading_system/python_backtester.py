import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import yfinance as yf
from src.trading_system.tools import _kite_to_yf
import logging

logger = logging.getLogger(__name__)

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()

class RsiOscillator(Strategy):
    n_rsi = 14
    lower_bound = 30
    upper_bound = 70

    def init(self):
        def compute_rsi(close_array, n):
            s = pd.Series(close_array)
            delta = s.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=n-1, adjust=False).mean()
            ema_down = down.ewm(com=n-1, adjust=False).mean()
            rs = ema_up / ema_down
            return (100 - (100 / (1 + rs))).values
            
        self.rsi = self.I(compute_rsi, self.data.Close, self.n_rsi)

    def next(self):
        if crossover(self.rsi, self.lower_bound):
            self.buy()
        elif crossover(self.upper_bound, self.rsi):
            self.position.close()

def run_backtest(symbol: str, strategy_name: str, days: int = 365, interval: str = '1d') -> dict:
    try:
        yf_sym = _kite_to_yf(symbol)
        period = f"{days}d"
        
        logger.info(f"Fetching data for {yf_sym} over {period} ({interval})")
        df = yf.download(yf_sym, period=period, interval=interval, progress=False)
        
        if df.empty:
            raise ValueError(f"No data fetched for {symbol}")
            
        # Backtesting.py requires columns: Open, High, Low, Close, Volume
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            # Try capitalizing
            df.columns = [c.capitalize() for c in df.columns]
            
        df = df[required_cols].dropna()

        # Select Strategy
        strat_cls = SmaCross
        if strategy_name.lower() == 'rsi':
            strat_cls = RsiOscillator
            
        # Initialize Backtest
        bt = Backtest(df, strat_cls, cash=100000, commission=.002, exclusive_orders=True)
        stats = bt.run()
        
        # Prepare JSON response
        result_dict = {
            "Return [%]": round(stats.get('Return [%]', 0), 2),
            "Buy & Hold Return [%]": round(stats.get('Buy & Hold Return [%]', 0), 2),
            "Max Drawdown [%]": round(stats.get('Max. Drawdown [%]', 0), 2),
            "Win Rate [%]": round(stats.get('Win Rate [%]', 0), 2),
            "Trades": int(stats.get('# Trades', 0)),
            "Sharpe Ratio": round(stats.get('Sharpe Ratio', 0), 2)
        }
        
        return {
            "success": True,
            "symbol": symbol,
            "strategy": strategy_name,
            "stats": result_dict,
            "raw_text": str(stats)
        }
    except Exception as e:
        logger.exception("Backtest failed")
        return {"success": False, "error": str(e)}
