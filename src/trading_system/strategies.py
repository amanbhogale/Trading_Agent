import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Takes a dataframe with OHLCV data and returns the dataframe with a 'sig' column.
        'sig' should be 1 for buy/long, -1 for sell/short, 0 for neutral.
        """
        pass

    @abstractmethod
    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        """
        Process a single tick/bar for live execution.
        Returns: 1 (buy), -1 (sell), 0 (hold).
        State is a mutable dictionary used to store rolling historical data.
        """
        pass


class MarketMakingStrategy(BaseStrategy):
    """
    Simulates a basic market making strategy.
    Provides liquidity by placing limit orders around the current price spread.
    """
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        window = kwargs.get("window", 20)
        spread_threshold = kwargs.get("spread_threshold", 0.001)
        
        # Calculate moving average and volatility to determine spread
        df["sma"] = df["close"].rolling(window).mean()
        df["std"] = df["close"].rolling(window).std()
        
        # Simple heuristic: if price drops below sma - spread, buy. If above sma + spread, sell.
        df["sig"] = 0
        df.loc[df["close"] < df["sma"] - (df["std"] * spread_threshold), "sig"] = 1
        df.loc[df["close"] > df["sma"] + (df["std"] * spread_threshold), "sig"] = -1
        
        return df

    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        window = kwargs.get("window", 20)
        spread_threshold = kwargs.get("spread_threshold", 0.001)
        
        # Initialize state list if missing
        if "history" not in state:
            state["history"] = []
            
        state["history"].append(current_bar["close"])
        
        # Keep only required window size
        if len(state["history"]) > window:
            state["history"] = state["history"][-window:]
            
        if len(state["history"]) < window:
            return 0  # Not enough data to compute
            
        history_arr = np.array(state["history"])
        sma = np.mean(history_arr)
        std = np.std(history_arr)
        
        current_close = current_bar["close"]
        if current_close < sma - (std * spread_threshold):
            return 1
        elif current_close > sma + (std * spread_threshold):
            return -1
            
        return 0


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy based on z-score mean reversion (pairs trading simplified to single asset relative to its moving average here).
    """
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        window = kwargs.get("window", 30)
        z_score_threshold = kwargs.get("z_score_threshold", 2.0)
        
        # Calculate rolling mean and standard deviation
        df["mean"] = df["close"].rolling(window=window).mean()
        df["std"] = df["close"].rolling(window=window).std()
        
        # Calculate z-score
        df["z_score"] = (df["close"] - df["mean"]) / df["std"]
        
        df["sig"] = 0
        # Buy when z_score is below negative threshold (undervalued)
        df.loc[df["z_score"] < -z_score_threshold, "sig"] = 1
        # Sell when z_score is above positive threshold (overvalued)
        df.loc[df["z_score"] > z_score_threshold, "sig"] = -1
        
        return df

    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        return 0

    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        return 0


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.
    Buys assets that have had high returns over the past periods and sells those with poor returns.
    """
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        lookback = kwargs.get("lookback", 14)
        
        # Calculate momentum as the rate of change (ROC)
        df["momentum"] = df["close"].pct_change(periods=lookback)
        
        df["sig"] = 0
        df.loc[df["momentum"] > 0, "sig"] = 1
        df.loc[df["momentum"] < 0, "sig"] = -1
        
        return df

    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        return 0


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy based on RSI and Bollinger Bands.
    """
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        bb_window = kwargs.get("bb_window", 20)
        
        df["sma"] = df["close"].rolling(bb_window).mean()
        df["std"] = df["close"].rolling(bb_window).std()
        df["bb_upper"] = df["sma"] + 2 * df["std"]
        df["bb_lower"] = df["sma"] - 2 * df["std"]
        
        df["sig"] = 0
        df.loc[df["close"] < df["bb_lower"], "sig"] = 1  # Oversold
        df.loc[df["close"] > df["bb_upper"], "sig"] = -1 # Overbought
        
        return df

    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        return 0


class SentimentBasedStrategy(BaseStrategy):
    """
    Sentiment Based Trading Strategy.
    Uses external sentiment scores (simulated here) to trade.
    """
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        # In a real scenario, this would merge with a sentiment dataset.
        # Here we simulate sentiment using a random walk or simply assume it's provided in kwargs/data.
        if "sentiment_score" not in df.columns:
            # Simulated sentiment score between -1 and 1
            np.random.seed(42)
            df["sentiment_score"] = np.random.uniform(-1, 1, len(df))
            
        threshold = kwargs.get("sentiment_threshold", 0.5)
        
        df["sig"] = 0
        df.loc[df["sentiment_score"] > threshold, "sig"] = 1
        df.loc[df["sentiment_score"] < -threshold, "sig"] = -1
        
        return df

    def step(self, current_bar: dict, state: dict, **kwargs) -> int:
        return 0
