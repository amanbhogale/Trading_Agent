"""
features.py — Technical Indicator Feature Engineering
======================================================
Uses the `ta` library (pure-Python, Python 3.14 compatible).
Replaces the previous pandas_ta (pta.*) implementation.

Usage in notebooks:
    from features import add_features
"""

import numpy as np
import pandas as pd
import ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators using the `ta` library.

    Covers all indicators previously computed with pandas_ta:
      Momentum  : RSI (7, 14), MACD, Stochastic K/D, Williams %R
      Trend     : EMA (9/21/50/200), SMA 20, ADX, CCI
      Volatility: Bollinger Bands, ATR (7, 14), rolling vol (20, 60)
      Volume    : OBV, MFI, volume ratio
      Derived   : log return, H-L spread, price/EMA50 ratio

    Args:
        df: OHLCV DataFrame with lowercase columns
            (open, high, low, close, volume).

    Returns:
        DataFrame with NaN rows dropped after indicator calculation.
    """
    df = df.copy()

    # ---- Momentum ----
    df["rsi_14"]      = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_7"]       = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

    _macd             = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = _macd.macd()
    df["macd_signal"] = _macd.macd_signal()
    df["macd_hist"]   = _macd.macd_diff()

    _stoch            = ta.momentum.StochasticOscillator(
                            df["high"], df["low"], df["close"], window=14, smooth_window=3
                        )
    df["stoch_k"]     = _stoch.stoch()
    df["stoch_d"]     = _stoch.stoch_signal()

    df["williams_r"]  = ta.momentum.WilliamsRIndicator(
                            df["high"], df["low"], df["close"], lbp=14
                        ).williams_r()

    # ---- Trend ----
    df["ema_9"]       = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"]      = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"]     = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["sma_20"]      = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()

    _adx              = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]         = _adx.adx()

    df["cci_20"]      = ta.trend.CCIIndicator(
                            df["high"], df["low"], df["close"], window=20
                        ).cci()

    # ---- Volatility ----
    _bb               = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"]    = _bb.bollinger_hband()
    df["bb_middle"]   = _bb.bollinger_mavg()
    df["bb_lower"]    = _bb.bollinger_lband()
    df["bb_bw"]       = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    df["atr_14"]      = ta.volatility.AverageTrueRange(
                            df["high"], df["low"], df["close"], window=14
                        ).average_true_range()
    df["atr_7"]       = ta.volatility.AverageTrueRange(
                            df["high"], df["low"], df["close"], window=7
                        ).average_true_range()

    df["vol_20"]      = df["close"].pct_change().rolling(20).std()
    df["vol_60"]      = df["close"].pct_change().rolling(60).std()

    # ---- Volume ----
    df["obv"]         = ta.volume.OnBalanceVolumeIndicator(
                            df["close"], df["volume"]
                        ).on_balance_volume()
    df["mfi_14"]      = ta.volume.MFIIndicator(
                            df["high"], df["low"], df["close"], df["volume"], window=14
                        ).money_flow_index()
    df["vol_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()

    # ---- Derived ----
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["hl_spread"]     = (df["high"] - df["low"]) / df["close"]
    df["price_ema50_r"] = df["close"] / df["ema_50"]

    df.dropna(inplace=True)
    return df
