import pandas as pd
import numpy as np
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

class BinanceDataset:
    def __init__(self, dataset_name="adamzzzz/binance-1m-klines-20240721", split="train"):
        self.dataset_name = dataset_name
        self.split = split
        self.data = None
        
    def load_and_preprocess(self):
        logger.info(f"Loading dataset {self.dataset_name}...")
        try:
            # We take a sample dataset
            ds = load_dataset(self.dataset_name, split=self.split)
            df = ds.to_pandas()
        except Exception as e:
            logger.warning(f"Failed to load HF dataset: {e}. Generating synthetic data for TFT testing.")
            df = self._generate_synthetic_data()
            
        # Standard OHLCV columns expected
        # If columns differ, we'll try to map them or fallback to synthetic
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required_cols):
            logger.warning("Missing required columns. Generating synthetic data.")
            df = self._generate_synthetic_data()

        # Sort by time if 'timestamp' exists
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # Feature Engineering for TFT
        # Log Returns (Stationary)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility (Rolling standard deviation of returns)
        df['volatility_20'] = df['log_return'].rolling(window=20).std()
        
        # Volume change
        df['volume_change'] = np.log(df['volume'] / df['volume'].shift(1).replace(0, np.nan))
        
        # Distance from Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        df = df.dropna().reset_index(drop=True)
        
        # Scale features
        feature_cols = ['log_return', 'volatility_20', 'volume_change', 'dist_sma_20']
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0

        self.data = df
        return df

    def _generate_synthetic_data(self, n_steps=10000):
        # Generate random walk
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.002, n_steps)
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_steps, freq='1min'),
            'open': prices * (1 + np.random.normal(0, 0.001, n_steps)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_steps))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_steps))),
            'close': prices,
            'volume': np.abs(np.random.normal(1000, 200, n_steps))
        })
        return df
