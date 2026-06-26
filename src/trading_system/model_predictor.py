import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import ta
import logging

logger = logging.getLogger(__name__)

class LSTMTradingModel(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 256, num_layers: int = 4,
                 output_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1    = nn.Linear(hidden_size, 64)
        self.relu   = nn.ReLU()
        self.drop   = nn.Dropout(0.30)
        self.fc2    = nn.Linear(64, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.layer_norm(out)
        out = self.drop(self.relu(self.fc1(out)))
        return self.fc2(out)

# Load device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache model and scaler globally
_model_instance = None
_scaler_instance = None

def get_model_and_scaler():
    global _model_instance, _scaler_instance
    if _model_instance is not None:
        return _model_instance, _scaler_instance
        
    model_path = Path("deep_learning_backtest/models/lstm_checkpoints/best_lstm.pth")
    if not model_path.exists():
        logger.error("LSTM checkpoint not found at %s", model_path)
        return None, None
        
    try:
        # Load model structure
        model = LSTMTradingModel(input_size=32, hidden_size=256, num_layers=4, output_size=3, dropout=0.2).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        _model_instance = model
        logger.info("LSTM model loaded successfully from %s", model_path)
    except Exception as e:
        logger.exception("Error loading LSTM model: %s", e)
        return None, None

    # Load scaler from processed pickle files if possible
    try:
        processed_dir = Path("deep_learning_backtest/data/processed")
        pkl_files = list(processed_dir.glob("*.pkl"))
        if pkl_files:
            # Load scaler from the first file (usually SPY or AAPL)
            with open(pkl_files[0], 'rb') as f:
                data = pickle.load(f)
                _scaler_instance = data.get('scaler')
                logger.info("Scaler loaded successfully from %s", pkl_files[0].name)
        else:
            logger.warning("No processed pickle files found to load scaler.")
    except Exception as e:
        logger.error("Error loading scaler: %s", e)
        
    return _model_instance, _scaler_instance

def compute_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact 32 features required by LSTM model."""
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

    # Fill NaNs or drop them
    df = df.bfill().fillna(0.0)
    return df

def predict_signals(candles: list) -> list:
    """
    Take raw candle dicts from database/API, calculate features, scale them,
    and predict buy/sell/hold signals with probabilities.
    Returns: a list of dicts with signal ('buy'|'sell'|'hold'), probs, and times
    """
    if len(candles) < 60:
        return []
        
    model, scaler = get_model_and_scaler()
    if not model:
        logger.error("Model prediction skipped: LSTM model not loaded.")
        return []
        
    # Convert list of candles to DataFrame
    df = pd.DataFrame(candles)
    # Ensure correct column names
    df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    
    # Check if time column exists, or convert index to datetime if needed
    if 'date' in df.columns:
        # Convert unix timestamps to pandas datetime
        df['date'] = pd.to_datetime(df['date'], unit='s')
    else:
        df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')

    # Calculate features
    df_feat = compute_lstm_features(df)
    
    # List of 32 features in exact order
    feat_order = [
        'open', 'high', 'low', 'close', 'volume', 'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist', 
        'stoch_k', 'stoch_d', 'williams_r', 'ema_9', 'ema_21', 'ema_50', 'sma_20', 'adx', 'cci_20', 
        'bb_upper', 'bb_lower', 'bb_bw', 'atr_14', 'atr_7', 'vol_20', 'vol_60', 'obv', 'mfi_14', 
        'vol_ratio', 'log_return', 'hl_spread', 'price_ema50_r'
    ]
    
    # Extract only required columns in exact order (keep as DataFrame to preserve feature names)
    feat_df = df_feat[feat_order]
    
    # Scale data — pass DataFrame so MinMaxScaler sees feature names (suppresses UserWarning)
    if scaler is not None:
        try:
            scaled_data = scaler.transform(feat_df)
        except Exception as e:
            logger.error("Scaler transform failed, using simple MinMaxScaler fallback: %s", e)
            from sklearn.preprocessing import MinMaxScaler
            fallback_scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = fallback_scaler.fit_transform(feat_df.values)
    else:
        from sklearn.preprocessing import MinMaxScaler
        fallback_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = fallback_scaler.fit_transform(feat_df.values)
        
    # Generate rolling sequences of length 60
    # LSTM input shape: (num_sequences, sequence_length=60, num_features=32)
    predictions = []
    
    with torch.no_grad():
        # Start predictions from index 59 (60th candle onwards)
        for i in range(59, len(scaled_data)):
            seq = scaled_data[i-59:i+1] # shape (60, 32)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device) # shape (1, 60, 32)
            
            logits = model(seq_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = logits.argmax(dim=1).cpu().numpy()[0]
            
            # Map predictions: 0 = Hold, 1 = Buy, 2 = Sell
            signal_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            
            candle_time = int(df['date'].iloc[i].timestamp())
            
            # Estimate a model-based predicted next close price as a helper
            # If buy signal, price prediction is slightly up, if sell, down.
            last_close = float(df['close'].iloc[i])
            buy_prob  = float(probs[1])
            sell_prob = float(probs[2])
            hold_prob = float(probs[0])
            predicted_pnl = (buy_prob - sell_prob) * 0.015  # simple directional shift
            predicted_price = round(last_close * (1.0 + predicted_pnl), 2)
            
            predictions.append({
                'time': int(candle_time),
                'signal': signal_map[int(pred_class)],
                'probs': [hold_prob, buy_prob, sell_prob],
                'predicted_price': predicted_price,
            })
            
    return predictions
