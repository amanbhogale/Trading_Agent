import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import ta
import logging

logger = logging.getLogger(__name__)

# Load device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# TEMPORAL FUSION TRANSFORMER (TFT) ARCHITECTURE CLASSES
# =====================================================================

class GatedLinearUnit(nn.Module):
    """GLU: applies sigmoid gating to control information flow."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns input feature importance at each timestep."""
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

        self.weight_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_features, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        flat_x = x.reshape(batch_size * seq_len, self.num_features)
        weights = self.softmax(self.weight_grn(flat_x))
        weights = weights.reshape(batch_size, seq_len, self.num_features)

        processed_features = []
        for i in range(self.num_features):
            feat = x[:, :, i:i+1]
            feat_flat = feat.reshape(batch_size * seq_len, 1)
            processed = self.feature_grns[i](feat_flat)
            processed = processed.reshape(batch_size, seq_len, self.hidden_dim)
            processed_features.append(processed)

        stacked = torch.stack(processed_features, dim=2)
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=2)

        return weighted, weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for price direction prediction."""
    def __init__(
        self,
        input_dim: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 1,
        seq_len: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        self.vsn = VariableSelectionNetwork(
            input_dim=input_dim,
            num_features=input_dim,
            hidden_dim=d_model,
            dropout=dropout,
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 50)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.post_attn_grn = GatedResidualNetwork(
            d_model, dim_feedforward, d_model, dropout
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        vsn_out, vsn_weights = self.vsn(x)
        vsn_out = self.pos_encoder(vsn_out)
        attn_out = self.transformer_encoder(vsn_out)
        last_step = attn_out[:, -1, :]
        gated = self.post_attn_grn(last_step)
        gated = self.final_norm(gated)
        logits = self.output_head(gated)
        return logits.squeeze(-1), vsn_weights


# =====================================================================
# LSTM MODEL ARCHITECTURE CLASS
# =====================================================================

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

# Cache loaded instances globally
_model_instance = None
_scaler_instance = None
_model_type = None  # 'lstm' or 'tft'
_threshold = 0.50
_seq_len = 30
_feature_cols = []

def get_model_and_scaler(model_type_override=None):
    global _model_instance, _scaler_instance, _model_type, _threshold, _seq_len, _feature_cols
    if model_type_override is not None:
        model_type_override = model_type_override.lower()
        if model_type_override != _model_type:
            _model_instance = None
            _scaler_instance = None
            
    if _model_instance is not None:
        return _model_instance, _scaler_instance
        
    # Check for TFT models first (V2 overall, V3 overall, V1) if not explicitly requesting LSTM
    load_tft = (model_type_override != 'lstm')
    if load_tft:
        tft_checkpoints = [
            (Path("deep_learning_backtest/models/tft_checkpoints/best_tft_v2_overall.pt"),
             Path("deep_learning_backtest/models/tft_checkpoints/best_tft_v2_meta.pkl")),
            (Path("deep_learning_backtest/models/tft_checkpoints/best_tft_v3_overall.pt"),
             Path("deep_learning_backtest/models/tft_checkpoints/best_tft_v3_meta.pkl")),
            (Path("deep_learning_backtest/models/tft_checkpoints/best_tft.pt"),
             Path("deep_learning_backtest/models/tft_checkpoints/best_tft_meta.pkl"))
        ]
        
        for model_path, meta_path in tft_checkpoints:
            if model_path.exists() and meta_path.exists():
                try:
                    # Load configuration and scaler from meta pickle
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    
                    config = meta.get('config', {})
                    _feature_cols = meta.get('feature_cols', [])
                    if not _feature_cols:
                        # Fallback default feature lists if not saved in meta
                        _feature_cols = [
                            'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
                            'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'williams_r',
                            'macd_norm', 'macd_sig_norm', 'macd_hist_norm',
                            'p_ema9', 'p_ema21', 'p_sma50', 'ema9_21',
                            'atr_norm', 'bb_pct', 'bb_width', 'vol_5', 'vol_20', 'vol_ratio',
                            'vratio', 'adx', 'hl_norm'
                        ] if 'v3' in str(model_path) else [
                            'Open', 'High', 'Low', 'Close', 'Volume',
                            'sma_10', 'sma_30', 'sma_50',
                            'macd', 'macd_signal', 'macd_hist',
                            'rsi', 'stoch',
                            'bb_high', 'bb_low', 'atr', 'bb_width',
                            'return_1d', 'return_2d', 'return_3d'
                        ]
                    
                    _seq_len = config.get('seq_len', 30)
                    _threshold = meta.get('threshold', 0.50)
                    _scaler_instance = meta.get('scaler')
                    
                    # Instantiate TFT model structure
                    model = TemporalFusionTransformer(
                        input_dim=len(_feature_cols),
                        d_model=config.get('d_model', 64),
                        nhead=config.get('nhead', 4),
                        num_layers=config.get('num_layers', 2),
                        dim_feedforward=config.get('dim_feedforward', 128),
                        dropout=config.get('dropout', 0.1),
                        num_classes=1,
                        seq_len=_seq_len
                    ).to(device)
                    
                    ckpt = torch.load(model_path, map_location=device)
                    model.load_state_dict(ckpt['model_state'])
                    model.eval()
                    
                    _model_instance = model
                    _model_type = 'tft'
                    logger.info("Temporal Fusion Transformer loaded successfully from %s (seq_len=%d)", model_path, _seq_len)
                    return _model_instance, _scaler_instance
                except Exception as e:
                    logger.warning("Failed to load TFT model from %s: %s", model_path, e)
                
    # Fallback to original LSTM model if not explicitly requesting TFT
    load_lstm = (model_type_override != 'tft')
    if load_lstm:
        lstm_path = Path("deep_learning_backtest/models/lstm_checkpoints/best_lstm.pth")
        if lstm_path.exists():
            try:
                model = LSTMTradingModel(input_size=32, hidden_size=256, num_layers=4, output_size=3, dropout=0.2).to(device)
                ckpt = torch.load(lstm_path, map_location=device)
                model.load_state_dict(ckpt['model_state'])
                model.eval()
                
                _model_instance = model
                _model_type = 'lstm'
                _seq_len = 60
                _threshold = 0.50
                
                # Load scaler from processed folder
                processed_dir = Path("deep_learning_backtest/data/processed")
                pkl_files = list(processed_dir.glob("*.pkl"))
                if pkl_files:
                    with open(pkl_files[0], 'rb') as f:
                        data = pickle.load(f)
                        _scaler_instance = data.get('scaler')
                        logger.info("Scaler loaded successfully from %s", pkl_files[0].name)
                        
                logger.info("Fallback LSTM model loaded successfully from %s", lstm_path)
                return _model_instance, _scaler_instance
            except Exception as e:
                logger.exception("Error loading fallback LSTM model: %s", e)
            
    return None, None

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

def compute_all_tft_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard, enhanced, and V3 scale-invariant features for TFT models."""
    df = df.copy()
    
    # Standardize column casing
    df.columns = [c.lower() for c in df.columns]
    
    # Base columns
    df['Open'] = df['open']
    df['High'] = df['high']
    df['Low'] = df['low']
    df['Close'] = df['close']
    df['Volume'] = df['volume']
    
    # V3 features
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_2'] = df['Close'].pct_change(2)
    df['ret_3'] = df['Close'].pct_change(3)
    df['ret_5'] = df['Close'].pct_change(5)
    df['ret_10'] = df['Close'].pct_change(10)
    df['ret_20'] = df['Close'].pct_change(20)

    df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
    df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
    df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

    macd = ta.trend.macd(df['Close'])
    macd_sig = ta.trend.macd_signal(df['Close'])
    df['macd_norm'] = macd / (df['Close'] + 1e-8)
    df['macd_sig_norm'] = macd_sig / (df['Close'] + 1e-8)
    df['macd_hist_norm'] = (macd - macd_sig) / (df['Close'] + 1e-8)

    ema9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    sma50 = df['Close'].rolling(50).mean()
    df['p_ema9'] = df['Close'] / (ema9 + 1e-8) - 1
    df['p_ema21'] = df['Close'] / (ema21 + 1e-8) - 1
    df['p_sma50'] = df['Close'] / (sma50 + 1e-8) - 1
    df['ema9_21'] = ema9 / (ema21 + 1e-8) - 1

    df['atr_norm'] = ta.volatility.average_true_range(
        df['High'], df['Low'], df['Close'], window=14
    ) / (df['Close'] + 1e-8)
    df['bb_pct'] = ta.volatility.bollinger_pband(df['Close'], window=20)
    df['bb_width'] = ta.volatility.bollinger_wband(df['Close'], window=20)
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)

    df['vratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['hl_norm'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)

    # Standard & Enhanced (V2) features
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['macd'] = ta.trend.macd(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    df['macd_hist'] = ta.trend.macd_diff(df['Close'])
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['bb_high'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['bb_low'] = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['bb_width_v2'] = (df['bb_high'] - df['bb_low']) / (df['Close'] + 1e-8)
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_2d'] = df['Close'].pct_change(2)
    df['return_3d'] = df['Close'].pct_change(3)

    # Enhanced (V2) features
    df['vol_20'] = df['return_1d'].rolling(20).std()
    df['vol_60'] = df['return_1d'].rolling(60).std()
    df['vol_ratio_regime'] = df['vol_20'] / (df['vol_60'] + 1e-8)
    df['price_sma50_ratio'] = df['Close'] / (df['sma_50'] + 1e-8)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)
    df['return_60d'] = df['Close'].pct_change(60)
    df['volume_ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    df['volume_change'] = df['Volume'].pct_change(1)
    df['hl_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)

    # Scale-invariant V2 ratios
    df['ema_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['price_ema9_ratio'] = df['Close'] / (df['ema_9'] + 1e-8)
    df['price_ema21_ratio'] = df['Close'] / (df['ema_21'] + 1e-8)
    df['ema9_ema21_ratio'] = df['ema_9'] / (df['ema_21'] + 1e-8)
    df['stoch_signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
    
    macd_signal = ta.trend.macd_signal(df['Close'])
    df['macd_signal_norm'] = macd_signal / (df['Close'] + 1e-8)

    # Backfill, fillna, and replace inf/-inf with 0.0
    df = df.bfill().fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)
    return df

def predict_signals(candles: list, model_type: str = None) -> list:
    """
    Take raw candle dicts from database/API, calculate features, scale them,
    and predict buy/sell/hold signals with probabilities.
    Supports both TFT and LSTM architectures transparently.
    Calculates volatility-adjusted Stop Loss (SL) and Take Profit (TP) levels.
    """
    global _model_type, _threshold, _seq_len, _feature_cols
    
    model, scaler = get_model_and_scaler(model_type_override=model_type)
    if not model:
        logger.error("Model prediction skipped: no model loaded (type: %s).", model_type)
        return []
        
    if len(candles) < 60:
        return []
        
    # Convert list of candles to DataFrame
    df = pd.DataFrame(candles)
    # Ensure correct column names
    df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    
    # Convert unix timestamps to pandas datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], unit='s')
    else:
        df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')

    # Compute 14-period ATR for volatility-adjusted stop loss calculations
    atr_series = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range().bfill().fillna(0.0)

    # Compute features dynamically based on model type
    if _model_type == 'tft':
        df_feat = compute_all_tft_features(df)
        feat_df = df_feat[_feature_cols]
    else:
        df_feat = compute_lstm_features(df)
        feat_order = [
            'open', 'high', 'low', 'close', 'volume', 'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist', 
            'stoch_k', 'stoch_d', 'williams_r', 'ema_9', 'ema_21', 'ema_50', 'sma_20', 'adx', 'cci_20', 
            'bb_upper', 'bb_lower', 'bb_bw', 'atr_14', 'atr_7', 'vol_20', 'vol_60', 'obv', 'mfi_14', 
            'vol_ratio', 'log_return', 'hl_spread', 'price_ema50_r'
        ]
        feat_df = df_feat[feat_order]
        
    # Scale data
    if scaler is not None:
        try:
            scaled_data = scaler.transform(feat_df)
            if _model_type == 'tft':
                scaled_data = np.clip(scaled_data, -5, 5)
        except Exception as e:
            logger.error("Scaler transform failed, using simple MinMaxScaler fallback: %s", e)
            from sklearn.preprocessing import MinMaxScaler
            fallback_scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_data = fallback_scaler.fit_transform(feat_df.values)
    else:
        from sklearn.preprocessing import MinMaxScaler
        fallback_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = fallback_scaler.fit_transform(feat_df.values)
        
    predictions = []
    
    with torch.no_grad():
        # Start predictions from index 59 (60th candle onwards) as required by client and test suite
        for i in range(59, len(scaled_data)):
            seq = scaled_data[i-_seq_len+1:i+1] # shape (_seq_len, num_features)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            if _model_type == 'tft':
                logits, _ = model(seq_tensor)
                p = torch.sigmoid(logits).cpu().numpy()[0]
                
                buy_prob = float(p)
                sell_prob = 1.0 - float(p)
                hold_prob = 0.0
                
                # Confidence Threshold Filter (Alternative A): Only signal if p is confident
                upper_t = max(0.502, _threshold)
                lower_t = min(0.498, 1.0 - _threshold)
                
                if p > upper_t:
                    pred_class = 'buy'
                elif p < lower_t:
                    pred_class = 'sell'
                else:
                    pred_class = 'hold'
            else:
                # LSTM model
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                hold_prob = float(probs[0])
                buy_prob = float(probs[1])
                sell_prob = float(probs[2])
                
                # Confidence Threshold Filter (Alternative A) for LSTM: Require at least 34% probability for signals
                pred_class_idx = logits.argmax(dim=1).cpu().numpy()[0]
                if pred_class_idx == 1 and buy_prob < 0.34:
                    pred_class_idx = 0
                elif pred_class_idx == 2 and sell_prob < 0.34:
                    pred_class_idx = 0
                    
                signal_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                pred_class = signal_map[int(pred_class_idx)]
                
            candle_time = int(df['date'].iloc[i].timestamp())
            last_close = float(df['close'].iloc[i])
            atr_val = float(atr_series.iloc[i])
            
            predicted_pnl = (buy_prob - sell_prob) * 0.015
            predicted_price = round(last_close * (1.0 + predicted_pnl), 2)
            
            # Volatility-adjusted Stop Loss & Take Profit targets (1:2 Risk-Reward Ratio)
            if pred_class == 'buy':
                stop_loss = round(last_close - 1.5 * atr_val, 2)
                take_profit = round(last_close + 3.0 * atr_val, 2)
            elif pred_class == 'sell':
                stop_loss = round(last_close + 1.5 * atr_val, 2)
                take_profit = round(last_close - 3.0 * atr_val, 2)
            else:
                stop_loss = None
                take_profit = None
            
            predictions.append({
                'time': int(candle_time),
                'signal': pred_class,
                'probs': [hold_prob, buy_prob, sell_prob],
                'predicted_price': predicted_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
            })
            
    return predictions
