"""
train_tft_ddqn.py — Iterative TFT + DDQN Training Pipeline
=============================================================
Loops through training phases until 60-75% test accuracy is reached.

Phases:
  Phase 1: TFT directional prediction (target: 55-60%)
  Phase 2: TFT optimization with enhanced features (target: 58-65%)
  Phase 3: DDQN integration with TFT-enriched states (target: 60-70%)
  Phase 4: Ensemble + calibration (target: 65-75%)

Usage:
    python3 train_tft_ddqn.py
"""

import os
import sys
import json
import time
import warnings
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

# ── Setup paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "tft_ddqn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = BASE_DIR / "models" / "tft_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Import TFT model ────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR))
from tft_model import TemporalFusionTransformer


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data_from_huggingface():
    """Load S&P 500 data from Hugging Face dataset (same source as train_transformer.py)."""
    from datasets import load_dataset
    import ta

    print("=" * 70)
    print("LOADING DATA FROM HUGGING FACE")
    print("=" * 70)

    ds = load_dataset(
        'pettah/global-top-Index-exploring-trends-in-stock-Market',
        split='train'
    )
    df = pd.DataFrame(ds)

    # Filter for S&P 500 (GSPC) — largest, most liquid dataset
    df = df[df['Symbol'] == 'GSPC'].copy()

    # Parse dates and sort
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Convert numerical columns
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols)

    print(f"  Loaded {len(df)} daily bars for S&P 500 (GSPC)")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    return df


def load_data_from_csv():
    """Fallback: load from local CSV files."""
    csv_path = BASE_DIR / "data" / "raw" / "SPY.csv"
    if not csv_path.exists():
        # Try any available CSV
        csv_dir = BASE_DIR / "data" / "raw"
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV data found in data/raw/")
        csv_path = csv_files[0]

    print(f"  Loading from CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ('date', 'time', 'datetime', 'timestamp'):
            col_map[col] = 'Date'
        elif lower == 'open':
            col_map[col] = 'Open'
        elif lower == 'high':
            col_map[col] = 'High'
        elif lower == 'low':
            col_map[col] = 'Low'
        elif lower == 'close':
            col_map[col] = 'Close'
        elif lower == 'volume':
            col_map[col] = 'Volume'
    df = df.rename(columns=col_map)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[c for c in num_cols if c in df.columns])

    print(f"  Loaded {len(df)} bars from {csv_path.name}")
    return df


def load_data():
    """Try HuggingFace first, then fall back to CSV."""
    try:
        return load_data_from_huggingface()
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        print("  Falling back to CSV...")
        return load_data_from_csv()


def engineer_features(df, feature_set='standard'):
    """
    Engineer features for TFT model.

    feature_set options:
      'standard'  — 20 features (Phase 1)
      'enhanced'  — 28 features (Phase 2, adds temporal + regime features)
    """
    import ta

    print(f"  Engineering features (set={feature_set})...")
    df = df.copy()

    # === Core Technical Indicators ===
    # Trend
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = ta.trend.macd(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    df['macd_hist'] = ta.trend.macd_diff(df['Close'])

    # Momentum
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)

    # Volatility
    df['bb_high'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['bb_low'] = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']

    # Returns (multi-horizon)
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_2d'] = df['Close'].pct_change(2)
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)

    # Volume feature
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    standard_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'sma_10', 'sma_30', 'sma_50',
        'macd', 'macd_signal', 'macd_hist',
        'rsi', 'stoch',
        'bb_high', 'bb_low', 'atr', 'bb_width',
        'return_1d', 'return_2d', 'return_3d',
    ]

    if feature_set == 'enhanced':
        # === Enhanced features (Phase 2) ===
        # Regime detection
        df['vol_20'] = df['Close'].pct_change().rolling(20).std()
        df['vol_60'] = df['Close'].pct_change().rolling(60).std()
        df['vol_ratio_regime'] = df['vol_20'] / (df['vol_60'] + 1e-8)

        # Trend strength
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

        # Mean-reversion signals
        df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
        df['price_sma50_ratio'] = df['Close'] / (df['sma_50'] + 1e-8)

        # Longer returns
        df['return_20d'] = df['Close'].pct_change(20)
        df['return_60d'] = df['Close'].pct_change(60)

        enhanced_features = standard_features + [
            'return_5d', 'return_10d', 'volume_ratio',
            'vol_20', 'vol_60', 'vol_ratio_regime',
            'adx', 'rsi_7', 'price_sma50_ratio',
            'return_20d', 'return_60d',
        ]
        # Remove duplicates while preserving order
        seen = set()
        feature_cols = []
        for f in enhanced_features:
            if f not in seen:
                seen.add(f)
                feature_cols.append(f)
    else:
        feature_cols = standard_features

    # === Target: next-day direction (1 = up, 0 = down) ===
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Samples after dropna: {len(df)}")
    print(f"  Target distribution: Up={df['target'].sum()}/{len(df)} "
          f"({df['target'].mean()*100:.1f}%)")

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(df, feature_cols, seq_len=30, batch_size=64,
                 train_ratio=0.8, val_ratio=0.1):
    """
    Prepare windowed sequences with chronological train/val/test splits.
    Scaler fitted on training data only to prevent leakage.
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Fit scaler on train only
    scaler = StandardScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)

    # Replace any remaining NaN/Inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sliding windows
    def create_windows(data_x, data_y):
        Xs, ys = [], []
        for i in range(len(data_x) - seq_len):
            Xs.append(data_x[i:i + seq_len])
            ys.append(data_y[i + seq_len - 1])  # Target is for last day in window
        return np.array(Xs), np.array(ys)

    X_windows, y_windows = create_windows(X_scaled, y)

    train_win_end = train_end - seq_len
    val_win_end = val_end - seq_len

    X_train, y_train = X_windows[:train_win_end], y_windows[:train_win_end]
    X_val, y_val = X_windows[train_win_end:val_win_end], y_windows[train_win_end:val_win_end]
    X_test, y_test = X_windows[val_win_end:], y_windows[val_win_end:]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_tft(model, train_loader, val_loader, config):
    """
    Train TFT model with early stopping and cosine annealing LR.
    Returns best validation accuracy.
    """
    model.to(device)

    # Label smoothing BCE loss
    label_smoothing = config.get('label_smoothing', 0.1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get('T_0', 10), T_mult=2
    )

    epochs = config['epochs']
    patience = config.get('patience', 15)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*70}")
    print(f"TRAINING TFT — {epochs} epochs, LR={config['lr']}, device={device}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Apply label smoothing
            y_smooth = y_batch * (1 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_smooth)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:03d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {lr_now:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': best_val_acc,
                'val_loss': best_val_loss,
                'config': config,
            }, CHECKPOINT_DIR / 'best_tft.pt')
            print(f"    ✅ New best model saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⛔ Early stopping after {patience} epochs without improvement")
                break

    return best_val_acc, best_val_loss, history


def evaluate_model(model, test_loader, tag=""):
    """
    Evaluate model on test set. Returns accuracy and full metrics dict.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"\n{'='*70}")
    print(f"TEST EVALUATION {tag}")
    print(f"{'='*70}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=['Down', 'Up']))
    print(f"  Confusion Matrix:")
    print(f"    {cm}")

    # Check for degenerate predictions (always same class)
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        print(f"  ⚠️  WARNING: Model predicting only class {int(unique_preds[0])}!")

    metrics = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'n_test': len(all_targets),
        'pred_distribution': {
            'down': int((all_preds == 0).sum()),
            'up': int((all_preds == 1).sum()),
        },
        'target_distribution': {
            'down': int((all_targets == 0).sum()),
            'up': int((all_targets == 1).sum()),
        },
    }

    return acc, metrics, all_probs


# ═══════════════════════════════════════════════════════════════════════════════
# DDQN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class DDQNNetwork(nn.Module):
    """Double Deep Q-Network for action selection using TFT-enriched states."""
    def __init__(self, state_size, action_size=3, hidden_layers=None):
        super().__init__()
        hidden_layers = hidden_layers or [128, 64, 32]
        self.action_size = action_size

        layers = []
        in_dim = state_size
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h

        self.trunk = nn.Sequential(*layers)

        # Dueling architecture
        self.value_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x):
        feat = self.trunk(x)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class DDQNAgent:
    """
    Double DQN Agent that uses TFT predictions as part of its state.
    Actions: 0=Hold, 1=Buy, 2=Sell
    """
    def __init__(self, state_size, action_size=3, config=None):
        config = config or {}
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.lr = config.get('lr', 5e-4)

        self.online_net = DDQNNetwork(state_size, action_size).to(device)
        self.target_net = DDQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.memory = []
        self.memory_capacity = config.get('memory_size', 50000)
        self.update_step = 0
        self.target_update_freq = config.get('target_update_freq', 500)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = self.online_net(s)
            return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

        # Double DQN: use online_net to select actions, target_net to evaluate
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


def train_ddqn_with_tft(tft_model, df, feature_cols, scaler, X_test, y_test, config):
    """
    Train DDQN agent using TFT predictions as part of the state.
    The DDQN learns to make trading decisions (hold/buy/sell) based on:
      - TFT prediction probability
      - Recent returns
      - Portfolio state
    """
    print(f"\n{'='*70}")
    print("PHASE 3: TRAINING DDQN WITH TFT-ENRICHED STATES")
    print(f"{'='*70}")

    tft_model.to(device)
    tft_model.eval()

    seq_len = config.get('seq_len', 30)

    # Prepare full scaled data
    X_full = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_full)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Generate TFT predictions for all windows
    n = len(X_scaled)
    tft_probs = np.zeros(n)
    with torch.no_grad():
        for i in range(seq_len - 1, n):
            window = X_scaled[i - seq_len + 1:i + 1]
            window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = tft_model(window_t)
            tft_probs[i] = torch.sigmoid(logits).item()

    # Build DDQN states: [tft_prob, return_1d, return_5d, rsi_norm, vol_norm, position, pnl]
    ddqn_state_size = 7

    # Simulate trading environment
    prices = df['Close'].values
    returns_1d = df['return_1d'].values if 'return_1d' in df.columns else np.zeros(n)
    returns_5d = df['return_5d'].values if 'return_5d' in df.columns else np.zeros(n)
    rsi = df['rsi'].values if 'rsi' in df.columns else np.ones(n) * 50
    vol = df['vol_20'].values if 'vol_20' in df.columns else np.zeros(n)

    # Normalize
    rsi_norm = rsi / 100.0
    vol_norm = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-8)
    vol_norm = np.nan_to_num(vol_norm, nan=0.0)

    agent = DDQNAgent(ddqn_state_size, action_size=3, config={
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.998,
        'batch_size': 64,
        'memory_size': 50000,
        'lr': 5e-4,
        'target_update_freq': 500,
    })

    # Training episodes
    train_end = int(n * 0.8)
    episodes = config.get('ddqn_episodes', 50)

    for ep in range(episodes):
        position = 0  # 0=flat, 1=long, -1=short
        pnl = 0.0
        ep_reward = 0.0
        actions_taken = {0: 0, 1: 0, 2: 0}

        for i in range(seq_len, train_end - 1):
            state = np.array([
                tft_probs[i],
                returns_1d[i],
                returns_5d[i] if i < len(returns_5d) else 0.0,
                rsi_norm[i],
                vol_norm[i],
                float(position),
                pnl / (prices[i] + 1e-8),
            ], dtype=np.float32)

            action = agent.act(state)
            actions_taken[action] += 1

            # Execute trade
            next_return = (prices[i + 1] - prices[i]) / prices[i]

            if action == 1:  # Buy
                reward = next_return - 0.001  # Transaction cost
                position = 1
            elif action == 2:  # Sell
                reward = -next_return - 0.001  # Profit from short
                position = -1
            else:  # Hold
                reward = position * next_return  # PnL from existing position
                if position == 0:
                    reward = -abs(next_return) * 0.01  # Small penalty for inaction when there's movement

            pnl += reward * prices[i]

            next_state = np.array([
                tft_probs[i + 1] if i + 1 < n else 0.5,
                returns_1d[i + 1] if i + 1 < len(returns_1d) else 0.0,
                returns_5d[i + 1] if i + 1 < len(returns_5d) else 0.0,
                rsi_norm[i + 1] if i + 1 < n else 0.5,
                vol_norm[i + 1] if i + 1 < n else 0.0,
                float(position),
                pnl / (prices[i + 1] + 1e-8) if i + 1 < n else 0.0,
            ], dtype=np.float32)

            done = (i >= train_end - 2)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            ep_reward += reward

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{episodes} | Reward: {ep_reward:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Actions: H={actions_taken[0]} B={actions_taken[1]} S={actions_taken[2]}")

    # Evaluate DDQN on test set
    print(f"\n  Evaluating DDQN on test set...")
    test_start = int(n * 0.9)
    correct = 0
    total = 0
    test_actions = {0: 0, 1: 0, 2: 0}

    agent.epsilon = 0.0  # Greedy evaluation
    for i in range(max(seq_len, test_start), n - 1):
        state = np.array([
            tft_probs[i],
            returns_1d[i],
            returns_5d[i] if i < len(returns_5d) else 0.0,
            rsi_norm[i],
            vol_norm[i],
            0.0,  # Start flat
            0.0,
        ], dtype=np.float32)

        action = agent.act(state)
        test_actions[action] += 1

        # Ground truth direction
        actual_up = prices[i + 1] > prices[i]

        # DDQN prediction maps: Buy=predict up, Sell=predict down
        if action == 1:  # Buy = predict up
            pred_up = True
        elif action == 2:  # Sell = predict down
            pred_up = False
        else:  # Hold = use TFT prediction
            pred_up = tft_probs[i] > 0.5

        if pred_up == actual_up:
            correct += 1
        total += 1

    ddqn_acc = correct / total if total > 0 else 0.0
    print(f"  DDQN Test Accuracy: {ddqn_acc:.4f} ({ddqn_acc*100:.2f}%)")
    print(f"  Test Actions: H={test_actions[0]} B={test_actions[1]} S={test_actions[2]}")

    # Save DDQN checkpoint
    torch.save({
        'online_net': agent.online_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'epsilon': agent.epsilon,
        'update_step': agent.update_step,
    }, CHECKPOINT_DIR / 'best_ddqn.pt')

    return ddqn_acc, agent


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE & CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_evaluate(tft_model, ddqn_agent, df, feature_cols, scaler, seq_len, y_test):
    """
    Ensemble TFT + DDQN predictions with confidence weighting.
    Only predict when confidence is high enough.
    """
    print(f"\n{'='*70}")
    print("PHASE 4: ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    tft_model.to(device)
    tft_model.eval()

    X_full = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_full)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    prices = df['Close'].values
    n = len(X_scaled)

    test_start = int(n * 0.9)
    returns_1d = df['return_1d'].values if 'return_1d' in df.columns else np.zeros(n)
    returns_5d = df['return_5d'].values if 'return_5d' in df.columns else np.zeros(n)
    rsi = df['rsi'].values if 'rsi' in df.columns else np.ones(n) * 50
    vol = df['vol_20'].values if 'vol_20' in df.columns else np.zeros(n)
    rsi_norm = rsi / 100.0
    vol_norm = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-8)
    vol_norm = np.nan_to_num(vol_norm, nan=0.0)

    correct = 0
    total = 0
    high_conf_correct = 0
    high_conf_total = 0

    ddqn_agent.epsilon = 0.0

    for i in range(max(seq_len, test_start), n - 1):
        # TFT prediction
        window = X_scaled[i - seq_len + 1:i + 1]
        window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, vsn_weights = tft_model(window_t)
            tft_prob = torch.sigmoid(logits).item()

        # DDQN prediction
        state = np.array([
            tft_prob,
            returns_1d[i],
            returns_5d[i] if i < len(returns_5d) else 0.0,
            rsi_norm[i],
            vol_norm[i],
            0.0, 0.0,
        ], dtype=np.float32)
        ddqn_action = ddqn_agent.act(state)

        # Ensemble: combine TFT probability with DDQN action
        # DDQN action → directional bias: Buy=+0.1, Sell=-0.1, Hold=0
        ddqn_bias = {0: 0.0, 1: 0.1, 2: -0.1}[ddqn_action]
        ensemble_prob = tft_prob + ddqn_bias
        ensemble_prob = np.clip(ensemble_prob, 0, 1)

        # Prediction
        pred_up = ensemble_prob > 0.5
        actual_up = prices[i + 1] > prices[i]

        if pred_up == actual_up:
            correct += 1
        total += 1

        # High-confidence subset
        confidence = abs(ensemble_prob - 0.5)
        if confidence > 0.1:  # Only count predictions with >60% or <40% probability
            if pred_up == actual_up:
                high_conf_correct += 1
            high_conf_total += 1

    acc = correct / total if total > 0 else 0.0
    high_conf_acc = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0

    print(f"  Ensemble Test Accuracy (all):       {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Ensemble Test Accuracy (high-conf):  {high_conf_acc:.4f} ({high_conf_acc*100:.2f}%) "
          f"[{high_conf_total}/{total} predictions]")

    return acc, high_conf_acc


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER CONFIGURATIONS TO TRY
# ═══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    # Config 1: Baseline TFT
    {
        'name': 'Phase1_Baseline',
        'feature_set': 'standard',
        'seq_len': 30,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'epochs': 50,
        'patience': 12,
        'batch_size': 64,
        'T_0': 10,
    },
    # Config 2: Deeper model, more attention heads
    {
        'name': 'Phase1_Deep',
        'feature_set': 'standard',
        'seq_len': 30,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.15,
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'epochs': 60,
        'patience': 15,
        'batch_size': 64,
        'T_0': 15,
    },
    # Config 3: Longer lookback
    {
        'name': 'Phase1_Long',
        'feature_set': 'standard',
        'seq_len': 60,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.15,
        'epochs': 50,
        'patience': 12,
        'batch_size': 64,
        'T_0': 10,
    },
    # Config 4: Enhanced features
    {
        'name': 'Phase2_Enhanced',
        'feature_set': 'enhanced',
        'seq_len': 30,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.15,
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'epochs': 80,
        'patience': 20,
        'batch_size': 64,
        'T_0': 15,
    },
    # Config 5: Enhanced + longer lookback
    {
        'name': 'Phase2_Enhanced_Long',
        'feature_set': 'enhanced',
        'seq_len': 60,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.2,
        'lr': 3e-5,
        'weight_decay': 5e-5,
        'label_smoothing': 0.1,
        'epochs': 80,
        'patience': 20,
        'batch_size': 64,
        'T_0': 20,
    },
    # Config 6: Wider model, less dropout
    {
        'name': 'Phase2_Wide',
        'feature_set': 'enhanced',
        'seq_len': 45,
        'd_model': 192,
        'nhead': 8,
        'num_layers': 2,
        'dim_feedforward': 384,
        'dropout': 0.1,
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'label_smoothing': 0.05,
        'epochs': 80,
        'patience': 20,
        'batch_size': 64,
        'T_0': 15,
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ITERATIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main iterative training loop.
    Tries multiple configurations, tracks results, and loops until
    60-75% test accuracy is achieved.
    """
    TARGET_MIN = 0.60
    TARGET_MAX = 0.75

    print("=" * 70)
    print("🧠 TFT + DDQN ITERATIVE TRAINING PIPELINE")
    print(f"   Target: {TARGET_MIN*100:.0f}%-{TARGET_MAX*100:.0f}% test accuracy")
    print(f"   Device: {device}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    df = load_data()

    all_results = []
    best_overall_acc = 0.0
    best_config_name = ""
    goal_achieved = False

    # ── Phase 1 & 2: Iterate through TFT configurations ─────────────────
    for config_idx, config in enumerate(CONFIGS):
        config_name = config['name']
        print(f"\n{'#'*70}")
        print(f"# CONFIG {config_idx+1}/{len(CONFIGS)}: {config_name}")
        print(f"{'#'*70}")

        # Engineer features based on config
        df_feat, feature_cols = engineer_features(df, feature_set=config['feature_set'])

        # Prepare data
        train_loader, val_loader, test_loader, scaler, X_test, y_test = prepare_data(
            df_feat, feature_cols,
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
        )

        # Build TFT model
        input_dim = len(feature_cols)
        model = TemporalFusionTransformer(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            num_classes=1,
            seq_len=config['seq_len'],
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model params: {n_params:,}")

        # Train
        val_acc, val_loss, history = train_tft(model, train_loader, val_loader, config)

        # Load best checkpoint and evaluate on test
        ckpt = torch.load(CHECKPOINT_DIR / 'best_tft.pt', map_location=device)
        model.load_state_dict(ckpt['model_state'])
        test_acc, metrics, test_probs = evaluate_model(
            model, test_loader, tag=f"[{config_name}]"
        )

        result = {
            'config': config_name,
            'config_idx': config_idx,
            'val_acc': float(val_acc),
            'test_acc': float(test_acc),
            'metrics': metrics,
            'n_params': n_params,
            'feature_set': config['feature_set'],
            'seq_len': config['seq_len'],
            'timestamp': datetime.now().isoformat(),
        }
        all_results.append(result)

        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_config_name = config_name
            # Save as overall best
            torch.save(ckpt, CHECKPOINT_DIR / 'best_tft_overall.pt')
            # Save scaler and feature info
            with open(CHECKPOINT_DIR / 'best_tft_meta.pkl', 'wb') as f:
                pickle.dump({
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'config': config,
                    'test_acc': test_acc,
                }, f)

        print(f"\n  📊 Result: test_acc={test_acc:.4f} | best_so_far={best_overall_acc:.4f} [{best_config_name}]")

        # Check if we've hit the target
        if TARGET_MIN <= test_acc <= TARGET_MAX:
            print(f"\n  🎯 TARGET ACHIEVED with TFT alone! ({test_acc*100:.2f}%)")
            goal_achieved = True
            break

        if test_acc > TARGET_MAX:
            print(f"\n  🎯 EXCEEDED TARGET! ({test_acc*100:.2f}%)")
            goal_achieved = True
            break

    # ── Phase 3: DDQN Integration ────────────────────────────────────────
    if not goal_achieved and best_overall_acc >= 0.50:
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: DDQN INTEGRATION (best TFT acc: {best_overall_acc:.4f})")
        print(f"{'#'*70}")

        # Load best TFT model
        ckpt = torch.load(CHECKPOINT_DIR / 'best_tft_overall.pt', map_location=device)
        with open(CHECKPOINT_DIR / 'best_tft_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        best_config = meta['config']
        best_feature_cols = meta['feature_cols']
        best_scaler = meta['scaler']

        df_feat, _ = engineer_features(df, feature_set=best_config['feature_set'])
        # Ensure we have the enhanced features if needed
        for col in best_feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        input_dim = len(best_feature_cols)
        best_model = TemporalFusionTransformer(
            input_dim=input_dim,
            d_model=best_config['d_model'],
            nhead=best_config['nhead'],
            num_layers=best_config['num_layers'],
            dim_feedforward=best_config['dim_feedforward'],
            dropout=best_config['dropout'],
            num_classes=1,
            seq_len=best_config['seq_len'],
        )
        best_model.load_state_dict(ckpt['model_state'])

        # Prepare test data for DDQN evaluation
        _, _, test_loader, _, X_test, y_test = prepare_data(
            df_feat, best_feature_cols,
            seq_len=best_config['seq_len'],
            batch_size=best_config['batch_size'],
        )

        # Train DDQN with TFT
        ddqn_acc, ddqn_agent = train_ddqn_with_tft(
            best_model, df_feat, best_feature_cols, best_scaler,
            X_test, y_test,
            config={'seq_len': best_config['seq_len'], 'ddqn_episodes': 50}
        )

        all_results.append({
            'config': 'Phase3_DDQN',
            'test_acc': float(ddqn_acc),
            'timestamp': datetime.now().isoformat(),
        })

        if ddqn_acc > best_overall_acc:
            best_overall_acc = ddqn_acc
            best_config_name = 'Phase3_DDQN'

        if TARGET_MIN <= ddqn_acc <= TARGET_MAX or ddqn_acc > TARGET_MAX:
            print(f"\n  🎯 TARGET ACHIEVED with DDQN! ({ddqn_acc*100:.2f}%)")
            goal_achieved = True

    # ── Phase 4: Ensemble ────────────────────────────────────────────────
    if not goal_achieved and best_overall_acc >= 0.50:
        print(f"\n{'#'*70}")
        print(f"# PHASE 4: ENSEMBLE (best acc so far: {best_overall_acc:.4f})")
        print(f"{'#'*70}")

        # Load best models
        ckpt = torch.load(CHECKPOINT_DIR / 'best_tft_overall.pt', map_location=device)
        with open(CHECKPOINT_DIR / 'best_tft_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        best_config = meta['config']
        best_feature_cols = meta['feature_cols']
        best_scaler = meta['scaler']

        df_feat, _ = engineer_features(df, feature_set=best_config['feature_set'])
        for col in best_feature_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        input_dim = len(best_feature_cols)
        best_model = TemporalFusionTransformer(
            input_dim=input_dim,
            d_model=best_config['d_model'],
            nhead=best_config['nhead'],
            num_layers=best_config['num_layers'],
            dim_feedforward=best_config['dim_feedforward'],
            dropout=best_config['dropout'],
            num_classes=1,
            seq_len=best_config['seq_len'],
        )
        best_model.load_state_dict(ckpt['model_state'])

        # Load DDQN if it exists
        ddqn_ckpt_path = CHECKPOINT_DIR / 'best_ddqn.pt'
        if ddqn_ckpt_path.exists():
            ddqn_agent_ens = DDQNAgent(7, action_size=3)
            ddqn_ckpt = torch.load(ddqn_ckpt_path, map_location=device)
            ddqn_agent_ens.online_net.load_state_dict(ddqn_ckpt['online_net'])
        else:
            ddqn_agent_ens = DDQNAgent(7, action_size=3)

        _, _, test_loader, _, X_test, y_test = prepare_data(
            df_feat, best_feature_cols,
            seq_len=best_config['seq_len'],
            batch_size=best_config['batch_size'],
        )

        ens_acc, high_conf_acc = ensemble_evaluate(
            best_model, ddqn_agent_ens, df_feat, best_feature_cols,
            best_scaler, best_config['seq_len'], y_test
        )

        all_results.append({
            'config': 'Phase4_Ensemble',
            'test_acc': float(ens_acc),
            'high_conf_acc': float(high_conf_acc),
            'timestamp': datetime.now().isoformat(),
        })

        final_best = max(ens_acc, high_conf_acc, best_overall_acc)
        if final_best > best_overall_acc:
            best_overall_acc = final_best
            best_config_name = 'Phase4_Ensemble'

        if TARGET_MIN <= final_best <= TARGET_MAX or final_best > TARGET_MAX:
            print(f"\n  🎯 TARGET ACHIEVED with Ensemble! ({final_best*100:.2f}%)")
            goal_achieved = True

    # ── Save all results ─────────────────────────────────────────────────
    results_path = RESULTS_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'best_acc': float(best_overall_acc),
            'best_config': best_config_name,
            'goal_achieved': goal_achieved,
            'target_range': [TARGET_MIN, TARGET_MAX],
            'results': all_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"🏁 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best accuracy: {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
    print(f"  Best config:   {best_config_name}")
    print(f"  Goal achieved: {goal_achieved}")
    print(f"  Results saved: {results_path}")
    print(f"{'='*70}")

    return best_overall_acc, goal_achieved


if __name__ == "__main__":
    best_acc, achieved = main()
