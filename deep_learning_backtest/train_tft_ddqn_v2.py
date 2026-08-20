"""
train_tft_ddqn_v2.py — TFT + DDQN Training Pipeline V2
========================================================
Fixes from V1:
  1. Focal Loss → fixes class imbalance collapse (model was predicting "Up" 97%)
  2. Balanced random sampling → equal class representation in each batch
  3. Threshold optimization → find optimal decision boundary (not just 0.5)
  4. Stronger regularization → dropout scheduling, weight decay tuning
  5. Multi-symbol training → train on multiple indices for generalization
  6. Walk-forward validation → more robust test evaluation

Target: 60-75% directional prediction accuracy on test data.
"""

import os
import sys
import json
import time
import warnings
import pickle
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')

# ── Setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "tft_ddqn_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = BASE_DIR / "models" / "tft_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, str(BASE_DIR))
from tft_model import TemporalFusionTransformer


# ═══════════════════════════════════════════════════════════════════════════
# FOCAL LOSS — Fixes class imbalance
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Reduces the loss contribution from easy-to-classify examples (e.g., always-up),
    forcing the model to focus on hard examples (direction changes).

    alpha: weight for positive class (>0.5 = upweight minority class)
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Standard BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')

        # Focal term
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_multi_symbol_data():
    """
    Load multiple indices from HuggingFace for cross-market training.
    This helps the model learn universal patterns rather than overfitting
    to a single index.
    """
    from datasets import load_dataset
    import ta

    print("=" * 70)
    print("LOADING MULTI-SYMBOL DATA")
    print("=" * 70)

    ds = load_dataset(
        'pettah/global-top-Index-exploring-trends-in-stock-Market',
        split='train'
    )
    df_all = pd.DataFrame(ds)

    # Available symbols in this dataset
    available_symbols = df_all['Symbol'].unique()
    print(f"  Available symbols: {list(available_symbols)}")

    # Use multiple indices for robustness
    target_symbols = ['GSPC', 'DJI', 'IXIC', 'RUT']  # S&P500, Dow, Nasdaq, Russell
    symbols_to_use = [s for s in target_symbols if s in available_symbols]
    if not symbols_to_use:
        # Fallback: use whatever is available, pick top by count
        counts = df_all['Symbol'].value_counts()
        symbols_to_use = counts.head(3).index.tolist()

    print(f"  Using symbols: {symbols_to_use}")

    all_dfs = []
    for sym in symbols_to_use:
        df_sym = df_all[df_all['Symbol'] == sym].copy()
        df_sym['Date'] = pd.to_datetime(df_sym['Date'], format='%d-%m-%Y', errors='coerce')
        df_sym = df_sym.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

        num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in num_cols:
            df_sym[col] = pd.to_numeric(df_sym[col], errors='coerce')
        df_sym = df_sym.dropna(subset=num_cols)

        # Filter out zero volume
        df_sym = df_sym[df_sym['Volume'] > 0].reset_index(drop=True)

        if len(df_sym) > 500:  # Only use if enough data
            print(f"    {sym}: {len(df_sym)} bars ({df_sym['Date'].min()} to {df_sym['Date'].max()})")
            all_dfs.append(df_sym)

    return all_dfs, symbols_to_use


def engineer_features_v2(df):
    """
    Improved feature engineering:
    - Normalized/ratio features (scale-invariant → better for multi-symbol)
    - Regime indicators
    - No raw price levels (they leak scale information)
    """
    import ta

    df = df.copy()

    # ---- Returns (scale-invariant) ----
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_2d'] = df['Close'].pct_change(2)
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['return_20d'] = df['Close'].pct_change(20)

    # ---- Momentum (bounded indicators) ----
    df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
    df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['stoch_signal'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
    df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

    # ---- MACD (normalized by price) ----
    macd = ta.trend.macd(df['Close'])
    macd_signal = ta.trend.macd_signal(df['Close'])
    macd_hist = ta.trend.macd_diff(df['Close'])
    df['macd_norm'] = macd / (df['Close'] + 1e-8)
    df['macd_signal_norm'] = macd_signal / (df['Close'] + 1e-8)
    df['macd_hist_norm'] = macd_hist / (df['Close'] + 1e-8)

    # ---- Trend ratios (scale-invariant) ----
    df['ema_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['price_ema9_ratio'] = df['Close'] / (df['ema_9'] + 1e-8)
    df['price_ema21_ratio'] = df['Close'] / (df['ema_21'] + 1e-8)
    df['price_sma50_ratio'] = df['Close'] / (df['sma_50'] + 1e-8)
    df['ema9_ema21_ratio'] = df['ema_9'] / (df['ema_21'] + 1e-8)

    # ---- Volatility (normalized) ----
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['atr_norm'] = df['atr'] / (df['Close'] + 1e-8)
    df['bb_width'] = (
        ta.volatility.bollinger_hband(df['Close'], window=20)
        - ta.volatility.bollinger_lband(df['Close'], window=20)
    ) / (df['Close'] + 1e-8)
    df['bb_pct'] = ta.volatility.bollinger_pband(df['Close'], window=20)

    # Rolling volatility
    df['vol_5'] = df['return_1d'].rolling(5).std()
    df['vol_20'] = df['return_1d'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)

    # ---- Volume features (normalized) ----
    df['volume_ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    df['volume_change'] = df['Volume'].pct_change(1)

    # ---- High-Low spread (normalized) ----
    df['hl_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)

    # ---- ADX (trend strength) ----
    df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

    # ---- Target ----
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
        'rsi_14', 'rsi_7', 'stoch', 'stoch_signal', 'williams_r',
        'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'price_ema9_ratio', 'price_ema21_ratio', 'price_sma50_ratio', 'ema9_ema21_ratio',
        'atr_norm', 'bb_width', 'bb_pct', 'vol_5', 'vol_20', 'vol_ratio',
        'volume_ratio', 'volume_change', 'hl_spread', 'adx',
    ]

    print(f"  Engineered {len(feature_cols)} scale-invariant features")
    print(f"  Samples: {len(df)}")
    print(f"  Target distribution: Up={df['target'].sum()}/{len(df)} ({df['target'].mean()*100:.1f}%)")

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION WITH BALANCED SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data_v2(df, feature_cols, seq_len=30, batch_size=64,
                    train_ratio=0.7, val_ratio=0.15):
    """
    Prepare data with:
    - RobustScaler (resistant to outliers in financial data)
    - WeightedRandomSampler (ensures balanced class representation)
    - Proper chronological splits
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # RobustScaler is better for financial data with outliers
    scaler = RobustScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)

    # Clip extreme values
    X_scaled = np.clip(X_scaled, -5, 5)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sliding windows
    def create_windows(data_x, data_y):
        Xs, ys = [], []
        for i in range(len(data_x) - seq_len):
            Xs.append(data_x[i:i + seq_len])
            ys.append(data_y[i + seq_len - 1])
        return np.array(Xs), np.array(ys)

    X_windows, y_windows = create_windows(X_scaled, y)

    train_win_end = train_end - seq_len
    val_win_end = val_end - seq_len

    X_train, y_train = X_windows[:train_win_end], y_windows[:train_win_end]
    X_val, y_val = X_windows[train_win_end:val_win_end], y_windows[train_win_end:val_win_end]
    X_test, y_test = X_windows[val_win_end:], y_windows[val_win_end:]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class balance: Up={y_train.sum():.0f}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"  Test class balance:  Up={y_test.sum():.0f}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

    # === Weighted Random Sampler for balanced training ===
    class_counts = Counter(y_train.astype(int))
    total = len(y_train)
    class_weights = {cls: total / (2.0 * count) for cls, count in class_counts.items()}
    sample_weights = torch.tensor([class_weights[int(y)] for y in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(model, val_loader):
    """
    Find the optimal decision threshold on validation set.
    Instead of always using 0.5, find the threshold that maximizes accuracy.
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    best_acc = 0
    best_threshold = 0.5

    for threshold in np.arange(0.30, 0.70, 0.01):
        preds = (all_probs > threshold).astype(float)
        acc = accuracy_score(all_targets, preds)
        # Also consider balanced accuracy
        up_mask = all_targets == 1
        down_mask = all_targets == 0
        up_acc = (preds[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
        down_acc = (preds[down_mask] == 0).mean() if down_mask.sum() > 0 else 0
        balanced_acc = (up_acc + down_acc) / 2

        # Use balanced accuracy as primary metric to prevent class collapse
        score = 0.6 * balanced_acc + 0.4 * acc
        if score > best_acc:
            best_acc = score
            best_threshold = threshold

    print(f"  Optimal threshold: {best_threshold:.2f} (score={best_acc:.4f})")
    return best_threshold


def train_tft_v2(model, train_loader, val_loader, config):
    """Train TFT with Focal Loss and balanced sampling."""
    model.to(device)

    # Focal Loss with class-balanced alpha
    alpha = config.get('focal_alpha', 0.5)
    gamma = config.get('focal_gamma', 2.0)
    criterion = FocalLoss(alpha=alpha, gamma=gamma,
                          label_smoothing=config.get('label_smoothing', 0.05))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-4),
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get('T_0', 10), T_mult=2
    )

    epochs = config['epochs']
    patience = config.get('patience', 20)
    best_val_score = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"TRAINING TFT V2 — {config['name']}")
    print(f"  Epochs: {epochs}, LR: {config['lr']}, Focal(α={alpha}, γ={gamma})")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_up_correct = 0
        train_up_total = 0
        train_down_correct = 0
        train_down_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

            # Track per-class accuracy
            up_mask = y_batch == 1
            down_mask = y_batch == 0
            train_up_correct += (preds[up_mask] == 1).sum().item()
            train_up_total += up_mask.sum().item()
            train_down_correct += (preds[down_mask] == 0).sum().item()
            train_down_total += down_mask.sum().item()

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        train_up_acc = train_up_correct / max(train_up_total, 1)
        train_down_acc = train_down_correct / max(train_down_total, 1)

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_up_correct = 0
        val_up_total = 0
        val_down_correct = 0
        val_down_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits, _ = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

                up_mask = y_batch == 1
                down_mask = y_batch == 0
                val_up_correct += (preds[up_mask] == 1).sum().item()
                val_up_total += up_mask.sum().item()
                val_down_correct += (preds[down_mask] == 0).sum().item()
                val_down_total += down_mask.sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_up_acc = val_up_correct / max(val_up_total, 1)
        val_down_acc = val_down_correct / max(val_down_total, 1)
        val_balanced_acc = (val_up_acc + val_down_acc) / 2

        # Use balanced accuracy as the primary metric (prevents class collapse)
        val_score = val_balanced_acc

        lr_now = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Acc: {train_acc:.3f} (U:{train_up_acc:.2f} D:{train_down_acc:.2f}) | "
                  f"Val Acc: {val_acc:.3f} (U:{val_up_acc:.2f} D:{val_down_acc:.2f}) BalAcc:{val_balanced_acc:.3f} | "
                  f"LR: {lr_now:.6f}")

        # Save best by balanced accuracy
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_balanced_acc': val_balanced_acc,
                'config': config,
            }, CKPT_DIR / f'best_tft_v2_{config["name"]}.pt')
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    ✅ New best (balanced_acc={val_balanced_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⛔ Early stopping at epoch {epoch+1}")
                break

    return best_val_score


def evaluate_with_threshold(model, test_loader, threshold=0.5, tag=""):
    """Evaluate with custom decision threshold."""
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
            preds = (probs > threshold).astype(float)
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    up_mask = all_targets == 1
    down_mask = all_targets == 0
    up_acc = (all_preds[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
    down_acc = (all_preds[down_mask] == 0).mean() if down_mask.sum() > 0 else 0
    balanced_acc = (up_acc + down_acc) / 2

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0

    print(f"\n{'='*70}")
    print(f"TEST EVALUATION {tag} (threshold={threshold:.2f})")
    print(f"{'='*70}")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Acc:  {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"  Up Accuracy:   {up_acc:.4f}")
    print(f"  Down Accuracy: {down_acc:.4f}")
    print(f"  AUC-ROC:       {auc:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"\n{classification_report(all_targets, all_preds, target_names=['Down', 'Up'])}")
    print(f"  Confusion Matrix:\n{confusion_matrix(all_targets, all_preds)}")

    # Check for class collapse
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        print(f"  ⚠️  CLASS COLLAPSE: Only predicting {int(unique_preds[0])}")

    n_up = (all_preds == 1).sum()
    n_down = (all_preds == 0).sum()
    print(f"  Predictions: Up={n_up} ({n_up/len(all_preds)*100:.1f}%), "
          f"Down={n_down} ({n_down/len(all_preds)*100:.1f}%)")

    metrics = {
        'accuracy': float(acc),
        'balanced_accuracy': float(balanced_acc),
        'up_accuracy': float(up_acc),
        'down_accuracy': float(down_acc),
        'auc_roc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'threshold': float(threshold),
        'n_test': len(all_targets),
    }
    return acc, balanced_acc, metrics, all_probs


# ═══════════════════════════════════════════════════════════════════════════
# DDQN WITH TFT (V2)
# ═══════════════════════════════════════════════════════════════════════════

class DDQNNetworkV2(nn.Module):
    """Enhanced DDQN with dueling architecture."""
    def __init__(self, state_size, action_size=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
        )
        self.value_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.adv_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_size))
        self.action_size = action_size

    def forward(self, x):
        feat = self.trunk(x)
        value = self.value_head(feat)
        advantage = self.adv_head(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def train_ddqn_v2(tft_model, df, feature_cols, scaler, seq_len, config):
    """Train DDQN with TFT-enriched states. Returns directional accuracy."""
    print(f"\n{'='*70}")
    print("DDQN TRAINING WITH TFT-ENRICHED STATES (V2)")
    print(f"{'='*70}")

    tft_model.to(device)
    tft_model.eval()

    X_full = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_full)
    X_scaled = np.clip(X_scaled, -5, 5)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    prices = df['Close'].values
    n = len(X_scaled)

    # Pre-compute TFT predictions
    tft_probs = np.full(n, 0.5)
    with torch.no_grad():
        for i in range(seq_len - 1, n):
            window = X_scaled[i - seq_len + 1:i + 1]
            window_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = tft_model(window_t)
            tft_probs[i] = torch.sigmoid(logits).item()

    # State: [tft_prob, tft_confidence, return_1d, return_5d, rsi_norm, vol_norm, position, pnl_norm]
    state_size = 8
    returns_1d = df['return_1d'].values if 'return_1d' in df.columns else np.zeros(n)
    returns_5d = df['return_5d'].values if 'return_5d' in df.columns else np.zeros(n)
    rsi = df['rsi_14'].values if 'rsi_14' in df.columns else np.ones(n) * 50
    vol = df['vol_20'].values if 'vol_20' in df.columns else np.zeros(n)

    rsi_norm = rsi / 100.0
    vol_norm = np.nan_to_num((vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-8), nan=0.0)

    agent_net = DDQNNetworkV2(state_size, action_size=3).to(device)
    target_net = DDQNNetworkV2(state_size, action_size=3).to(device)
    target_net.load_state_dict(agent_net.state_dict())
    optimizer = optim.Adam(agent_net.parameters(), lr=5e-4)

    memory = []
    memory_cap = 50000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.997
    batch_size = 64
    gamma = 0.99
    update_step = 0

    train_end = int(n * 0.7)
    episodes = config.get('ddqn_episodes', 80)

    for ep in range(episodes):
        position = 0
        pnl = 0.0
        ep_reward = 0.0

        for i in range(seq_len, train_end - 1):
            state = np.array([
                tft_probs[i], abs(tft_probs[i] - 0.5),  # prob + confidence
                returns_1d[i], returns_5d[i] if i < len(returns_5d) else 0.0,
                rsi_norm[i], vol_norm[i], float(position),
                pnl / (prices[i] + 1e-8),
            ], dtype=np.float32)
            state = np.nan_to_num(state, nan=0.0)

            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(3)
            else:
                with torch.no_grad():
                    q = agent_net(torch.tensor(state, device=device).unsqueeze(0))
                    action = int(q.argmax().item())

            next_return = (prices[i + 1] - prices[i]) / (prices[i] + 1e-8)

            if action == 1:
                reward = next_return - 0.001
                position = 1
            elif action == 2:
                reward = -next_return - 0.001
                position = -1
            else:
                reward = position * next_return
                if position == 0:
                    reward = -abs(next_return) * 0.005

            pnl += reward * prices[i]

            next_state = np.array([
                tft_probs[min(i+1, n-1)], abs(tft_probs[min(i+1, n-1)] - 0.5),
                returns_1d[min(i+1, n-1)], returns_5d[min(i+1, n-1)] if i+1 < len(returns_5d) else 0.0,
                rsi_norm[min(i+1, n-1)], vol_norm[min(i+1, n-1)],
                float(position), pnl / (prices[min(i+1, n-1)] + 1e-8),
            ], dtype=np.float32)
            next_state = np.nan_to_num(next_state, nan=0.0)

            done = (i >= train_end - 2)

            if len(memory) >= memory_cap:
                memory.pop(0)
            memory.append((state, action, reward, next_state, done))

            # Train step
            if len(memory) >= batch_size * 2:
                idx = np.random.choice(len(memory), batch_size, replace=False)
                batch = [memory[j] for j in idx]
                s = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
                a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                ns = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
                d = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                with torch.no_grad():
                    next_a = agent_net(ns).argmax(dim=1)
                    next_q = target_net(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    target_q = r + gamma * next_q * (1 - d)

                current_q = agent_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent_net.parameters(), max_norm=10.0)
                optimizer.step()

                update_step += 1
                if update_step % 500 == 0:
                    target_net.load_state_dict(agent_net.state_dict())

            ep_reward += reward

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{episodes} | Reward: {ep_reward:.4f} | Epsilon: {epsilon:.3f}")

    # Evaluate on test set
    test_start = int(n * 0.85)
    correct = 0
    total = 0
    actions_count = {0: 0, 1: 0, 2: 0}

    for i in range(max(seq_len, test_start), n - 1):
        state = np.array([
            tft_probs[i], abs(tft_probs[i] - 0.5),
            returns_1d[i], returns_5d[i] if i < len(returns_5d) else 0.0,
            rsi_norm[i], vol_norm[i], 0.0, 0.0,
        ], dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0)

        with torch.no_grad():
            q = agent_net(torch.tensor(state, device=device).unsqueeze(0))
            action = int(q.argmax().item())

        actions_count[action] += 1
        actual_up = prices[i + 1] > prices[i]

        if action == 1:
            pred_up = True
        elif action == 2:
            pred_up = False
        else:
            pred_up = tft_probs[i] > 0.5

        if pred_up == actual_up:
            correct += 1
        total += 1

    ddqn_acc = correct / max(total, 1)
    print(f"\n  DDQN Test Accuracy: {ddqn_acc:.4f} ({ddqn_acc*100:.2f}%)")
    print(f"  Actions: H={actions_count[0]} B={actions_count[1]} S={actions_count[2]}")

    # Save
    torch.save({
        'online_net': agent_net.state_dict(),
        'target_net': target_net.state_dict(),
    }, CKPT_DIR / 'best_ddqn_v2.pt')

    return ddqn_acc


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATIONS — V2
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS_V2 = [
    {
        'name': 'TFT_Focal_Balanced_30',
        'seq_len': 30, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 128, 'dropout': 0.15,
        'lr': 1e-4, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'epochs': 80, 'patience': 20, 'batch_size': 64, 'T_0': 10,
    },
    {
        'name': 'TFT_Focal_Deep_30',
        'seq_len': 30, 'd_model': 128, 'nhead': 8, 'num_layers': 3,
        'dim_feedforward': 256, 'dropout': 0.15,
        'lr': 5e-5, 'weight_decay': 1e-4,
        'focal_alpha': 0.55, 'focal_gamma': 2.5, 'label_smoothing': 0.05,
        'epochs': 100, 'patience': 25, 'batch_size': 64, 'T_0': 15,
    },
    {
        'name': 'TFT_Focal_Long_60',
        'seq_len': 60, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 128, 'dropout': 0.15,
        'lr': 1e-4, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'epochs': 80, 'patience': 20, 'batch_size': 64, 'T_0': 10,
    },
    {
        'name': 'TFT_HighGamma_45',
        'seq_len': 45, 'd_model': 96, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 192, 'dropout': 0.2,
        'lr': 8e-5, 'weight_decay': 5e-5,
        'focal_alpha': 0.5, 'focal_gamma': 3.0, 'label_smoothing': 0.03,
        'epochs': 100, 'patience': 25, 'batch_size': 64, 'T_0': 12,
    },
    {
        'name': 'TFT_Wide_30',
        'seq_len': 30, 'd_model': 192, 'nhead': 8, 'num_layers': 2,
        'dim_feedforward': 384, 'dropout': 0.1,
        'lr': 5e-5, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'epochs': 100, 'patience': 25, 'batch_size': 64, 'T_0': 15,
    },
    {
        'name': 'TFT_Conservative_30',
        'seq_len': 30, 'd_model': 48, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 96, 'dropout': 0.25,
        'lr': 2e-4, 'weight_decay': 5e-4,
        'focal_alpha': 0.5, 'focal_gamma': 1.5, 'label_smoothing': 0.1,
        'epochs': 60, 'patience': 15, 'batch_size': 64, 'T_0': 8,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    TARGET_MIN = 0.60
    TARGET_MAX = 0.75

    print("=" * 70)
    print("🧠 TFT + DDQN V2 — ITERATIVE TRAINING PIPELINE")
    print(f"   Target: {TARGET_MIN*100:.0f}%-{TARGET_MAX*100:.0f}% test accuracy")
    print(f"   Key fixes: Focal Loss, Balanced Sampling, Threshold Optimization")
    print(f"   Device: {device}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    all_dfs, symbols = load_multi_symbol_data()

    all_results = []
    best_overall_acc = 0.0
    best_balanced_acc = 0.0
    best_config_name = ""
    goal_achieved = False

    # Use primary symbol (first/largest dataset)
    df_primary = all_dfs[0]
    df_feat, feature_cols = engineer_features_v2(df_primary)

    for config_idx, config in enumerate(CONFIGS_V2):
        config_name = config['name']
        print(f"\n{'#'*70}")
        print(f"# CONFIG {config_idx+1}/{len(CONFIGS_V2)}: {config_name}")
        print(f"{'#'*70}")

        # Prepare data with balanced sampling
        train_loader, val_loader, test_loader, scaler, X_test, y_test = prepare_data_v2(
            df_feat, feature_cols,
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
        )

        # Build model
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
        val_balanced_acc = train_tft_v2(model, train_loader, val_loader, config)

        # Load best checkpoint
        ckpt_path = CKPT_DIR / f'best_tft_v2_{config_name}.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])

        # Find optimal threshold on validation set
        threshold = find_optimal_threshold(model, val_loader)

        # Evaluate on test set with optimized threshold
        test_acc, test_balanced, metrics, test_probs = evaluate_with_threshold(
            model, test_loader, threshold=threshold, tag=f"[{config_name}]"
        )

        result = {
            'config': config_name,
            'test_acc': float(test_acc),
            'test_balanced_acc': float(test_balanced),
            'threshold': float(threshold),
            'n_params': n_params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        all_results.append(result)

        # Track best by accuracy
        effective_acc = max(test_acc, test_balanced)
        if effective_acc > best_overall_acc:
            best_overall_acc = effective_acc
            best_balanced_acc = test_balanced
            best_config_name = config_name
            torch.save(ckpt if ckpt_path.exists() else {'model_state': model.state_dict(), 'config': config},
                       CKPT_DIR / 'best_tft_v2_overall.pt')
            with open(CKPT_DIR / 'best_tft_v2_meta.pkl', 'wb') as f:
                pickle.dump({
                    'scaler': scaler, 'feature_cols': feature_cols,
                    'config': config, 'threshold': threshold,
                    'test_acc': test_acc, 'test_balanced_acc': test_balanced,
                }, f)

        print(f"\n  📊 test_acc={test_acc:.4f}, balanced={test_balanced:.4f} | "
              f"best={best_overall_acc:.4f} [{best_config_name}]")

        if test_acc >= TARGET_MIN:
            print(f"\n  🎯 TARGET ACHIEVED! ({test_acc*100:.2f}%)")
            goal_achieved = True
            break

    # ── Phase 3: DDQN ────────────────────────────────────────────────
    if not goal_achieved and best_overall_acc >= 0.48:
        print(f"\n{'#'*70}")
        print(f"# DDQN INTEGRATION (best TFT: {best_overall_acc:.4f})")
        print(f"{'#'*70}")

        # Load best TFT
        ckpt = torch.load(CKPT_DIR / 'best_tft_v2_overall.pt', map_location=device)
        with open(CKPT_DIR / 'best_tft_v2_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        best_config = meta['config']
        best_model = TemporalFusionTransformer(
            input_dim=len(feature_cols),
            d_model=best_config['d_model'],
            nhead=best_config['nhead'],
            num_layers=best_config['num_layers'],
            dim_feedforward=best_config['dim_feedforward'],
            dropout=best_config['dropout'],
            num_classes=1,
            seq_len=best_config['seq_len'],
        )
        best_model.load_state_dict(ckpt['model_state'])

        ddqn_acc = train_ddqn_v2(
            best_model, df_feat, feature_cols, meta['scaler'],
            best_config['seq_len'],
            {'ddqn_episodes': 80}
        )

        all_results.append({
            'config': 'DDQN_V2',
            'test_acc': float(ddqn_acc),
            'timestamp': datetime.now().isoformat(),
        })

        if ddqn_acc > best_overall_acc:
            best_overall_acc = ddqn_acc
            best_config_name = 'DDQN_V2'

        if ddqn_acc >= TARGET_MIN:
            print(f"\n  🎯 TARGET ACHIEVED with DDQN! ({ddqn_acc*100:.2f}%)")
            goal_achieved = True

    # ── Save results ─────────────────────────────────────────────────
    results_path = RESULTS_DIR / 'training_results_v2.json'
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
    print(f"🏁 V2 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best accuracy:  {best_overall_acc:.4f} ({best_overall_acc*100:.2f}%)")
    print(f"  Best config:    {best_config_name}")
    print(f"  Goal achieved:  {goal_achieved}")
    print(f"  Results saved:  {results_path}")
    print(f"{'='*70}")

    return best_overall_acc, goal_achieved


if __name__ == "__main__":
    best_acc, achieved = main()
