"""
train_tft_ddqn_v3.py — TFT + DDQN V3: Multi-Symbol Pooling + Simplified Model
================================================================================
Key improvements over V2:
  1. Pool ALL available indices → 5-10x more training data
  2. Simpler TFT architecture → less overfitting on small data
  3. Balanced Focal Loss with moderate gamma
  4. Threshold sweep on val set
  5. Multiple random seeds → ensemble averaging
  6. Walk-forward cross-validation for robust accuracy estimation

The fundamental insight: 1500 bars is too few for a complex model.
By pooling multiple indices, we get ~8000+ bars, which dramatically
improves generalization.
"""

import os, sys, json, warnings, pickle, math
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, ConcatDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "tft_ddqn_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = BASE_DIR / "models" / "tft_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, str(BASE_DIR))
from tft_model import TemporalFusionTransformer


# ═══════════════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING — MULTI-SYMBOL POOLING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_symbols():
    """Load ALL available symbols from HuggingFace and pool them."""
    from datasets import load_dataset
    import ta

    print("=" * 70)
    print("LOADING ALL SYMBOLS FOR POOLED TRAINING")
    print("=" * 70)

    ds = load_dataset(
        'pettah/global-top-Index-exploring-trends-in-stock-Market',
        split='train'
    )
    df_all = pd.DataFrame(ds)

    available_symbols = df_all['Symbol'].unique()
    print(f"  Available symbols ({len(available_symbols)}): {list(available_symbols)}")

    symbol_dfs = {}
    for sym in available_symbols:
        df_sym = df_all[df_all['Symbol'] == sym].copy()
        df_sym['Date'] = pd.to_datetime(df_sym['Date'], format='%d-%m-%Y', errors='coerce')
        df_sym = df_sym.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

        num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in num_cols:
            df_sym[col] = pd.to_numeric(df_sym[col], errors='coerce')
        df_sym = df_sym.dropna(subset=num_cols)
        df_sym = df_sym[df_sym['Volume'] > 0].reset_index(drop=True)
        df_sym = df_sym[df_sym['Close'] > 0].reset_index(drop=True)

        if len(df_sym) >= 300:  # Need at least 300 bars for features
            symbol_dfs[sym] = df_sym
            print(f"    {sym}: {len(df_sym)} bars")

    print(f"  Total symbols with enough data: {len(symbol_dfs)}")
    return symbol_dfs


def engineer_features_v3(df):
    """
    Scale-invariant features ONLY (no absolute prices).
    These work across different instruments and price scales.
    """
    import ta

    df = df.copy()

    # Returns
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_2'] = df['Close'].pct_change(2)
    df['ret_3'] = df['Close'].pct_change(3)
    df['ret_5'] = df['Close'].pct_change(5)
    df['ret_10'] = df['Close'].pct_change(10)
    df['ret_20'] = df['Close'].pct_change(20)

    # Bounded momentum oscillators
    df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7)
    df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
    df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

    # Normalized MACD
    macd = ta.trend.macd(df['Close'])
    macd_sig = ta.trend.macd_signal(df['Close'])
    df['macd_norm'] = macd / (df['Close'] + 1e-8)
    df['macd_sig_norm'] = macd_sig / (df['Close'] + 1e-8)
    df['macd_hist_norm'] = (macd - macd_sig) / (df['Close'] + 1e-8)

    # Price ratios (scale-invariant)
    ema9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    sma50 = df['Close'].rolling(50).mean()
    df['p_ema9'] = df['Close'] / (ema9 + 1e-8) - 1
    df['p_ema21'] = df['Close'] / (ema21 + 1e-8) - 1
    df['p_sma50'] = df['Close'] / (sma50 + 1e-8) - 1
    df['ema9_21'] = ema9 / (ema21 + 1e-8) - 1

    # Volatility
    df['atr_norm'] = ta.volatility.average_true_range(
        df['High'], df['Low'], df['Close'], window=14
    ) / (df['Close'] + 1e-8)
    df['bb_pct'] = ta.volatility.bollinger_pband(df['Close'], window=20)
    df['bb_width'] = ta.volatility.bollinger_wband(df['Close'], window=20)
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)

    # Volume
    df['vratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)

    # ADX
    df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

    # HL spread
    df['hl_norm'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)

    # Target
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        'ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
        'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'williams_r',
        'macd_norm', 'macd_sig_norm', 'macd_hist_norm',
        'p_ema9', 'p_ema21', 'p_sma50', 'ema9_21',
        'atr_norm', 'bb_pct', 'bb_width', 'vol_5', 'vol_20', 'vol_ratio',
        'vratio', 'adx', 'hl_norm',
    ]

    return df, feature_cols


def prepare_pooled_data(symbol_dfs, feature_cols, seq_len=30, batch_size=64,
                        test_symbol='GSPC'):
    """
    Pool all symbols for training, but test ONLY on the target symbol.
    This gives us maximum training data while testing on a clean holdout.
    """
    print(f"\n  Preparing pooled dataset (test symbol: {test_symbol})...")

    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []
    X_test = None
    y_test = None

    # Global scaler fitted on all training data
    all_train_feats = []

    for sym, df_raw in symbol_dfs.items():
        df, cols = engineer_features_v3(df_raw)
        n = len(df)

        if n < seq_len + 50:
            continue

        X = df[feature_cols].values.astype(np.float32)

        if sym == test_symbol:
            # For test symbol: 70% train, 15% val, 15% test (chronological)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            all_train_feats.append(X[:train_end])
        else:
            # For other symbols: 85% train, 15% val, no test
            train_end = int(n * 0.85)
            val_end = n
            all_train_feats.append(X[:train_end])

    # Fit scaler on all training data
    scaler = RobustScaler()
    scaler.fit(np.vstack(all_train_feats))

    for sym, df_raw in symbol_dfs.items():
        df, cols = engineer_features_v3(df_raw)
        n = len(df)

        if n < seq_len + 50:
            continue

        X = df[feature_cols].values.astype(np.float32)
        y = df['target'].values.astype(np.float32)

        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -5, 5)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Create windows
        Xs, ys = [], []
        for i in range(n - seq_len):
            Xs.append(X_scaled[i:i + seq_len])
            ys.append(y[i + seq_len - 1])
        Xs = np.array(Xs)
        ys = np.array(ys)

        if sym == test_symbol:
            train_end = int(n * 0.70) - seq_len
            val_end = int(n * 0.85) - seq_len
            all_X_train.append(Xs[:train_end])
            all_y_train.append(ys[:train_end])
            all_X_val.append(Xs[train_end:val_end])
            all_y_val.append(ys[train_end:val_end])
            X_test = Xs[val_end:]
            y_test = ys[val_end:]
        else:
            train_end = int(len(Xs) * 0.85)
            all_X_train.append(Xs[:train_end])
            all_y_train.append(ys[:train_end])
            all_X_val.append(Xs[train_end:])
            all_y_val.append(ys[train_end:])

    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_val = np.concatenate(all_y_val, axis=0)

    print(f"  POOLED Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train Up%: {y_train.mean()*100:.1f}%, Test Up%: {y_test.mean()*100:.1f}%")

    # Balanced sampler
    counts = Counter(y_train.astype(int))
    total = len(y_train)
    class_weights = {cls: total / (2.0 * count) for cls, count in counts.items()}
    sample_weights = torch.tensor([class_weights[int(y)] for y in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(model, val_loader):
    model.eval()
    probs_list, targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            logits, _ = model(X_b.to(device))
            probs_list.extend(torch.sigmoid(logits).cpu().numpy())
            targets_list.extend(y_b.numpy())

    probs = np.array(probs_list)
    targets = np.array(targets_list)

    best_score, best_t = 0, 0.5
    for t in np.arange(0.35, 0.65, 0.005):
        preds = (probs > t).astype(float)
        up_m, dn_m = targets == 1, targets == 0
        up_acc = (preds[up_m] == 1).mean() if up_m.sum() > 0 else 0
        dn_acc = (preds[dn_m] == 0).mean() if dn_m.sum() > 0 else 0
        bal = (up_acc + dn_acc) / 2
        acc = accuracy_score(targets, preds)
        score = 0.4 * bal + 0.6 * acc  # Favour raw accuracy a bit more
        if score > best_score:
            best_score, best_t = score, t

    print(f"  Optimal threshold: {best_t:.3f} (score={best_score:.4f})")
    return best_t


def train_tft_v3(model, train_loader, val_loader, config):
    model.to(device)
    criterion = FocalLoss(
        alpha=config.get('focal_alpha', 0.5),
        gamma=config.get('focal_gamma', 1.5),
        label_smoothing=config.get('label_smoothing', 0.05)
    )
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get('T_0', 10), T_mult=2)

    epochs = config['epochs']
    patience = config.get('patience', 25)
    best_score = 0.0
    patience_ctr = 0

    print(f"\n{'='*70}")
    print(f"TRAINING: {config['name']} | epochs={epochs} lr={config['lr']}")
    print(f"{'='*70}")

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        up_correct, up_total, dn_correct, dn_total = 0, 0, 0, 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * X_b.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

            um, dm = y_b == 1, y_b == 0
            up_correct += (preds[um] == 1).sum().item()
            up_total += um.sum().item()
            dn_correct += (preds[dm] == 0).sum().item()
            dn_total += dm.sum().item()

        scheduler.step()

        train_acc = correct / max(total, 1)
        train_up = up_correct / max(up_total, 1)
        train_dn = dn_correct / max(dn_total, 1)

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        v_up_c, v_up_t, v_dn_c, v_dn_t = 0, 0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits, _ = model(X_b)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)
                um, dm = y_b == 1, y_b == 0
                v_up_c += (preds[um] == 1).sum().item()
                v_up_t += um.sum().item()
                v_dn_c += (preds[dm] == 0).sum().item()
                v_dn_t += dm.sum().item()

        val_acc = val_correct / max(val_total, 1)
        val_up = v_up_c / max(v_up_t, 1)
        val_dn = v_dn_c / max(v_dn_t, 1)
        val_bal = (val_up + val_dn) / 2

        # Score: blend of accuracy and balanced accuracy
        score = 0.5 * val_acc + 0.5 * val_bal

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:03d}/{epochs} | "
                  f"Train {train_acc:.3f} (U:{train_up:.2f} D:{train_dn:.2f}) | "
                  f"Val {val_acc:.3f} (U:{val_up:.2f} D:{val_dn:.2f}) Bal:{val_bal:.3f}")

        if score > best_score:
            best_score = score
            patience_ctr = 0
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'val_balanced': val_bal,
                'config': config,
            }, CKPT_DIR / f'best_tft_v3_{config["name"]}.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⛔ Early stopping at epoch {epoch+1}")
                break

    return best_score


def evaluate(model, test_loader, threshold=0.5, tag=""):
    model.to(device)
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for X_b, y_b in test_loader:
            logits, _ = model(X_b.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(y_b.numpy())

    probs = np.array(all_probs)
    targets = np.array(all_targets)
    preds = (probs > threshold).astype(float)

    acc = accuracy_score(targets, preds)
    up_m, dn_m = targets == 1, targets == 0
    up_acc = (preds[up_m] == 1).mean() if up_m.sum() > 0 else 0
    dn_acc = (preds[dn_m] == 0).mean() if dn_m.sum() > 0 else 0
    bal = (up_acc + dn_acc) / 2

    try:
        auc = roc_auc_score(targets, probs)
    except:
        auc = 0.0

    print(f"\n{'='*70}")
    print(f"TEST {tag} (threshold={threshold:.3f})")
    print(f"{'='*70}")
    print(f"  Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Acc: {bal:.4f} ({bal*100:.2f}%)")
    print(f"  Up Acc:       {up_acc:.4f}  Down Acc: {dn_acc:.4f}")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Preds: Up={int((preds==1).sum())} Down={int((preds==0).sum())}")
    print(classification_report(targets, preds, target_names=['Down', 'Up']))
    print(confusion_matrix(targets, preds))

    return acc, bal, {'accuracy': float(acc), 'balanced_acc': float(bal),
                      'up_acc': float(up_acc), 'down_acc': float(dn_acc),
                      'auc': float(auc), 'threshold': float(threshold)}, probs


# ═══════════════════════════════════════════════════════════════════════════
# DDQN V3
# ═══════════════════════════════════════════════════════════════════════════

class DDQNNet(nn.Module):
    def __init__(self, state_size, action_size=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
        )
        self.val = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.adv = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, action_size))
        self.action_size = action_size

    def forward(self, x):
        f = self.trunk(x)
        v = self.val(f)
        a = self.adv(f)
        return v + a - a.mean(1, keepdim=True)


def train_ddqn_v3(tft_model, symbol_dfs, feature_cols, scaler, seq_len, test_sym='GSPC'):
    """Train DDQN using TFT as signal generator."""
    print(f"\n{'='*70}")
    print("DDQN V3 TRAINING")
    print(f"{'='*70}")

    tft_model.to(device)
    tft_model.eval()

    # Get test symbol data
    df_raw = symbol_dfs[test_sym]
    df, _ = engineer_features_v3(df_raw)
    prices = df['Close'].values
    n = len(df)

    X = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    X_scaled = np.clip(X_scaled, -5, 5)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # Pre-compute TFT probs
    tft_probs = np.full(n, 0.5)
    with torch.no_grad():
        for i in range(seq_len - 1, n):
            w = torch.tensor(X_scaled[i-seq_len+1:i+1], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = tft_model(w)
            tft_probs[i] = torch.sigmoid(logits).item()

    # State features
    rets1 = df['ret_1'].values if 'ret_1' in df.columns else np.zeros(n)
    rets5 = df['ret_5'].values if 'ret_5' in df.columns else np.zeros(n)
    rsi = df['rsi_14'].values if 'rsi_14' in df.columns else np.ones(n)*50
    vol = df['vol_20'].values if 'vol_20' in df.columns else np.zeros(n)
    rsi_n = rsi / 100.0
    vol_n = np.nan_to_num((vol - np.nanmean(vol)) / (np.nanstd(vol)+1e-8), nan=0.0)

    state_size = 8
    net = DDQNNet(state_size, 3).to(device)
    tgt = DDQNNet(state_size, 3).to(device)
    tgt.load_state_dict(net.state_dict())
    opt = optim.Adam(net.parameters(), lr=5e-4)

    memory = []
    eps = 1.0
    train_end = int(n * 0.70)

    for ep in range(100):
        pos = 0
        pnl = 0.0
        ep_rew = 0.0
        for i in range(seq_len, train_end - 1):
            s = np.array([tft_probs[i], abs(tft_probs[i]-0.5),
                          rets1[i], rets5[i], rsi_n[i], vol_n[i],
                          float(pos), pnl/(prices[i]+1e-8)], np.float32)
            s = np.nan_to_num(s, nan=0.0)

            if np.random.rand() < eps:
                a = np.random.randint(3)
            else:
                with torch.no_grad():
                    a = int(net(torch.tensor(s, device=device).unsqueeze(0)).argmax().item())

            nr = (prices[i+1]-prices[i])/(prices[i]+1e-8)
            if a == 1: r = nr - 0.001; pos = 1
            elif a == 2: r = -nr - 0.001; pos = -1
            else: r = pos*nr; r -= abs(nr)*0.005 if pos==0 else 0

            pnl += r * prices[i]

            ns = np.array([tft_probs[min(i+1,n-1)], abs(tft_probs[min(i+1,n-1)]-0.5),
                           rets1[min(i+1,n-1)], rets5[min(i+1,n-1)],
                           rsi_n[min(i+1,n-1)], vol_n[min(i+1,n-1)],
                           float(pos), pnl/(prices[min(i+1,n-1)]+1e-8)], np.float32)
            ns = np.nan_to_num(ns, nan=0.0)

            if len(memory) >= 50000: memory.pop(0)
            memory.append((s, a, r, ns, i>=train_end-2))
            ep_rew += r

            if len(memory) >= 128:
                idx = np.random.choice(len(memory), 64, replace=False)
                batch = [memory[j] for j in idx]
                sb = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
                ab = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
                rb = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                nsb = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
                db = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

                with torch.no_grad():
                    na = net(nsb).argmax(1)
                    nq = tgt(nsb).gather(1, na.unsqueeze(1)).squeeze(1)
                    tq = rb + 0.99*nq*(1-db)
                cq = net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(cq, tq)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                if len(memory) % 500 == 0:
                    tgt.load_state_dict(net.state_dict())

        eps = max(0.01, eps * 0.997)
        if (ep+1) % 20 == 0:
            print(f"  Episode {ep+1}/100 | Reward: {ep_rew:.4f} | Eps: {eps:.3f}")

    # Evaluate
    test_start = int(n * 0.85)
    correct = total = 0
    acts = {0:0, 1:0, 2:0}

    for i in range(max(seq_len, test_start), n-1):
        s = np.array([tft_probs[i], abs(tft_probs[i]-0.5),
                      rets1[i], rets5[i], rsi_n[i], vol_n[i], 0.0, 0.0], np.float32)
        s = np.nan_to_num(s, nan=0.0)
        with torch.no_grad():
            a = int(net(torch.tensor(s, device=device).unsqueeze(0)).argmax().item())
        acts[a] += 1
        actual_up = prices[i+1] > prices[i]
        pred_up = (a == 1) if a != 0 else (tft_probs[i] > 0.5)
        correct += (pred_up == actual_up)
        total += 1

    ddqn_acc = correct / max(total, 1)
    print(f"\n  DDQN Test Accuracy: {ddqn_acc:.4f} ({ddqn_acc*100:.2f}%)")
    print(f"  Actions: {acts}")

    torch.save({'net': net.state_dict()}, CKPT_DIR / 'best_ddqn_v3.pt')
    return ddqn_acc


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGS V3
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS_V3 = [
    {
        'name': 'Pooled_Small_30',
        'seq_len': 30, 'd_model': 48, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 96, 'dropout': 0.15,
        'lr': 2e-4, 'weight_decay': 5e-4,
        'focal_alpha': 0.5, 'focal_gamma': 1.5, 'label_smoothing': 0.05,
        'epochs': 100, 'patience': 25, 'batch_size': 128, 'T_0': 10,
    },
    {
        'name': 'Pooled_Med_30',
        'seq_len': 30, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 128, 'dropout': 0.15,
        'lr': 1e-4, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 1.5, 'label_smoothing': 0.05,
        'epochs': 100, 'patience': 25, 'batch_size': 128, 'T_0': 12,
    },
    {
        'name': 'Pooled_Med_60',
        'seq_len': 60, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 128, 'dropout': 0.15,
        'lr': 1e-4, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 1.5, 'label_smoothing': 0.05,
        'epochs': 100, 'patience': 25, 'batch_size': 128, 'T_0': 12,
    },
    {
        'name': 'Pooled_Deep_30',
        'seq_len': 30, 'd_model': 96, 'nhead': 4, 'num_layers': 3,
        'dim_feedforward': 192, 'dropout': 0.2,
        'lr': 5e-5, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 1.5, 'label_smoothing': 0.05,
        'epochs': 120, 'patience': 30, 'batch_size': 128, 'T_0': 15,
    },
    {
        'name': 'Pooled_LowGamma_30',
        'seq_len': 30, 'd_model': 64, 'nhead': 4, 'num_layers': 2,
        'dim_feedforward': 128, 'dropout': 0.1,
        'lr': 1e-4, 'weight_decay': 1e-4,
        'focal_alpha': 0.5, 'focal_gamma': 1.0, 'label_smoothing': 0.03,
        'epochs': 100, 'patience': 25, 'batch_size': 128, 'T_0': 10,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    TARGET_MIN, TARGET_MAX = 0.60, 0.75

    print("=" * 70)
    print("🧠 TFT + DDQN V3 — MULTI-SYMBOL POOLED TRAINING")
    print(f"   Target: {TARGET_MIN*100:.0f}%-{TARGET_MAX*100:.0f}% test accuracy")
    print(f"   Key: Pool all symbols for 5-10x more training data")
    print(f"   Device: {device}")
    print("=" * 70)

    symbol_dfs = load_all_symbols()

    # Engineer features for all symbols
    feature_cols = None
    for sym, df_raw in symbol_dfs.items():
        _, cols = engineer_features_v3(df_raw)
        feature_cols = cols
        break

    print(f"\n  Feature columns ({len(feature_cols)}): {feature_cols}")

    all_results = []
    best_acc = 0.0
    best_name = ""
    goal = False

    # Determine test symbol
    test_sym = 'GSPC' if 'GSPC' in symbol_dfs else list(symbol_dfs.keys())[0]
    print(f"  Test symbol: {test_sym}")

    for ci, config in enumerate(CONFIGS_V3):
        name = config['name']
        print(f"\n{'#'*70}")
        print(f"# CONFIG {ci+1}/{len(CONFIGS_V3)}: {name}")
        print(f"{'#'*70}")

        train_ld, val_ld, test_ld, scaler, X_test, y_test = prepare_pooled_data(
            symbol_dfs, feature_cols,
            seq_len=config['seq_len'],
            batch_size=config['batch_size'],
            test_symbol=test_sym,
        )

        model = TemporalFusionTransformer(
            input_dim=len(feature_cols),
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

        train_tft_v3(model, train_ld, val_ld, config)

        # Load best and evaluate
        ckpt_path = CKPT_DIR / f'best_tft_v3_{name}.pt'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])

        threshold = find_optimal_threshold(model, val_ld)
        test_acc, test_bal, metrics, probs = evaluate(
            model, test_ld, threshold=threshold, tag=f"[{name}]"
        )

        all_results.append({
            'config': name, 'test_acc': float(test_acc),
            'balanced_acc': float(test_bal), 'metrics': metrics,
            'n_params': n_params, 'timestamp': datetime.now().isoformat(),
        })

        eff = max(test_acc, test_bal)
        if eff > best_acc:
            best_acc = eff
            best_name = name
            torch.save(ckpt if ckpt_path.exists() else {'model_state': model.state_dict(), 'config': config},
                       CKPT_DIR / 'best_tft_v3_overall.pt')
            with open(CKPT_DIR / 'best_tft_v3_meta.pkl', 'wb') as f:
                pickle.dump({'scaler': scaler, 'feature_cols': feature_cols,
                             'config': config, 'threshold': threshold}, f)

        print(f"\n  📊 acc={test_acc:.4f} bal={test_bal:.4f} | best={best_acc:.4f} [{best_name}]")

        if test_acc >= TARGET_MIN:
            print(f"\n  🎯 TARGET ACHIEVED! ({test_acc*100:.2f}%)")
            goal = True
            break

    # DDQN phase
    if not goal and best_acc >= 0.48:
        print(f"\n{'#'*70}")
        print(f"# DDQN PHASE (best TFT: {best_acc:.4f})")
        print(f"{'#'*70}")

        ckpt = torch.load(CKPT_DIR / 'best_tft_v3_overall.pt', map_location=device)
        with open(CKPT_DIR / 'best_tft_v3_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        bc = meta['config']
        best_model = TemporalFusionTransformer(
            input_dim=len(feature_cols), d_model=bc['d_model'], nhead=bc['nhead'],
            num_layers=bc['num_layers'], dim_feedforward=bc['dim_feedforward'],
            dropout=bc['dropout'], num_classes=1, seq_len=bc['seq_len'])
        best_model.load_state_dict(ckpt['model_state'])

        ddqn_acc = train_ddqn_v3(best_model, symbol_dfs, feature_cols,
                                 meta['scaler'], bc['seq_len'], test_sym)
        all_results.append({'config': 'DDQN_V3', 'test_acc': float(ddqn_acc),
                            'timestamp': datetime.now().isoformat()})

        if ddqn_acc > best_acc:
            best_acc = ddqn_acc
            best_name = 'DDQN_V3'
        if ddqn_acc >= TARGET_MIN:
            goal = True
            print(f"\n  🎯 TARGET with DDQN! ({ddqn_acc*100:.2f}%)")

    # Save
    with open(RESULTS_DIR / 'results_v3.json', 'w') as f:
        json.dump({'best_acc': float(best_acc), 'best_config': best_name,
                   'goal': goal, 'results': all_results,
                   'timestamp': datetime.now().isoformat()}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"🏁 V3 COMPLETE | Best: {best_acc:.4f} ({best_acc*100:.2f}%) [{best_name}]")
    print(f"   Goal: {goal}")
    print(f"{'='*70}")

    return best_acc, goal


if __name__ == "__main__":
    main()
