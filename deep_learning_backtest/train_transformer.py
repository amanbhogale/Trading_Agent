import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import ta
import os

def load_and_preprocess():
    print("Loading dataset from Hugging Face...")
    ds = load_dataset('pettah/global-top-Index-exploring-trends-in-stock-Market', split='train')
    df = pd.DataFrame(ds)
    
    # Filter for S&P 500 (GSPC)
    df = df[df['Symbol'] == 'GSPC'].copy()
    
    # Parse dates and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # Convert numerical columns
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols)
    
    print(f"Loaded {len(df)} daily bars for S&P 500.")
    return df

def engineer_features(df):
    print("Engineering features...")
    # Trend Indicators
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # MACD
    df['macd'] = ta.trend.macd(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    
    # Momentum
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    
    # Volatility
    df['bb_high'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['bb_low'] = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Lags (daily returns)
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_2d'] = df['Close'].pct_change(2)
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    
    # Target: 1 if next day Close is higher than current Close, else 0
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop rows with NaNs resulting from indicators and target shift
    df = df.dropna().reset_index(drop=True)
    return df

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x) # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x) # (batch_size, seq_len, d_model)
        # Global pooling (take the mean over sequence length)
        out = out.mean(dim=1) # (batch_size, d_model)
        logits = self.fc(out) # (batch_size, 1)
        return logits.squeeze(-1)

def prepare_loaders(df, seq_len=30, batch_size=32):
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'sma_10', 'sma_30', 'sma_50', 'macd', 'macd_signal',
        'rsi', 'stoch', 'bb_high', 'bb_low', 'atr',
        'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d'
    ]
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Chronological splits (80% train, 10% val, 10% test)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    # Fit scaler on train only
    scaler = StandardScaler()
    scaler.fit(X[:train_end])
    X_scaled = scaler.transform(X)
    
    # Create sliding windows
    def create_windows(data_x, data_y):
        Xs, ys = [], []
        for i in range(len(data_x) - seq_len):
            Xs.append(data_x[i:i+seq_len])
            ys.append(data_y[i+seq_len-1]) # target on the last day of window
        return np.array(Xs), np.array(ys)
    
    X_windows, y_windows = create_windows(X_scaled, y)
    
    train_win_end = train_end - seq_len
    val_win_end = val_end - seq_len
    
    X_train, y_train = X_windows[:train_win_end], y_windows[:train_win_end]
    X_val, y_val = X_windows[train_win_end:val_win_end], y_windows[train_win_end:val_win_end]
    X_test, y_test = X_windows[val_win_end:], y_windows[val_win_end:]
    
    print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)}, Test windows: {len(X_test)}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(feature_cols), X_test, y_test

def train_model(model, train_loader, val_loader, epochs=15, lr=1e-4, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_transformer.pt')
            
    print(f"Best validation accuracy: {best_val_acc:.4f}")

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load('best_transformer.pt'))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    return acc

if __name__ == "__main__":
    df = load_and_preprocess()
    df = engineer_features(df)
    train_loader, val_loader, test_loader, input_dim, _, _ = prepare_loaders(df)
    
    model = TimeSeriesTransformer(input_dim=input_dim)
    train_model(model, train_loader, val_loader, epochs=15)
    evaluate_model(model, test_loader)
