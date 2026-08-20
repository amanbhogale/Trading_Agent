import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x

class TFT_DDQN(nn.Module):
    def __init__(self, obs_dim, action_dim, d_model=64, n_heads=4, num_layers=2, seq_len=64):
        super(TFT_DDQN, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Feature Embedder
        self.feature_embedding = nn.Linear(obs_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len*2)
        
        # Multi-Head Attention layers (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dueling Heads
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # x shape expected: [batch_size, seq_len, obs_dim]
        # If single step is passed, unsqueeze to shape [batch_size, 1, obs_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # 1. Embed features
        x = self.feature_embedding(x)
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. Pass through Transformer
        x = self.transformer_encoder(x)
        
        # 4. We only care about the output at the final timestep for Q-values
        # Alternatively, for sequence training we can output for all timesteps
        # We will output for all timesteps to match the buffer sequence shape
        # x shape: [batch_size, seq_len, d_model]
        
        value = self.value_stream(x)           # [batch_size, seq_len, 1]
        advantage = self.advantage_stream(x)   # [batch_size, seq_len, action_dim]
        
        # Dueling aggregation
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=2, keepdim=True))
        
        return q_values
