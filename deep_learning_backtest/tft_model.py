"""
tft_model.py — Temporal Fusion Transformer for Trading Signal Prediction
=========================================================================
Implements TFT-Lite architecture with:
  - Variable Selection Network (VSN) — learns per-timestep feature importance
  - Gated Residual Network (GRN) — non-linear processing with skip connections
  - Multi-Head Attention — temporal pattern discovery
  - Quantile output head — uncertainty estimation

Architecture designed for directional prediction (binary: up/down).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedLinearUnit(nn.Module):
    """GLU: applies sigmoid gating to control information flow."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN: core building block of TFT.
    Applies non-linear transformation with skip connection and gating.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection projection if dimensions don't match
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # Optional context integration
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
    """
    VSN: learns which input features are important at each timestep.
    Outputs softmax weights over features, then applies them.
    """
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Feature-level GRNs (process each feature independently)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

        # Variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim, hidden_dim, num_features, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        batch_size, seq_len, _ = x.shape

        # Calculate variable selection weights
        # Flatten temporal dim for weight computation
        flat_x = x.reshape(batch_size * seq_len, self.num_features)
        weights = self.softmax(self.weight_grn(flat_x))  # (batch*seq, num_features)
        weights = weights.reshape(batch_size, seq_len, self.num_features)

        # Process each feature through its own GRN
        processed_features = []
        for i in range(self.num_features):
            feat = x[:, :, i:i+1]  # (batch, seq_len, 1)
            feat_flat = feat.reshape(batch_size * seq_len, 1)
            processed = self.feature_grns[i](feat_flat)  # (batch*seq, hidden_dim)
            processed = processed.reshape(batch_size, seq_len, self.hidden_dim)
            processed_features.append(processed)

        # Stack and apply weights
        # processed_features: list of (batch, seq_len, hidden_dim)
        stacked = torch.stack(processed_features, dim=2)  # (batch, seq, num_feat, hidden)
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # (batch, seq, hidden)

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
    """
    TFT-Lite: Temporal Fusion Transformer for directional prediction.

    Architecture:
      Input → VSN → Positional Encoding → Multi-Head Attention → GRN → Output

    Output: binary classification logit (up/down direction).
    """
    def __init__(
        self,
        input_dim: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 1,  # Binary classification (single logit)
        seq_len: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_dim=input_dim,
            num_features=input_dim,
            hidden_dim=d_model,
            dropout=dropout,
        )

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 50)

        # 3. Multi-Head Attention (Transformer Encoder)
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

        # 4. Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            d_model, dim_feedforward, d_model, dropout
        )

        # 5. Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # 6. Layer norm for final output
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim) — raw feature sequences

        Returns:
            logits: (batch_size, 1) — directional prediction logit
            vsn_weights: (batch_size, seq_len, input_dim) — feature importance
        """
        # Variable Selection
        vsn_out, vsn_weights = self.vsn(x)  # (batch, seq, d_model)

        # Positional encoding
        vsn_out = self.pos_encoder(vsn_out)

        # Multi-Head Attention
        attn_out = self.transformer_encoder(vsn_out)  # (batch, seq, d_model)

        # Post-attention GRN (applied to last timestep)
        last_step = attn_out[:, -1, :]  # (batch, d_model)
        gated = self.post_attn_grn(last_step)  # (batch, d_model)
        gated = self.final_norm(gated)

        # Output
        logits = self.output_head(gated)  # (batch, 1)

        return logits.squeeze(-1), vsn_weights

    def predict_proba(self, x):
        """Get probability of up direction."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)
