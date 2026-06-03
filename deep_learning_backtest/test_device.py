import torch
import torch.nn as nn

class LSTMTradingModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int = 3, dropout: float = 0.25):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is: {device}')

model = LSTMTradingModel(32, 64, 2, 3, 0.2).to(device)

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')

x = torch.randn(64, 60, 32).to(device)
out = model(x)
print(f'Output shape: {out.shape}')
