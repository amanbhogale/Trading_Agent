import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Colab Setup
cell_1 = """# Connect to Google Drive and Set up Environment
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
BASE_DIR = '/content/drive/MyDrive/Trading_Agent'
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create necessary directories in Drive
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Change directory
os.chdir(BASE_DIR)
print("✅ Connected to Drive and set working directory to:", os.getcwd())
"""

# Cell 2: Imports
cell_2 = """# Import necessary libraries
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pickle, yaml, os, time, random
from pathlib import Path
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Libraries imported. Using device: {device}')
"""

# Cell 3: Env
cell_3 = """# Trading Environment with Sharpe Reward Shaping
class TradingEnv(gym.Env):
    \"\"\"
    Custom Trading Environment.
    Reward shaping applied: Rolling Sharpe ratio.
    \"\"\"
    def __init__(self, df: pd.DataFrame, initial_balance=10000, max_steps=None,
                 reward_fn: str = 'sharpe', window=20):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_steps = max_steps if max_steps else len(df) - 1
        self.reward_fn = reward_fn
        self.window = window
        
        self.action_space = gym.spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.equity_curve = [self.balance]
        self.returns_hist = []
        return self._obs()

    def _obs(self):
        # Placeholder for actual feature extraction logic
        obs = np.zeros(15, dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df):
            return self._obs(), 0.0, True, False, {}
            
        current_price = self.df.iloc[self.current_step]['Close']
        prev_price = self.df.iloc[self.current_step - 1]['Close']
        
        # Simplistic PnL logic for demonstration
        step_return = (current_price - prev_price) / prev_price if prev_price > 0 else 0
        
        if action == 1: # Buy
            self.position = 1
        elif action == 2: # Sell
            self.position = -1
        else:
            self.position = 0 # Flat
            
        pnl = self.position * step_return
        self.returns_hist.append(pnl)
        
        # SHAPED REWARD: Rolling Sharpe
        if self.reward_fn == 'sharpe' and len(self.returns_hist) > 1:
            recent_returns = self.returns_hist[-self.window:]
            std = np.std(recent_returns) + 1e-6
            reward = np.mean(recent_returns) / std
        else:
            reward = pnl
            
        done = self.current_step >= self.max_steps
        return self._obs(), float(reward), done, False, {}
"""

# Cell 4: Buffer
cell_4 = """# Action-Balanced Replay Buffer
Transition = namedtuple('Transition', ['state','action','reward','next_state','done'])

class ActionBalancedReplayBuffer:
    def __init__(self, capacity: int, action_size: int = 3):
        self.capacity = capacity
        self.action_size = action_size
        self.buffer = []
        self.pos = 0
        self.action_indices = {a: [] for a in range(action_size)}
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            old_action = self.buffer[self.pos].action
            if self.pos in self.action_indices[old_action]:
                self.action_indices[old_action].remove(self.pos)
            self.buffer[self.pos] = Transition(state, action, reward, next_state, done)
            
        self.action_indices[action].append(self.pos)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        samples_per_action = batch_size // self.action_size
        batch_indices = []
        for a in range(self.action_size):
            avail_indices = self.action_indices[a]
            if len(avail_indices) >= samples_per_action:
                idx = np.random.choice(avail_indices, samples_per_action, replace=False)
                batch_indices.extend(idx)
            else:
                batch_indices.extend(avail_indices)
                
        while len(batch_indices) < batch_size:
            batch_indices.append(np.random.randint(len(self.buffer)))
            
        samples = [self.buffer[i] for i in batch_indices]
        
        batch = Transition(*zip(*samples))
        states      = torch.tensor(np.array(batch.state),      dtype=torch.float32, device=device)
        actions     = torch.tensor(np.array(batch.action),     dtype=torch.long,    device=device)
        rewards     = torch.tensor(np.array(batch.reward),     dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
        dones       = torch.tensor(np.array(batch.done),       dtype=torch.float32, device=device)
        
        weights = torch.ones(batch_size, dtype=torch.float32, device=device)
        indices = batch_indices
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors: np.ndarray):
        pass # Not used in balanced buffer
        
    def __len__(self): return len(self.buffer)
"""

# Cell 5: Net
cell_5 = """# Distributional Dueling DQN (C51)
class DistributionalDuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int,
                 hidden_layers: list = [256, 128, 64],
                 n_atoms=51, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
        layers = []
        in_dim = state_size
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.1)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        
        self.value_fc = nn.Linear(hidden_layers[-1], 64)
        self.value_out = nn.Linear(64, n_atoms)
        
        self.adv_fc = nn.Linear(hidden_layers[-1], 64)
        self.adv_out = nn.Linear(64, action_size * n_atoms)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        feat = self.trunk(x)
        
        V = self.value_out(F.relu(self.value_fc(feat))) 
        V = V.view(batch_size, 1, self.n_atoms)
        
        A = self.adv_out(F.relu(self.adv_fc(feat)))
        A = A.view(batch_size, self.action_size, self.n_atoms)
        
        Q_logits = V + A - A.mean(dim=1, keepdim=True)
        Q_probs = F.softmax(Q_logits, dim=2)
        return Q_probs

    def act_greedy(self, state_np: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            probs = self.forward(s)
            expected_Q = (probs * self.support).sum(dim=2)
            return int(expected_Q.argmax(dim=1).item())
"""

# Cell 6: Agent
cell_6 = """# DQNAgent (Rainbow-Lite)
class DQNAgent:
    \"\"\"Rainbow-Lite Agent: Distributional Dueling C51 + Action-Balanced Replay\"\"\"
    def __init__(self, state_size, action_size, cfg):
        self.state_size  = state_size
        self.action_size = action_size
        self.gamma       = cfg['gamma']
        self.lr          = cfg['learning_rate']
        self.batch_size  = cfg['batch_size']
        self.target_freq = cfg['target_update_freq']

        self.epsilon      = cfg['epsilon_start']
        self.epsilon_min  = cfg['epsilon_end']
        self.epsilon_decay= cfg['epsilon_decay']

        self.memory = ActionBalancedReplayBuffer(cfg['memory_size'])

        self.online_net = DistributionalDuelingDQN(state_size, action_size, cfg['hidden_layers']).to(device)
        self.target_net = DistributionalDuelingDQN(state_size, action_size, cfg['hidden_layers']).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.update_step = 0
        self.v_min = self.online_net.v_min
        self.v_max = self.online_net.v_max
        self.n_atoms = self.online_net.n_atoms
        self.support = self.online_net.support
        self.dz = (self.v_max - self.v_min) / (self.n_atoms - 1)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return self.online_net.act_greedy(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        states, actions, rewards, next_states, dones, weights, indices = \\
            self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_probs = self.online_net(next_states)
            next_q = (next_probs * self.support).sum(2)
            next_actions = next_q.argmax(1) 
            
            next_target_probs = self.target_net(next_states)
            next_target_probs = next_target_probs[range(self.batch_size), next_actions, :] 
            
            Tz = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.dz
            l = b.floor().long()
            u = b.ceil().long()
            
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) & (l == u)] += 1

            m = states.new_zeros(self.batch_size, self.n_atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.n_atoms), self.batch_size).long().unsqueeze(1).to(device)
            
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_target_probs * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_target_probs * (b - l.float())).view(-1))

        online_probs = self.online_net(states)
        online_probs = online_probs[range(self.batch_size), actions, :]
        
        loss = -torch.sum(m * torch.log(online_probs + 1e-8), dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        import torch
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
            'update_step': self.update_step,
        }, path)

    def load(self, path):
        import torch
        ckpt = torch.load(path, map_location=device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon     = ckpt['epsilon']
        self.update_step = ckpt['update_step']
"""

# Cell 7: Config
cell_7 = """# Configuration and Data Preparation
cfg = {
    'gamma': 0.99,
    'learning_rate': 0.0005,
    'batch_size': 64,
    'target_update_freq': 500,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'memory_size': 50000,
    'hidden_layers': [256, 128, 64]
}

# Save Config to Drive
with open(os.path.join(CONFIG_DIR, 'rainbow_dqn_cfg.yaml'), 'w') as f:
    yaml.dump(cfg, f)

STATE_SIZE = 15
ACTION_SIZE = 3

# Mock Data (Replace with loading real CSV from DATA_DIR)
# e.g., df = pd.read_csv(os.path.join(DATA_DIR, 'btc_data.csv'))
dates = pd.date_range('2023-01-01', periods=1000, freq='H')
prices = np.linspace(100, 150, 1000) + np.random.normal(0, 1, 1000)
df = pd.DataFrame({'Close': prices}, index=dates)

train_env = TradingEnv(df[:800])
eval_env = TradingEnv(df[800:])

agent = DQNAgent(STATE_SIZE, ACTION_SIZE, cfg)
print(f'✅ Agent & Environment initialized. Ready to train.')
"""

# Cell 8: Loop
cell_8 = """# Training Loop
EPISODES = 50
MAX_STEPS = len(df[:800])

best_eval_return = -np.inf

for ep in range(EPISODES):
    state = train_env.reset()
    ep_reward = 0.0
    
    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, _, info = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        loss = agent.learn()
        
        ep_reward += reward
        state = next_state
        if done:
            break
            
    # Evaluation
    eval_state = eval_env.reset()
    eval_reward = 0.0
    for _ in range(eval_env.max_steps):
        # Always act greedy during eval
        with torch.no_grad():
            eval_action = agent.online_net.act_greedy(eval_state)
        next_state, r, d, _, _ = eval_env.step(eval_action)
        eval_reward += r
        eval_state = next_state
        if d: break
        
    print(f"Ep {ep+1}/{EPISODES} | Train Reward: {ep_reward:.2f} | Eval Reward: {eval_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Save best model to Google Drive
    if eval_reward > best_eval_return:
        best_eval_return = eval_reward
        agent.save(os.path.join(MODEL_DIR, 'best_rainbow_dqn.pth'))
        print(f"   [+] New best model saved to Drive! Eval Reward: {eval_reward:.2f}")

print("Training Complete!")
"""

nb.cells = [
    nbf.v4.new_code_cell(cell_1),
    nbf.v4.new_code_cell(cell_2),
    nbf.v4.new_code_cell(cell_3),
    nbf.v4.new_code_cell(cell_4),
    nbf.v4.new_code_cell(cell_5),
    nbf.v4.new_code_cell(cell_6),
    nbf.v4.new_code_cell(cell_7),
    nbf.v4.new_code_cell(cell_8)
]

output_path = '/home/zombie/Documents/Trading_Agent/deep_learning_backtest/Colab_Rainbow_DQN.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Successfully generated {output_path}")
