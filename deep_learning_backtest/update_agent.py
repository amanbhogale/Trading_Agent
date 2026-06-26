import nbformat
import json
import re

nb_path = '/home/zombie/Documents/Trading_Agent/deep_learning_backtest/04_dqn_agent.ipynb'
out_path = '/home/zombie/Documents/Trading_Agent/deep_learning_backtest/04_dqn_agent_rainbow.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

buffer_code = """from collections import namedtuple
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ['state','action','reward','next_state','done'])

class ActionBalancedReplayBuffer:
    def __init__(self, capacity: int, action_size: int = 3):
        self.capacity = capacity
        self.action_size = action_size
        self.buffer = []
        self.pos = 0
        self.action_indices = {a: [] for a in range(action_size)}
        
    def push(self, *args):
        state, action, reward, next_state, done = args
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            old_action = self.buffer[self.pos].action
            if self.pos in self.action_indices[old_action]:
                self.action_indices[old_action].remove(self.pos)
            self.buffer[self.pos] = Transition(*args)
            
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
        pass
        
    def __len__(self): return len(self.buffer)

print('✅ ActionBalancedReplayBuffer defined')
"""

net_code = """import torch
import torch.nn as nn
import torch.nn.functional as F

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

STATE_SIZE  = 15   
ACTION_SIZE = 3
HIDDEN      = [256, 128, 64]

online_net = DistributionalDuelingDQN(STATE_SIZE, ACTION_SIZE, HIDDEN).to(device)
target_net = DistributionalDuelingDQN(STATE_SIZE, ACTION_SIZE, HIDDEN).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

total_params = sum(p.numel() for p in online_net.parameters() if p.requires_grad)
print(online_net)
print(f'\\n🔢 Trainable parameters: {total_params:,}')
"""

agent_code = """class DQNAgent:
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
        self.scheduler   = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
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

    def train_step(self):
        if len(self.memory) < self.batch_size * 2:
            return 0.0

        states, actions, rewards, next_states, dones, weights, indices = \\
            self.memory.sample(self.batch_size)

        with torch.no_grad():
            # Next state probabilities
            next_probs = self.online_net(next_states) # (batch, action, atoms)
            next_q = (next_probs * self.support).sum(2)
            next_actions = next_q.argmax(1) # Double DQN action selection
            
            next_target_probs = self.target_net(next_states)
            # Select target probabilities for the argmax action
            next_target_probs = next_target_probs[range(self.batch_size), next_actions, :] # (batch, atoms)
            
            # Compute projection of Target distribution
            Tz = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.dz
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix case where l == u
            l[(u > 0) & (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) & (l == u)] += 1

            m = states.new_zeros(self.batch_size, self.n_atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.n_atoms), self.batch_size).long().unsqueeze(1).to(device)
            
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_target_probs * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_target_probs * (b - l.float())).view(-1))

        # Online probabilities
        online_probs = self.online_net(states)
        online_probs = online_probs[range(self.batch_size), actions, :]
        
        # Cross entropy loss
        loss = -torch.sum(m * torch.log(online_probs + 1e-8), dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_step += 1
        if self.update_step % self.target_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

print('✅ DQNAgent ready | Rainbow-Lite')
"""

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if 'class PrioritizedReplayBuffer:' in source:
            cell.source = buffer_code
        elif 'class DuelingDQN(nn.Module):' in source:
            cell.source = net_code
        elif 'class DQNAgent:' in source:
            cell.source = agent_code

with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Created {out_path}")
