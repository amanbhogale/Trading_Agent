import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DDQNAgent:
    def __init__(self, model, target_model, lr=1e-4, gamma=0.99, tau=0.005, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.policy_net = model.to(self.device)
        self.target_net = target_model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss() # Huber loss
        
    def select_action(self, obs_sequence, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(0, 3) # Random action
            
        with torch.no_grad():
            # obs_sequence shape: [1, seq_len, obs_dim]
            q_values = self.policy_net(obs_sequence)
            # We want the action for the LAST timestep in the sequence
            action = q_values[0, -1, :].argmax().item()
            return action
            
    def update(self, batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        # obs_batch shape: [batch_size, seq_len, obs_dim]
        # action_batch shape: [batch_size, seq_len]
        
        # 1. Get current Q values
        current_q = self.policy_net(obs_batch)
        # Gather Q values for the actions taken
        current_q_taken = current_q.gather(2, action_batch.unsqueeze(2)).squeeze(2)
        
        # 2. Get next Q values for Target (Double DQN logic)
        with torch.no_grad():
            # Select action with policy net
            next_q_policy = self.policy_net(next_obs_batch)
            best_next_actions = next_q_policy.argmax(dim=2, keepdim=True)
            
            # Evaluate action with target net
            next_q_target = self.target_net(next_obs_batch)
            next_q_values = next_q_target.gather(2, best_next_actions).squeeze(2)
            
            # Target Q = reward + gamma * next_q * (1 - done)
            target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
        # We only calculate loss on the last step of the sequence (or the whole sequence)
        # For sequence models, calculating loss over the whole sequence stabilizes training
        loss = self.loss_fn(current_q_taken, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in Transformers/RNNs
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()
