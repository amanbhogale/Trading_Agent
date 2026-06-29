import numpy as np
import torch

class TransformerReplayBuffer:
    def __init__(self, capacity, obs_dim, seq_len=64, device='cuda'):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.device = device
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        # We need to sample sequences of length `seq_len`
        # To avoid wrapping around the buffer index improperly, we select valid end indices
        # Valid end index must be >= seq_len if buffer isn't full, or we just handle wrapping cautiously
        
        if self.size < self.seq_len:
            raise ValueError("Not enough data to sample a sequence.")
            
        valid_indices = []
        while len(valid_indices) < batch_size:
            idx = np.random.randint(self.seq_len, self.size)
            # Ensure the sequence doesn't cross the current pointer if it hasn't looped
            # For simplicity, if we haven't looped, ptr == size. Any idx from seq_len to size is valid
            # If we have looped, ptr is somewhere. We must avoid sequences that cross the ptr boundary.
            if self.size == self.capacity:
                if (idx - self.seq_len) < self.ptr <= idx:
                    continue
                    
            valid_indices.append(idx)
            
        obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        actions_batch = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rewards_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        dones_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        
        for i, idx in enumerate(valid_indices):
            obs_batch[i] = self.obs[idx - self.seq_len : idx]
            actions_batch[i] = self.actions[idx - self.seq_len : idx]
            rewards_batch[i] = self.rewards[idx - self.seq_len : idx]
            next_obs_batch[i] = self.next_obs[idx - self.seq_len : idx]
            dones_batch[i] = self.dones[idx - self.seq_len : idx]
            
        return (
            torch.FloatTensor(obs_batch).to(self.device),
            torch.LongTensor(actions_batch).to(self.device),
            torch.FloatTensor(rewards_batch).to(self.device),
            torch.FloatTensor(next_obs_batch).to(self.device),
            torch.FloatTensor(dones_batch).to(self.device)
        )
