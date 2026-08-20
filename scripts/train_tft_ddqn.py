import sys
import os
import torch
import numpy as np
import logging

# Add parent dir to path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.trading_system.rl.dataset import BinanceDataset
from src.trading_system.rl.env import BinanceTradingEnv
from src.trading_system.rl.buffer import TransformerReplayBuffer
from src.trading_system.rl.model import TFT_DDQN
from src.trading_system.rl.agent import DDQNAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Hardware config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    if device == 'cpu':
        logger.warning("CUDA is not available. Training a TFT on CPU will be extremely slow.")

    # 1. Dataset
    dataset = BinanceDataset()
    df = dataset.load_and_preprocess()
    logger.info(f"Dataset loaded. Total rows: {len(df)}")
    
    # 2. Environment
    env = BinanceTradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 3. Model & Agent
    seq_len = 64
    d_model = 64
    model = TFT_DDQN(obs_dim=obs_dim, action_dim=action_dim, d_model=d_model, seq_len=seq_len)
    target_model = TFT_DDQN(obs_dim=obs_dim, action_dim=action_dim, d_model=d_model, seq_len=seq_len)
    
    agent = DDQNAgent(model, target_model, device=device)
    
    # 4. Buffer
    buffer = TransformerReplayBuffer(capacity=10000, obs_dim=obs_dim, seq_len=seq_len, device=device)
    
    # 5. Training Loop Config
    epochs = 10
    batch_size = 32
    max_steps = 1000 # steps per epoch for demo purposes
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    logger.info("Starting training loop...")
    
    # Pre-fill buffer slightly
    obs, _ = env.reset()
    for _ in range(seq_len * 2):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    for epoch in range(epochs):
        obs, _ = env.reset()
        
        # Maintain a sliding window of observations to feed the agent
        obs_window = [obs for _ in range(seq_len)]
        
        epoch_reward = 0
        losses = []
        
        for step in range(max_steps):
            # Format window for TFT
            obs_tensor = torch.FloatTensor(np.array(obs_window)).unsqueeze(0).to(device)
            
            action = agent.select_action(obs_tensor, epsilon)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Update buffer
            buffer.add(obs, action, reward, next_obs, terminated)
            
            # Update sliding window
            obs_window.pop(0)
            obs_window.append(next_obs)
            
            obs = next_obs
            epoch_reward += reward
            
            # Train step
            if buffer.size >= batch_size + seq_len:
                batch = buffer.sample(batch_size)
                loss = agent.update(batch)
                losses.append(loss)
                
            if terminated or truncated:
                break
                
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        avg_loss = np.mean(losses) if losses else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs} | Reward: {epoch_reward:.4f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    main()
