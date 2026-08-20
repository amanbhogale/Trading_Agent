import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BinanceTradingEnv(gym.Env):
    """
    Gymnasium environment for trading on Binance minute-level data.
    Designed to yield individual steps, leaving temporal windowing to the replay buffer/TFT.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, trading_fee=0.001):
        super(BinanceTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        
        # Features to expose to the agent
        self.feature_cols = ['log_return', 'volatility_20', 'volume_change', 'dist_sma_20']
        
        # Action space: 0 = Hold, 1 = Buy (Long), 2 = Sell (Short)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 
        # features + current position [1, 0, -1] for Long, Flat, Short
        obs_dim = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.trades = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random start point to prevent overfitting to the start of the series
        self.current_step = np.random.randint(0, int(self.max_steps * 0.8))
        self.position = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.trades = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].values
        obs = np.append(features, self.position).astype(np.float32)
        return obs

    def step(self, action):
        # Current price (close)
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute action and calculate transition costs
        prev_position = self.position
        
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1
        else:
            self.position = 0
            
        trade_penalty = 0.0
        if self.position != prev_position:
            # We made a trade, incur fee on the entire balance for simplicity
            trade_penalty = np.log(1 - self.trading_fee)
            self.trades += 1

        # Step forward
        self.current_step += 1
        next_price = self.df.iloc[self.current_step]['close']
        
        # Calculate Reward (Log Return of position)
        asset_log_return = np.log(next_price / current_price)
        
        # If long, we get asset return. If short, we get inverse. If flat, 0.
        step_reward = (self.position * asset_log_return) + trade_penalty
        
        # Update net worth approximately
        self.net_worth = self.net_worth * np.exp(step_reward)
        
        # Check termination
        terminated = False
        if self.net_worth <= self.initial_balance * 0.1: # Lost 90%
            terminated = True
            step_reward -= 1.0 # Huge penalty for bankruptcy
            
        truncated = self.current_step >= self.max_steps
        
        info = {
            'net_worth': self.net_worth,
            'position': self.position,
            'step': self.current_step,
            'price': next_price
        }
        
        return self._next_observation(), step_reward, terminated, truncated, info

    def render(self):
        pass
