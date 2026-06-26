import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up results directory
OUTPUT_DIR = "eda_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Fetch Data
tickers = {
    "US Market": "AAPL",
    "Indian Market": "RELIANCE.NS",
    "Crypto": "BTC-USD"
}

print("Downloading data...")
data = yf.download(list(tickers.values()), start="2020-01-01", end="2024-01-01")['Close']
data.dropna(inplace=True)

# Calculate daily log returns
returns = np.log(data / data.shift(1)).dropna()

# 2. Exploratory Data Analysis (EDA)
print("Generating EDA plots...")

# Plot 1: Return Distributions
plt.figure(figsize=(15, 5))
for i, ticker in enumerate(tickers.values(), 1):
    plt.subplot(1, 3, i)
    sns.histplot(returns[ticker], kde=True, bins=50)
    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Log Return")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "return_distributions.png"))
plt.close()

# Plot 2: Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Asset Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
plt.close()

# 3. Adding Uncertainty (Randomization via Bootstrapping)
# We will bootstrap the returns of one asset to simulate 100 possible future paths for the next 252 trading days.
print("Running Bootstrapping simulations...")
target_asset = "AAPL"
historical_returns = returns[target_asset].values
last_price = data[target_asset].iloc[-1]

num_simulations = 100
trading_days = 252

simulated_paths = np.zeros((trading_days, num_simulations))
simulated_paths[0] = last_price

for sim in range(num_simulations):
    # Randomize data by sampling historical returns with replacement (Bootstrapping)
    random_returns = np.random.choice(historical_returns, size=trading_days, replace=True)
    
    # Calculate price path
    price_path = [last_price]
    for r in random_returns:
        price_path.append(price_path[-1] * np.exp(r))
        
    simulated_paths[:, sim] = price_path[1:]

# Plot 3: Bootstrapped Price Paths
plt.figure(figsize=(10, 6))
plt.plot(simulated_paths, color='blue', alpha=0.1)
plt.title(f"{target_asset} - 100 Bootstrapped Price Paths (1 Year)")
plt.xlabel("Days into Future")
plt.ylabel("Simulated Price")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{target_asset}_bootstrapped_paths.png"))
plt.close()

print(f"Done! EDA and simulation plots have been saved to the '{OUTPUT_DIR}' directory.")
