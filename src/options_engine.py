import numpy as np
from scipy.stats import norm

class GreeksCalculator:
    """Calculates Options Greeks based on Black-Scholes."""
    @staticmethod
    def d1_d2(S, K, T, r, sigma, q=0.0):
        # q is continuous dividend yield or foreign interest rate
        # For standard BSM without dividends, q=0
        if T <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def calculate_greeks(cls, S, K, T, r, sigma, opt_type='call', q=0.0):
        if T <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
            
        d1, d2 = cls.d1_d2(S, K, T, r, sigma, q)
        
        if opt_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            theta = (-np.exp(-q * T) * (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            theta = (-np.exp(-q * T) * (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))
                     
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        return {
            "delta": delta,
            "gamma": gamma,
            # Theta usually reported per day
            "theta": theta / 365,
            # Vega usually reported for 1% change in volatility
            "vega": vega / 100,
            "rho": rho / 100
        }

class BlackScholesPricer:
    """Pricing standard Equity options."""
    @staticmethod
    def price(S, K, T, r, sigma, opt_type='call'):
        d1, d2 = GreeksCalculator.d1_d2(S, K, T, r, sigma)
        if opt_type.lower() == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class GarmanKohlhagenPricer:
    """Pricing Forex options (continuous yield)."""
    @staticmethod
    def price(S, K, T, r_d, r_f, sigma, opt_type='call'):
        # r_d = domestic rate, r_f = foreign rate (yield)
        d1, d2 = GreeksCalculator.d1_d2(S, K, T, r_d, sigma, q=r_f)
        if opt_type.lower() == 'call':
            return S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)

class Black76Pricer:
    """Pricing Commodity/Futures options."""
    @staticmethod
    def price(F, K, T, r, sigma, opt_type='call'):
        # F is the forward/futures price. Note that S=F*exp(-rT) in standard BSM effectively.
        if T <= 0:
            return max(0.0, F - K) if opt_type.lower() == 'call' else max(0.0, K - F)
            
        d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if opt_type.lower() == 'call':
            return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

class MonteCarloSimulator:
    """Risk engine for simulating portfolio Value at Risk (VaR)."""
    @staticmethod
    def simulate_paths(S0, mu, sigma, T, steps=252, simulations=10000):
        """Simulate geometric brownian motion paths."""
        dt = T / steps
        # Random normal values for each step and simulation
        Z = np.random.normal(0, 1, (steps, simulations))
        
        # Precompute the drift and diffusion per step
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Calculate daily returns
        daily_returns = np.exp(drift + diffusion * Z)
        
        # Build price paths starting at S0
        paths = np.zeros_like(daily_returns)
        paths[0] = S0
        for t in range(1, steps):
            paths[t] = paths[t-1] * daily_returns[t]
            
        return paths

    @staticmethod
    def calculate_var(portfolio_value, simulated_final_values, confidence_level=0.99):
        """Calculate Value at Risk based on simulated outcomes."""
        simulated_pnl = simulated_final_values - portfolio_value
        var = np.percentile(simulated_pnl, (1 - confidence_level) * 100)
        cvar = simulated_pnl[simulated_pnl <= var].mean() # Expected Shortfall
        return {
            "VaR": abs(var),
            "CVaR": abs(cvar),
            "confidence": confidence_level
        }
