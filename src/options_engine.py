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
    def calculate_options_var(current_spot, simulated_final_values, positions, r=0.05, iv=0.15, horizon_days=1, confidence_level=0.99):
        """
        Calculate VaR for an Options Portfolio.
        simulated_final_values: np.array of shape (simulations,) representing spot prices at horizon.
        positions: list of dicts: [{'type': 'call'|'put'|'future', 'strike': 25000, 'qty': 100, 'tte': 0.1}]
        horizon_days: how many days forward we are evaluating the risk (defaults to 1 day).
        """
        current_port_value = 0.0
        simulated_port_values = np.zeros_like(simulated_final_values, dtype=float)
        
        for pos in positions:
            opt_type = pos['type'].lower()
            qty = pos['qty']
            
            if opt_type == 'future':
                current_port_value += current_spot * qty
                simulated_port_values += simulated_final_values * qty
                continue
                
            K = pos['strike']
            tte = pos['tte']
            
            # Current value of this position
            c_price = BlackScholesPricer.price(current_spot, K, tte, r, iv, opt_type)
            current_port_value += c_price * qty
            
            # Value at the end of the risk horizon (decayed time to expiry)
            new_tte = max(tte - (horizon_days / 365.0), 1e-5)
            s_prices = BlackScholesPricer.price(simulated_final_values, K, new_tte, r, iv, opt_type)
            simulated_port_values += s_prices * qty
            
        simulated_pnl = simulated_port_values - current_port_value
        var = np.percentile(simulated_pnl, (1 - confidence_level) * 100)
        
        tail_losses = simulated_pnl[simulated_pnl <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var
        
        return {
            "VaR": abs(var) if var < 0 else 0.0,
            "CVaR": abs(cvar) if cvar < 0 else 0.0,
            "current_portfolio_value": current_port_value,
            "simulated_worst_case": current_port_value + (var if var < 0 else 0),
            "confidence": confidence_level
        }
