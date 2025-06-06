import numpy as np
from scipy.stats import norm

def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def implied_volatility(target_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma)
        sigma -= (price - target_price) / (v + 1e-8)
        if abs(price - target_price) < tol:
            return sigma
    return np.nan

# greeks.py
from black_scholes import d1_d2

def delta(S, K, T, r, sigma, option_type='call'):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)

def gamma(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def theta(S, K, T, r, sigma, option_type='call'):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == 'call':
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

def rho(S, K, T, r, sigma, option_type='call'):
    _, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# utils.py
def format_float(value, decimals=4):
    return round(value, decimals)

def payoff_diagram(S_range, K, option_type='call', position='long'):
    payoff = []
    for S in S_range:
        if option_type == 'call':
            val = max(S - K, 0)
        else:
            val = max(K - S, 0)
        payoff.append(val if position == 'long' else -val)
    return payoff
