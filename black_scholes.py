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

# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from black_scholes import black_scholes_price, implied_volatility
from greeks import delta, gamma, vega, theta, rho
from utils import payoff_diagram

st.set_page_config(page_title="Black-Scholes Option Explorer", layout="wide")

with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/133811158?v=4", width=120)
    st.markdown("**Poulami Nandi**")
    st.markdown("Postdoc @ UPenn | Ex-IITK | AI/ML Researcher")
    st.markdown("[GitHub](https://github.com/Poulami-Nandi)")
    st.markdown("[Google Scholar](https://scholar.google.com/citations?user=cZVRJvwAAAAJ&hl=en)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/poulami-nandi/)")
    st.markdown("---")
    st.markdown("Configure your option parameters below")

st.title("📈 Black-Scholes Option Explorer")

S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
option_type = st.sidebar.selectbox("Option Type", options=['call', 'put'])
position = st.sidebar.selectbox("Position", options=['long', 'short'])

price = black_scholes_price(S, K, T, r, sigma, option_type)
del_val = delta(S, K, T, r, sigma, option_type)
gam_val = gamma(S, K, T, r, sigma)
veg_val = vega(S, K, T, r, sigma)
the_val = theta(S, K, T, r, sigma, option_type)
rho_val = rho(S, K, T, r, sigma, option_type)

st.subheader("Calculated Option Metrics")
st.metric(label="Option Price", value=f"${price:.4f}")
st.write({
    "Delta": round(del_val, 4),
    "Gamma": round(gam_val, 4),
    "Vega": round(veg_val, 4),
    "Theta": round(the_val, 4),
    "Rho": round(rho_val, 4)
})

# Payoff Diagram
st.subheader("Payoff Diagram at Expiry")
S_range = np.linspace(0.5*S, 1.5*S, 100)
payoffs = payoff_diagram(S_range, K, option_type, position)
fig, ax = plt.subplots()
ax.plot(S_range, payoffs)
ax.set_title(f"{position.capitalize()} {option_type.capitalize()} Payoff")
ax.set_xlabel("Stock Price at Expiry")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', linestyle='--')
st.pyplot(fig)

# requirements.txt
streamlit
numpy
scipy
matplotlib

# LICENSE
MIT License

Copyright (c) 2025 Poulami Nandi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
