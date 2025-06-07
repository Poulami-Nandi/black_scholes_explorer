# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from black_scholes import black_scholes_price, implied_volatility
from greeks import delta, gamma, vega, theta, rho
from utils import payoff_diagram

st.set_page_config(page_title="Black-Scholes Option Explorer", layout="wide")

st.markdown("A live dashboard to explore Black-Scholes Option.")

# Header with bio and image
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**👤 Created by:** Dr. Poulami Nandi  ")
    st.markdown("Physicist · Quant Researcher · Data Scientist")
    st.markdown("**🏛️ Affiliations:**  ")
    st.markdown("[University of Pennsylvania](https://live-sas-physics.pantheon.sas.upenn.edu/people/poulami-nandi) · "
                "[IIT Kanpur](https://www.iitk.ac.in/) · "
                "[IIT Gandhinagar](https://www.usief.org.in/home-institution-india/indian-institute-of-technology-gandhinagar/) · "
                "[UC Davis](https://www.ucdavis.edu/) · "
                "[TU Wien](http://www.itp.tuwien.ac.at/CPT/index.htm?date=201838&cats=xbrbknmztwd)")
    st.markdown("**📧 Email:** [nandi.poulami91@gmail.com](mailto:nandi.poulami91@gmail.com), [pnandi@sas.upenn.edu](mailto:pnandi@sas.upenn.edu)")
    st.markdown("**🔗 Links:**  ")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)  |  "
                "[GitHub](https://github.com/Poulami-Nandi)  |  "
                "[Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)")
with col2:
    st.image("https://github.com/Poulami-Nandi/IV_surface_analyzer/raw/main/images/own/own_image.jpg", caption="Dr. Poulami Nandi", use_container_width=True)

st.subheader("📘 Mathematical Formulation")

st.markdown("The **Black-Scholes formula** for pricing a European call option is:")

st.latex(r'''
C = S N(d_1) - K e^{-rT} N(d_2)
''')

st.markdown("Where:")

st.latex(r'''
d_1 = \frac{\ln(\frac{S}{K}) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
''')

st.markdown("""
- \( S \): Current stock price  
- \( K \): Strike price  
- \( T \): Time to maturity (in years)  
- \( r \): Risk-free interest rate  
- \( \sigma \): Volatility of the underlying asset  
- \( N(\cdot) \): Cumulative distribution function of the standard normal distribution
""")



with st.sidebar:
    st.markdown("Configure your option parameters below")

st.title("📈 Black-Scholes Option Explorer")

ticker = st.sidebar.text_input("Enter Stock Ticker (optional)", value="AAPL")
if ticker:
    try:
        stock_data = yf.Ticker(ticker)
        hist = stock_data.history(period="1mo")
        current_price = hist['Close'][-1]
        st.sidebar.write(f"Latest price for {ticker.upper()}: ${current_price:.2f}")
    except Exception as e:
        st.sidebar.error("Failed to fetch stock price")
        current_price = 100.0
else:
    current_price = 100.0

S = st.sidebar.number_input("Spot Price (S)", value=float(current_price))
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
fig1, ax1 = plt.subplots()
ax1.plot(S_range, payoffs)
ax1.set_title(f"{position.capitalize()} {option_type.capitalize()} Payoff")
ax1.set_xlabel("Stock Price at Expiry")
ax1.set_ylabel("Profit / Loss")
ax1.axhline(0, color='black', linestyle='--')
st.pyplot(fig1)

# Greeks vs Spot Price
st.subheader("Greek Sensitivities vs Spot Price")
S_vals = np.linspace(0.5*S, 1.5*S, 100)
deltas = [delta(s, K, T, r, sigma, option_type) for s in S_vals]
gammas = [gamma(s, K, T, r, sigma) for s in S_vals]
vegas = [vega(s, K, T, r, sigma) for s in S_vals]

fig2, ax2 = plt.subplots()
ax2.plot(S_vals, deltas, label='Delta')
ax2.plot(S_vals, gammas, label='Gamma')
ax2.plot(S_vals, vegas, label='Vega')
ax2.set_title("Greeks vs Spot Price")
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Sensitivity")
ax2.legend()
st.pyplot(fig2)

# Greeks vs Volatility
st.subheader("Greek Sensitivities vs Volatility")
sigmas = np.linspace(0.1, 0.6, 100)
deltas_v = [delta(S, K, T, r, s, option_type) for s in sigmas]
gammas_v = [gamma(S, K, T, r, s) for s in sigmas]
vegas_v = [vega(S, K, T, r, s) for s in sigmas]

fig3, ax3 = plt.subplots()
ax3.plot(sigmas, deltas_v, label='Delta')
ax3.plot(sigmas, gammas_v, label='Gamma')
ax3.plot(sigmas, vegas_v, label='Vega')
ax3.set_title("Greeks vs Volatility")
ax3.set_xlabel("Volatility (σ)")
ax3.set_ylabel("Sensitivity")
ax3.legend()
st.pyplot(fig3)

# Greeks vs Time to Maturity
st.subheader("Greek Sensitivities vs Time to Maturity")
T_vals = np.linspace(0.01, 2.0, 100)
deltas_t = [delta(S, K, t, r, sigma, option_type) for t in T_vals]
gammas_t = [gamma(S, K, t, r, sigma) for t in T_vals]
vegas_t = [vega(S, K, t, r, sigma) for t in T_vals]

fig4, ax4 = plt.subplots()
ax4.plot(T_vals, deltas_t, label='Delta')
ax4.plot(T_vals, gammas_t, label='Gamma')
ax4.plot(T_vals, vegas_t, label='Vega')
ax4.set_title("Greeks vs Time to Maturity")
ax4.set_xlabel("Time to Maturity (T)")
ax4.set_ylabel("Sensitivity")
ax4.legend()
st.pyplot(fig4)

# Implied Volatility Smile
st.subheader("Implied Volatility Smile")
K_range = np.linspace(0.5*S, 1.5*S, 50)
market_price = price
implied_vols = [implied_volatility(market_price, S, k, T, r, option_type) for k in K_range]

fig5, ax5 = plt.subplots()
ax5.plot(K_range, implied_vols)
ax5.set_title("Implied Volatility Smile")
ax5.set_xlabel("Strike Price (K)")
ax5.set_ylabel("Implied Volatility")
ax5.axvline(K, color='gray', linestyle='--', alpha=0.6)
st.pyplot(fig5)

# Volatility Surface
st.subheader("Implied Volatility Surface")
K_vals = np.linspace(0.8 * S, 1.2 * S, 25)
T_vals = np.linspace(0.1, 2.0, 25)
K_grid, T_grid = np.meshgrid(K_vals, T_vals)
IV_surface = np.array([
    [implied_volatility(price, S, k, t, r, option_type) for k in K_vals]
    for t in T_vals
])

fig6 = plt.figure(figsize=(10, 6))
ax6 = fig6.add_subplot(111, projection='3d')
surf = ax6.plot_surface(K_grid, T_grid, IV_surface, cmap='viridis', edgecolor='none')
ax6.set_title("Implied Volatility Surface")
ax6.set_xlabel("Strike Price (K)")
ax6.set_ylabel("Time to Maturity (T)")
ax6.set_zlabel("Implied Volatility")
fig6.colorbar(surf, shrink=0.5, aspect=10)
st.pyplot(fig6)

# Compare with Yahoo Finance Option Chain
st.subheader("Compare with Market Option Prices")
try:
    options_dates = stock_data.options
    if options_dates:
        selected_date = st.selectbox("Select Expiration Date", options_dates)
        opt_chain = stock_data.option_chain(selected_date)
        market_calls = opt_chain.calls[['strike', 'lastPrice']].dropna()
        market_calls['ModelPrice'] = market_calls['strike'].apply(lambda k: black_scholes_price(S, k, T, r, sigma, 'call'))

        fig7, ax7 = plt.subplots()
        ax7.plot(market_calls['strike'], market_calls['lastPrice'], label='Market Price')
        ax7.plot(market_calls['strike'], market_calls['ModelPrice'], label='BS Model Price')
        ax7.set_title("Call Option Prices: Market vs Model")
        ax7.set_xlabel("Strike Price")
        ax7.set_ylabel("Option Price")
        ax7.legend()
        st.pyplot(fig7)
except Exception as e:
    st.warning("Could not fetch option chain or compare with model due to: " + str(e))
