import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from black_scholes import black_scholes_price, implied_volatility
from greeks import delta, gamma, vega, theta, rho
from utils import payoff_diagram

st.set_page_config(page_title="Black-Scholes Option Explorer", layout="wide")
st.markdown(
    "A live dashboard to explore Black-Scholes Option."
)

# Create two columns
col1, col2 = st.columns([3, 1])  # Left wider for text, right for image

# 🧾 Left side: Bio and Links
with col1:
    st.markdown("**👤 Created by:** Dr. Poulami Nandi  \n"
                "Physicist · Quant Researcher · Data Scientist")

    st.markdown("**🏛️ Affiliations:**  \n"
                "[University of Pennsylvania](https://live-sas-physics.pantheon.sas.upenn.edu/people/poulami-nandi) · "
                "[IIT Kanpur](https://www.iitk.ac.in/) · "
                "[IIT Gandhinagar](https://www.usief.org.in/home-institution-india/indian-institute-of-technology-gandhinagar/) · "
                "[UC Davis](https://www.ucdavis.edu/) · "
                "[TU Wien](http://www.itp.tuwien.ac.at/CPT/index.htm?date=201838&cats=xbrbknmztwd)")

    st.markdown("**📧 Email:**  \n"
                "[nandi.poulami91@gmail.com](mailto:nandi.poulami91@gmail.com), "
                "[pnandi@sas.upenn.edu](mailto:pnandi@sas.upenn.edu)")

    st.markdown("**🔗 Links:**  \n"
                "[LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)  |  "
                "[GitHub](https://github.com/Poulami-Nandi)  |  "
                "[Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)")

# Right side: Image
with col2:
    st.image(
        "https://github.com/Poulami-Nandi/IV_surface_analyzer/raw/main/images/own/own_image.jpg",
        caption="Dr. Poulami Nandi",
        use_container_width=True
    )


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

# Greeks vs Stock Price
st.subheader("Greek Sensitivities vs Stock Price")
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
