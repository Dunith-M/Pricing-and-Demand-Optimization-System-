import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Price vs Demand Explorer")

df = pd.read_csv("artifacts/reports/inference/price_recommendations.csv")

listing_id = st.selectbox("Select Listing", df["listing_id"])

row = df[df["listing_id"] == listing_id].iloc[0]

current_price = row["current_price"]
base_demand = row["predicted_demand_before"]

# simulate curve
prices = np.linspace(current_price * 0.7, current_price * 1.3, 30)
elasticity = -1.2

demand = base_demand * (prices / current_price) ** elasticity
revenue = prices * demand

# ===== DEMAND CURVE =====
st.subheader("📉 Demand Curve")

fig, ax = plt.subplots()
ax.plot(prices, demand)
ax.set_xlabel("Price")
ax.set_ylabel("Demand")

st.pyplot(fig)

# ===== REVENUE CURVE =====
st.subheader("📊 Revenue Curve")

fig2, ax2 = plt.subplots()
ax2.plot(prices, revenue)

ax2.axvline(current_price, linestyle="--", label="Current Price")
ax2.axvline(row["optimized_price"], linestyle="--", label="Optimized Price")

ax2.legend()

st.pyplot(fig2)

# ===== INSIGHT =====
st.subheader("🔥 Key Insight")

st.metric("Current Price", f"{current_price:.2f}")
st.metric("Optimized Price", f"{row['optimized_price']:.2f}")
st.metric("Revenue Gain", f"{row['expected_improvement']:.2f}")