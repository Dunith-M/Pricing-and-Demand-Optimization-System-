import streamlit as st
import pandas as pd

st.title("⚙️ Optimization Insights")

df = pd.read_csv("artifacts/reports/inference/price_recommendations.csv")

# ===== ACTION DISTRIBUTION =====
st.subheader("📊 Action Distribution")

action_counts = df["recommendation"].value_counts()

st.bar_chart(action_counts)

# ===== REVENUE IMPACT =====
st.subheader("💰 Revenue Impact")

before = df["predicted_revenue_before"].sum()
after = df["predicted_revenue_after"].sum()

st.metric("Revenue Before", f"{before:.2f}")
st.metric("Revenue After", f"{after:.2f}")
st.metric("Total Gain", f"{after - before:.2f}")

# ===== SUMMARY =====
st.subheader("📌 Summary")

st.write({
    "Total Listings": len(df),
    "Increase Count": int((df["recommendation"] == "increase").sum()),
    "Decrease Count": int((df["recommendation"] == "decrease").sum()),
    "Keep Count": int((df["recommendation"] == "keep").sum()),
})