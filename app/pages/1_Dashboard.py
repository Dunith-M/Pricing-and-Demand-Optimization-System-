import streamlit as st
import pandas as pd

st.title("📊 Price Recommendations Dashboard")

# Load data
df = pd.read_csv("artifacts/reports/inference/price_recommendations.csv")

# ===== KPI CARDS =====
total_listings = len(df)
total_gain = df["expected_improvement"].sum()
avg_price_change = ((df["optimized_price"] - df["current_price"]) / df["current_price"]).mean() * 100
changed_pct = (df["recommendation"] != "keep").mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Listings", total_listings)
col2.metric("Revenue Increase", f"{total_gain:.2f}")
col3.metric("Avg Price Change %", f"{avg_price_change:.2f}%")
col4.metric("% Listings Changed", f"{changed_pct:.2f}%")

st.divider()

# ===== FILTERS =====
st.subheader("🔍 Filters")

action_filter = st.multiselect(
    "Recommendation Type",
    options=df["recommendation"].unique(),
    default=df["recommendation"].unique()
)

price_range = st.slider(
    "Price Range",
    float(df["current_price"].min()),
    float(df["current_price"].max()),
    (float(df["current_price"].min()), float(df["current_price"].max()))
)

filtered_df = df[
    (df["recommendation"].isin(action_filter)) &
    (df["current_price"].between(price_range[0], price_range[1]))
]

# ===== TABLE =====
st.subheader("📋 Recommendations")

display_df = filtered_df[
    [
        "listing_id",
        "current_price",
        "optimized_price",
        "expected_improvement",
        "recommendation",
        "reason",
    ]
]

st.dataframe(display_df, use_container_width=True)