import streamlit as st

st.set_page_config(
    page_title="Airbnb Price Optimizer",
    layout="wide"
)

st.title("🏠 Airbnb Pricing Optimization System")

st.markdown("""
### What this system does:
- Predict demand
- Optimize pricing
- Recommend actions
- Show business impact

👈 Use the sidebar to navigate between pages.
""")




"""
Current situation:
Current price = $100
Predicted demand = 5 bookings

👉 Revenue now:

100 × 5 = $500
After optimization:
Optimized price = $110
Predicted demand = 4.8 bookings

👉 New revenue:

110 × 4.8 = $528
Expected Improvement:
528 − 500 = +$28

👉 Meaning:

If you follow the recommendation, you are expected to earn $28 more
"""