# =========================
# app.py
# =========================

import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("loan_model.pkl")
encoder = joblib.load("encoder.pkl")



# Page Config
st.set_page_config(
    page_title="Loan Approval System",
    layout="centered"
)

st.markdown("""
<style>
.stApp{
background: linear-gradient(
135deg,
#E3F2FD,
#F8F9FA,
#E8F5E9
);
}

.input-card{
    background: white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)



#=========================
# Title
#=========================

st.image(
    "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
    use_container_width=True
)

st.markdown("""
<h1 style='text-align:center;color:#1565C0;'>
🏦 Loan Approval Prediction System
</h1>
""", unsafe_allow_html=True)



col1, col2, col3 = st.columns(3)

col1.metric("🏦 Loans Processed", "10,000+")
col2.metric("✅ Approval Rate", "78%")
col3.metric("⭐ Rating", "4.9/5")




# =========================
# Inputs
# =========================
st.markdown('<div class="input-card">', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Age", 18, 100, 25)
    credit_score = st.number_input("📈 Credit Score", 300, 900, 700)
    loan_amount = st.number_input("💵 Loan Amount", 0, value=200000)

with col2:
    income = st.number_input("💰 Monthly Income", 0, value=50000)
    existing_emi = st.number_input("💳 Existing EMI", 0, value=5000)
    previous_defaults = st.number_input("⚠ Previous Defaults", 0, 10, 0)

employment = st.selectbox(
    "💼 Employment Type",
    ["Salaried", "Business", "Self-Employed"]
)

st.markdown('</div>', unsafe_allow_html=True)

# Encode Employment
employment_encoded = encoder.transform([employment])[0]

# =========================
# Prediction
# =========================

if st.button("🔍 Predict Loan Status"):

    input_data = np.array([[
        age,
        income,
        credit_score,
        existing_emi,
        employment_encoded,
        loan_amount,
        previous_defaults
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    approval_probability = probability[0][1] * 100

    st.markdown("---")
    st.subheader("📊 Prediction Dashboard")

    col1, col2 = st.columns(2)

    col1.metric(
        "💰 Monthly Income",
        f"₹{income:,}"
    )

    col2.metric(
        "📈 Approval Chance",
        f"{approval_probability:.2f}%"
    )

    st.progress(approval_probability / 100)

    if approval_probability >= 80:
        st.success("🟢 High Approval Chance")
    elif approval_probability >= 50:
        st.warning("🟡 Medium Approval Chance")
    else:
        st.error("🔴 Low Approval Chance")

    if prediction[0] == 1:
        st.balloons()
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(
        f"🎯 Approval Probability: {approval_probability:.2f}%"
    )