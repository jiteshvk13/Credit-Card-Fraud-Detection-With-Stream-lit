import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('credit_card_fraud.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to check for fraud risk.")

# Input fields for 30 features
features = []
for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

# Predict button
if st.button("Predict"):
    row = np.array(features).reshape(1, -1)
    row_scaled = scaler.transform(row)
    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]

    st.write(f"###  Prediction: {'Fraud' if prediction == 1 else 'Legitimate'}")
    st.write(f"###  Fraud Probability: `{round(probability, 4)}`")

    if probability > 0.5:
        st.warning("⚠️ High risk of fraud!")
    else:
        st.success("Transaction looks safe.")
