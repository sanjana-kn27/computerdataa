import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "price_model.pkl"
FEATURES = ["speed", "ram", "hd", "screen"]

st.title("Computer Price Predictor")
st.write("Enter the computer specifications to estimate its price.")

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error("Model file not found. Make sure price_model.pkl is in the same folder as app.py.")
    st.stop()

speed = st.number_input("Speed", min_value=0.0, value=1.0)
ram = st.number_input("RAM", min_value=0.0, value=8.0)
hd = st.number_input("Hard drive (HD)", min_value=0.0, value=256.0)
screen = st.number_input("Screen size", min_value=0.0, value=14.0)

if st.button("Predict price"):
    computer = pd.DataFrame(
        [[speed, ram, hd, screen]],
        columns=FEATURES
    )
    predicted_price = model.predict(computer)[0]
    st.success(f"Predicted price: {predicted_price:,.2f}")
