import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Stock Price Prediction using LSTM")

st.write("Enter recent stock price to predict next price")

price = st.number_input("Enter price:", min_value=0.0, value=150.0)

if st.button("Predict"):
    try:
        # Convert input to correct shape
        data = np.array([[price]])
        data_scaled = scaler.transform(data)

        # reshape properly for LSTM
        data_scaled = data_scaled.reshape((1, 1, 1))

        prediction = model.predict(data_scaled)

        # inverse scale
        prediction = scaler.inverse_transform(prediction)

        st.success(f"Predicted Price: {prediction[0][0]:.2f}")

    except Exception as e:
        st.error("Prediction failed. Model may require sequence input.")
        st.write("Error:", e)