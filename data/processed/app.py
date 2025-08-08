import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and label encoder
model = joblib.load('model.pkl','r+')
encoder = joblib.load('label_encoder.pkl','r+')

# Title
st.title("🌍 Climate Predictor")
st.subheader("Predict Average Temperature for a Country in a Given Year")

# --- Input Fields ---

# Get list of country names from the label encoder
country_names = encoder.classes_
selected_country = st.selectbox("Select Country", country_names)

# Convert country to encoded value
country_encoded = encoder.transform([selected_country])[0]

# Other inputs
year = st.number_input("Enter Year", min_value=1900, max_value=2100, value=2025)
co2 = st.number_input("Annual CO₂ Emissions (in tonnes)", min_value=0.0, step=0.1)
forest_area = st.number_input("Forest Area (in sq.km or %)", min_value=0.0, step=0.1)
urban_growth = st.number_input("Urban Growth (in %)", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict Temperature"):
    input_data = np.array([[country_encoded, year, co2, forest_area, urban_growth]])
    prediction = model.predict(input_data)
    st.success(f"🌡️ Predicted Average Temperature: {prediction[0]:.2f} °C")
