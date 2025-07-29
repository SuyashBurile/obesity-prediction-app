import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open('obesity_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_encoders.pkl', 'rb') as f:
    feature_encoders = pickle.load(f)

with open('target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

# Streamlit UI
st.title("Obesity Prediction App")

st.markdown("""
Welcome! This app predicts **Obesity Category** based on personal and lifestyle inputs.
""")

# Input Fields
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

# Calculate BMI
bmi = round(weight / (height ** 2), 2)
st.write(f"**Calculated BMI:** {bmi}")

# Predict Button
if st.button("Predict"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Height": [height],
        "Weight": [weight],
        "BMI": [bmi],
        "PhysicalActivityLevel": [physical_activity]
    })

    # Apply label encoders to categorical features
    for col in feature_encoders:
        input_data[col] = feature_encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = target_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Obesity Category: **{predicted_class}**")
