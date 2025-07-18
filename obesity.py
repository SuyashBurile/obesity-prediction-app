# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load model
pickle_in = open("obesity_model.pkl", "rb")
classifier = pickle.load(pickle_in)

# Prediction function
def predict_attack(Age, Gender, Height, Weight, BMI, PhysicalActivityLevel):
    Age = float(Age)
    Gender = int(Gender)
    Height = float(Height)
    Weight = float(Weight)
    BMI = float(BMI)
    PhysicalActivityLevel = int(PhysicalActivityLevel)

    prediction = classifier.predict([[Age, Gender, Height, Weight, BMI, PhysicalActivityLevel]])
    return int(prediction[0])

# App interface
def main():
    st.set_page_config(page_title="Obesity Category Predictor", page_icon="🧬", layout="centered")

    st.markdown(
        """
        <style>
        .main { background-color: #f8f9fa; }
        .title-box {
            background-color: #2c3e50;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .title-box h2 {
            color: white;
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
        }
        .stTextInput > div > label {
            font-weight: 600;
            color: #333;
        }
        </style>
        <div class="title-box">
            <h2>🧬 Obesity Category Prediction System</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Enter the following details:")

    Age = st.text_input("Age (e.g., 25)")
    Gender = st.text_input("Gender (0 = Female, 1 = Male)")
    Height = st.text_input("Height in cm (e.g., 170)")
    Weight = st.text_input("Weight in kg (e.g., 68)")
    BMI = st.text_input("BMI (e.g., 22.5)")
    PhysicalActivityLevel = st.text_input("Physical Activity Level (1–4)")

    result = ""
    if st.button("Predict Category"):
        try:
            result = predict_attack(Age, Gender, Height, Weight, BMI, PhysicalActivityLevel)
            if result == 0:
                st.success("✅ Prediction: Normal Weight (Category 0)")
            elif result == 1:
                st.success("⚠️ Prediction: Obese (Category 1)")
            elif result == 2:
                st.success("📈 Prediction: Overweight (Category 2)")
            elif result == 3:
                st.success("📉 Prediction: Underweight (Category 3)")
            else:
                st.error("❌ Unexpected category predicted!")
        except:
            st.error("🚫 Invalid input. Please enter numeric values correctly.")

if __name__ == '__main__':
    main()
