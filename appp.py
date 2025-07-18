import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open("obesity_model.pkl", "rb"))

st.title("Obesity Category Prediction App")
st.write("Enter the following details to predict your obesity category:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (in meters)", min_value=0.5, max_value=2.5, value=1.7)
weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, value=70.0)
bmi = weight / (height ** 2)
activity_level = st.selectbox("Physical Activity Level (1 = low, 4 = high)", [1, 2, 3, 4])

# Gender encoding (assuming Male = 0, Female = 1 for model)
gender_encoded = 1 if gender == "Female" else 0

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, gender_encoded, height, weight, bmi, activity_level]])
    prediction = model.predict(input_data)[0]

    category_map = {
        0: "Normal weight",
        1: "Overweight",
        2: "Obese",
        3: "Extremely Obese"
    }

    st.success(f"Predicted Obesity Category: **{category_map.get(prediction, 'Unknown')}**")
