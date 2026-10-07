import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing objects
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaller.pkl")
expected_columns = joblib.load("columns.pkl")

# Title
st.title("Heart Disease Prediction by Suraj")

st.markdown("### Enter Patient Details")

# Input fields
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.slider(
    "Resting Blood Pressure (mm Hg)",
    80, 200, 120
)

cholesterol = st.slider(
    "Cholesterol (mg/dl)",
    100, 600, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate Achieved",
    60, 220, 150
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=7.0,
    value=0.0,
    step=0.1
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# Prediction
if st.button("Predict"):

    # Convert UI values to dataset values
    sex_value = "M" if sex == "Male" else "F"
    exercise_value = "Y" if exercise_angina == "Yes" else "N"

    # Create input dictionary
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex_value: 1,
        "ChestPainType_" + chest_pain_type: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_value: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns exactly as training data
    input_df = input_df[expected_columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")