import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer taking the tourism package its details like City Occupation OwnCar etc.
Enter the customer details data below to get a prediction.
""")

# Numerical inputs
age = st.number_input("Age", min_value=18, max_value=100, value=40)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    max_value=60,
    value=10
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=20,
    value=3
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=20,
    value=2
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    max_value=1000000,
    value=20000
)

# Categorical inputs
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

passport = st.selectbox(
    "Passport",
    ["No", "Yes"]
)

own_car = st.selectbox(
    "Own Car",
    ["No", "Yes"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# Convert Yes/No to 0/1
passport = 1 if passport == "Yes" else 0
own_car = 1 if own_car == "Yes" else 0

# Create DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})


if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = "Package Taken" if prediction == 1 else "Package NOT Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
