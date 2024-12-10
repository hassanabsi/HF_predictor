import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Function to load the data
@st.cache_data
def load_data():
    return pd.read_csv("S1Data.csv")

# Function to train the model
def train_model(data):
    X = data.drop(columns=["Event"])
    y = data["Event"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, log_reg.predict(X_test))
    
    # Save the model and scaler
    joblib.dump(log_reg, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return accuracy

# Load saved model and scaler
def load_model_and_scaler():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# Main function
def main():
    st.title("Heart Failure Prediction App")
    data = load_data()

    # Train the model and display accuracy
    if not st.session_state.get("model_trained"):
        accuracy = train_model(data)
        st.session_state["model_trained"] = True
    model, scaler = load_model_and_scaler()
    
    st.sidebar.header("Enter Values for Prediction")

# Time: Typically ranges from 0 days to several years (converted to days)
time = st.sidebar.slider(
    "Time (days since diagnosis)", min_value=0, max_value=3650, 
    help="Time in days since diagnosis (e.g., up to 10 years)"
)

# Age: Based on human lifespan, typically 0-120 years
age = st.sidebar.slider(
    "Age (years)", min_value=0, max_value=120, 
    help="Age of the patient in years"
)

# Ejection Fraction: Normal range is 50-70%; less than 40% indicates heart failure
ejection_fraction = st.sidebar.slider(
    "Ejection Fraction (%)", min_value=10, max_value=80, 
    help="Percentage of blood leaving the heart each time it contracts (normal: 50-70%)"
)

# Sodium: Normal serum sodium levels range from 135 to 145 mEq/L
sodium = st.sidebar.slider(
    "Sodium (mEq/L)", min_value=120, max_value=160, 
    help="Serum sodium level (normal: 135-145 mEq/L)"
)

# Creatinine: Normal levels are 0.5 to 1.2 mg/dL; higher levels indicate kidney issues
creatinine = st.sidebar.slider(
    "Creatinine (mg/dL)", min_value=0.3, max_value=10.0, 
    help="Serum creatinine level (normal: 0.5-1.2 mg/dL; higher values may indicate kidney dysfunction)"
)

# Platelets: Normal platelet count is 150,000 to 450,000 per microliter
platelets = st.sidebar.slider(
    "Platelets (kilo/mL)", min_value=50, max_value=800, 
    help="Platelet count in kiloplatelets/mL (normal: 150-450)"
)

# CPK: Normal creatine phosphokinase levels are 20-200 U/L
cpk = st.sidebar.slider(
    "CPK (U/L)", min_value=10, max_value=2000, 
    help="Creatine phosphokinase level (normal: 20-200 U/L; elevated levels may indicate muscle damage)"
)

# Gender: 0 = Female, 1 = Male
gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"], 
    help="Gender of the patient"
)

# Smoking: Binary input, 1 = Yes, 0 = No
smoking = st.sidebar.selectbox(
    "Smoking", ["Yes", "No"], 
    help="Smoking status"
)

# Diabetes: Binary input, 1 = Yes, 0 = No
diabetes = st.sidebar.selectbox(
    "Diabetes", ["Yes", "No"], 
    help="Diabetes status"
)

# BP: Categorized into High, Normal, or Low
bp = st.sidebar.selectbox(
    "Blood Pressure", ["High", "Normal", "Low"], 
    help="Blood pressure status"
)

# Anaemia: Binary input, 1 = Yes, 0 = No
anaemia = st.sidebar.selectbox(
    "Anaemia", ["Yes", "No"], 
    help="Anaemia status"
)


    # Scale the input
    input_data = scaler.transform([[time, age, ejection_fraction, sodium, creatinine, platelets, cpk, gender, smoking, diabetes, bp, anaemia]])
    
    # Predict
    prediction = model.predict(input_data)
    result = "Heart Failure" if prediction[0] else "No Heart Failure"
    st.write(f"## Prediction: {result}")
    st.error("High risk of heart failure." if prediction[0] else "Low risk of heart failure.")

if __name__ == "__main__":
    main()
