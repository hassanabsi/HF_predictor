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
    if not st.session_state.get("model_trained", False):
        accuracy = train_model(data)
        st.session_state["model_trained"] = True
        st.success(f"Model trained successfully with an accuracy of {accuracy * 100:.2f}%")
    
    model, scaler = load_model_and_scaler()
    
    st.sidebar.header("Enter Values for Prediction")

    # Input parameters
    time = st.sidebar.slider("Time (days since diagnosis)", min_value=0, max_value=365, help="Time in days since diagnosis")
    age = st.sidebar.slider("Age (years)", min_value=0, max_value=120, help="Age of the patient in years")
    ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", min_value=10, max_value=80, help="Heart ejection fraction percentage")
    sodium = st.sidebar.slider("Sodium (mEq/L)", min_value=120, max_value=160, help="Serum sodium level")
    creatinine = st.sidebar.slider("Creatinine (mg/dL)", min_value=0.3, max_value=10.0, help="Serum creatinine level")
    platelets = st.sidebar.slider("Platelets (kilo/mL)", min_value=10, max_value=1000, help="Platelet count in kilo/mL")
    cpk = st.sidebar.slider("CPK (U/L)", min_value=0, max_value=1500, help="Creatine phosphokinase level")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Gender of the patient")
    smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"], help="Smoking status")
    diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"], help="Diabetes status")
    bp = st.sidebar.selectbox("Blood Pressure", ["High", "Normal", "Low"], help="Blood pressure status")
    anaemia = st.sidebar.selectbox("Anaemia", ["Yes", "No"], help="Anaemia status")

    # Convert categorical inputs into numeric values
    gender = 1 if gender == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    anaemia = 1 if anaemia == "Yes" else 0
    bp_mapping = {"High": 2, "Normal": 1, "Low": 0}
    bp = bp_mapping[bp]

    # Prepare input data
    input_data = scaler.transform([[time, age, ejection_fraction, sodium, creatinine, platelets, cpk, gender, smoking, diabetes, bp, anaemia]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Heart Failure" if prediction == 1 else "No Heart Failure"
    
    # Display prediction result
    st.write(f"## Prediction: {result}")
    if prediction == 1:
        st.error("High risk of heart failure. Please consult your doctor.")
    else:
        st.success("Low risk of heart failure. Keep maintaining a healthy lifestyle.")

if __name__ == "__main__":
    main()
