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
    time = st.sidebar.slider("Time", min_value=0, max_value=100)
    age = st.sidebar.slider("Age", min_value=0, max_value=100)
    ejection_fraction = st.sidebar.slider("Ejection Fraction", min_value=0, max_value=100)
    sodium = st.sidebar.slider("Sodium", min_value=0, max_value=200)
    creatinine = st.sidebar.slider("Creatinine", min_value=0, max_value=200)
    platelets = st.sidebar.slider("Platelets", min_value=0, max_value=200)
    cpk = st.sidebar.slider("CPK", min_value=0, max_value=200)
    gender = 1 if st.sidebar.selectbox("Gender", ["Male", "Female"]) == "Male" else 0
    smoking = 1 if st.sidebar.selectbox("Smoking", ["Yes", "No"]) == "Yes" else 0
    diabetes = 1 if st.sidebar.selectbox("Diabetes", ["Yes", "No"]) == "Yes" else 0
    anaemia = 1 if st.sidebar.selectbox("Anaemia", ["Yes", "No"]) == "Yes" else 0
    bp_mapping = {"High": 2, "Normal": 1, "Low": 0}
    bp = bp_mapping[st.sidebar.selectbox("BP", ["High", "Normal", "Low"])]

    # Scale the input
    input_data = scaler.transform([[time, age, ejection_fraction, sodium, creatinine, platelets, cpk, gender, smoking, diabetes, bp, anaemia]])
    
    # Predict
    prediction = model.predict(input_data)
    result = "Heart Failure" if prediction[0] else "No Heart Failure"
    st.write(f"## Prediction: {result}")
    st.error("High risk of heart failure." if prediction[0] else "Low risk of heart failure.")

if __name__ == "__main__":
    main()
