import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to load the data
@st.cache_data
def load_data():
    return pd.read_csv("S1Data.csv")

# Function to train the model
@st.cache_data
def train_model(data):
    X = data.drop(columns=["Event"])
    y = data["Event"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Scaling (optional but recommended)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, log_reg.predict(X_test))
    return log_reg, accuracy

# Main function
def main():
    st.title("Heart Failure Prediction App")
    st.markdown("""
        ### Welcome to the Heart Failure Prediction App
        This application predicts the likelihood of heart failure based on medical parameters.
        Please enter the relevant values in the sidebar for prediction.
    """)

    data = load_data()

    # Train the model
    model, accuracy = train_model(data)

    # Display accuracy
    st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

    st.sidebar.header("Enter Values for Prediction")
    time = st.sidebar.slider("Time", min_value=0, max_value=100, help="Time in days since diagnosis")
    age = st.sidebar.slider("Age", min_value=0, max_value=100, help="Age of the patient")
    ejection_fraction = st.sidebar.slider("Ejection Fraction", min_value=0, max_value=100, help="Percentage of blood leaving the heart each time it contracts")
    sodium = st.sidebar.slider("Sodium", min_value=0, max_value=200, help="Serum sodium level in mEq/L")
    creatinine = st.sidebar.slider("Creatinine", min_value=0, max_value=200, help="Serum creatinine level in mg/dL")
    platelets = st.sidebar.slider("Platelets", min_value=0, max_value=200, help="Platelet count in kiloplatelets/mL")
    cpk = st.sidebar.slider("CPK", min_value=0, max_value=200, help="Creatine phosphokinase level in U/L")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Gender of the patient")
    smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"], help="Smoking status")
    diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"], help="Diabetes status")
    bp = st.sidebar.selectbox("BP", ["High", "Normal", "Low"], help="Blood pressure status")
    anaemia = st.sidebar.selectbox("Anaemia", ["Yes", "No"], help="Anaemia status")

    # Map categorical values to numerical values
    gender = 1 if gender == "Male" else 0
    smoking = 1 if smoking == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    anaemia = 1 if anaemia == "Yes" else 0
    bp_mapping = {"High": 2, "Normal": 1, "Low": 0}
    bp = bp_mapping[bp]

    # Make prediction
    prediction = model.predict([[time, age, ejection_fraction, sodium, creatinine, platelets, cpk, gender, smoking, diabetes, bp, anaemia]])

    # Display prediction
    result = "Heart Failure" if prediction[0] else "No Heart Failure"
    st.write(f"## Prediction: {result}")

    if result == "Heart Failure":
        st.error("The model predicts a high risk of heart failure.")
    else:
        st.success("The model predicts a low risk of heart failure.")

if __name__ == "__main__":
    main()
