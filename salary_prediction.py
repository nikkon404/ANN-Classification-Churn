import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Load the trained regression model
model = tf.keras.models.load_model("salary_regression_model.h5")

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Bank Customer Salary Prediction")

# User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, max_value=200000.0, value=5000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=2)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
exited = st.selectbox("Exited", [0, 1])
# Prepare input data dictionary (no salary here)
input_data = {
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited],
}

# Convert to DataFrame
input_data_df = pd.DataFrame(input_data)

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine all features
input_data_combined = pd.concat(
    [input_data_df.reset_index(drop=True), geo_encoded_df], axis=1
)

# Scale input
input_data_scaled = scaler.transform(input_data_combined)

# Predict salary
predicted_salary = model.predict(input_data_scaled)[0][0]

# Display prediction
st.success(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")
