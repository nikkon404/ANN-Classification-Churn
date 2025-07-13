import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

#  load the trained model
model = tf.keras.models.load_model("model.h5")

# load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Bank Customer Churn Prediction")

# user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, value=5000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input(
    "Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0
)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=2)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# prepare input data
input_data = {
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
}

# Convert input_data to a DataFrame
input_data_df = pd.DataFrame(input_data)

#  One hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine input data with encoded geography
input_data_combined = pd.concat(
    [input_data_df.reset_index(drop=True), geo_encoded_df], axis=1
)

#  Scale the input data
input_data_scaled = scaler.transform(input_data_combined)

#  Prediction churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

#  Display prediction
if prediction_proba > 0.5:
    st.success(
        f"The customer is likely to leave the bank with a probability of {prediction_proba:.2f}"
    )
else:
    st.success(
        f"The customer is likely to stay with the bank with a probability of {1 - prediction_proba:.2f}"
    )
