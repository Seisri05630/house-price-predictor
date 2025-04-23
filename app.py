import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and components
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('features.pkl')  # List of features used in training

# App title
st.title("🏡 Real-Time House Price Predictor (Simple Version)")
st.write("Enter the details below to predict the house price:")

# Choose a few important features
important_inputs = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
input_data = {}

# Ask user to input these few values
input_data['OverallQual'] = st.slider("Overall Quality (1 = Very Poor, 10 = Excellent)", 1, 10, 5)
input_data['GrLivArea'] = st.number_input("Above Ground Living Area (sq ft)", value=1500)
input_data['GarageCars'] = st.slider("Garage Capacity (Cars)", 0, 4, 2)
input_data['TotalBsmtSF'] = st.number_input("Total Basement Area (sq ft)", value=800)
input_data['YearBuilt'] = st.number_input("Year Built", value=2000)

# Create DataFrame with all selected features, fill unused with 0
full_input = {f: 0 for f in selected_features}  # Default all to zero
full_input.update(input_data)  # Replace with user input for important ones

# Make into DataFrame
df_input = pd.DataFrame([full_input])

# Scale the input
scaled_input = scaler.transform(df_input)

# Predict and display
if st.button("🔍 Predict House Price"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
