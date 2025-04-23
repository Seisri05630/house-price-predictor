import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# 🎨 Background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1599422314077-f4dfdaa4cd9d");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()

# 🔧 Load model, scaler and features
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('features.pkl')  # List of all features

# 🎯 Sidebar
st.sidebar.title("📘 About This App")
st.sidebar.info(
    "Predict housing prices using a smart regression model. "
    "This app uses machine learning to give real-time and batch predictions based on selected features."
)
st.sidebar.markdown("Built with ❤️ using Streamlit")

# 🏠 App Title
st.title("🏡 Real-Time House Price Predictor")

# Top 5 important inputs
important_inputs = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
input_data = {}

st.subheader("🔢 Enter Details for a Single House")

# 🎲 Sample values button
if st.button("🎲 Use Sample Values"):
    input_data = {
        'OverallQual': 7,
        'GrLivArea': 1800,
        'GarageCars': 2,
        'TotalBsmtSF': 1000,
        'YearBuilt': 2005
    }
else:
    # Collect user input
    input_data['OverallQual'] = st.slider("Overall Quality (1 = Poor, 10 = Excellent)", 1, 10, 5)
    input_data['GrLivArea'] = st.number_input("Above Ground Living Area (sq ft)", value=1500)
    input_data['GarageCars'] = st.slider("Garage Capacity (Cars)", 0, 4, 2)
    input_data['TotalBsmtSF'] = st.number_input("Total Basement Area (sq ft)", value=800)
    input_data['YearBuilt'] = st.number_input("Year Built", value=2000)

# 🧠 Predict single house price
if st.button("🔍 Predict House Price"):
    full_input = {f: 0 for f in selected_features}
    full_input.update(input_data)
    df_input = pd.DataFrame([full_input])
    scaled_input = scaler.transform(df_input)
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")

# 📂 Batch prediction from CSV
st.subheader("📄 Or Upload a CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        # Fill missing columns with zero
        for col in selected_features:
            if col not in data.columns:
                data[col] = 0

        # Keep only selected features
        data = data[selected_features]
        scaled_batch = scaler.transform(data)
        batch_preds = model.predict(scaled_batch)
        data['PredictedPrice'] = batch_preds
        st.write("✅ Prediction Complete!")
        st.dataframe(data)
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv, file_name="predicted_prices.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error: {e}")
