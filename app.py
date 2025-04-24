import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page config ---
st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Background style ---
def add_bg():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1568605114967-8130f3a36994");
             background-attachment: fixed;
             background-size: cover;
             font-family: 'Segoe UI', sans-serif;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg()

# --- Load model and data ---
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('features.pkl')

# --- Sidebar Info ---
st.sidebar.title("📘 About")
st.sidebar.info(
    "This app predicts house prices based on top features using a trained regression model. "
    "You can try it with sample values or upload a CSV for bulk predictions."
)
st.sidebar.markdown("Built with ❤️ by [Your Name]")

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #003366;'>🏡 Smart House Price Predictor</h1>", unsafe_allow_html=True)

# --- Sample Input & Prediction ---
st.subheader("🎯 Enter House Features")

input_data = {}

col1, col2 = st.columns(2)
with col1:
    input_data['OverallQual'] = st.slider("Overall Quality (1–10)", 1, 10, 7)
    input_data['GarageCars'] = st.slider("Garage Capacity (cars)", 0, 4, 2)
    input_data['YearBuilt'] = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
with col2:
    input_data['GrLivArea'] = st.number_input("Living Area (sq ft)", value=1800)
    input_data['TotalBsmtSF'] = st.number_input("Basement Area (sq ft)", value=1000)

if st.button("🎲 Use Sample Values"):
    input_data = {
        'OverallQual': 8,
        'GrLivArea': 2000,
        'GarageCars': 2,
        'TotalBsmtSF': 900,
        'YearBuilt': 2010
    }
    st.experimental_rerun()

# Predict button
if st.button("🔍 Predict House Price"):
    full_input = {f: 0 for f in selected_features}
    full_input.update(input_data)
    df_input = pd.DataFrame([full_input])
    scaled_input = scaler.transform(df_input)
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")

# --- Batch Prediction Section ---
st.markdown("---")
st.subheader("📄 Upload a CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        for col in selected_features:
            if col not in data.columns:
                data[col] = 0

        data = data[selected_features]
        scaled = scaler.transform(data)
        preds = model.predict(scaled)
        data['PredictedPrice'] = preds

        st.success("✅ Predictions complete. Scroll to preview or download below.")
        st.dataframe(data.style.format({"PredictedPrice": "${:,.2f}"}))

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predicted_prices.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# --- Footer ---
st.markdown("""
<hr style="border: 0.5px solid #ddd;">
<div style='text-align: center; font-size: 0.9em;'>
Created by <a href="https://github.com/yourusername" target="_blank">Your Name</a> • Powered by Streamlit & XGBoost
</div>
""", unsafe_allow_html=True)
