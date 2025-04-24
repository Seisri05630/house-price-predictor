import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# ========== Load assets ==========
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_animation = load_lottie_url("https://lottie.host/27d022db-719f-45d2-9ec8-dff503cf432e/7BlGLVy3zd.json")

# ========== Load model and transformers ==========
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ========== Custom Styling ==========
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== Sidebar Navigation ==========
with st.sidebar:
    choice = option_menu("Navigation", ["Home", "Predict", "Upload"],
                         icons=["house", "graph-up", "cloud-upload"], menu_icon="cast", default_index=0)

# ========== Pages ==========
if choice == "Home":
    st.title("🏠 Smart House Price Predictor")
    st.write("Predict house prices using a smart ML model")

    if lottie_animation:
        st_lottie(lottie_animation, height=250)
    else:
        st.warning("⚠️ Animation couldn't be loaded. Check your internet or try another URL.")


elif choice == "Predict":
    st.title("🔍 Predict House Price")

    col1, col2 = st.columns(2)
    with col1:
        overall_qual = st.slider("🏗️ Overall Quality", 1, 10, 7)
        garage_cars = st.slider("🚗 Garage Capacity", 0, 4, 2)
    with col2:
        year_built = st.number_input("📅 Year Built", 1800, 2025, 2010)
        gr_liv_area = st.number_input("📏 Living Area (sq ft)", value=1800)

    input_dict = {
        "OverallQual": overall_qual,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "GrLivArea": gr_liv_area
    }

    df = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df)

    if st.button("Predict Price"):
        prediction = model.predict(df_scaled)
        st.success(f"Estimated House Price: ₹{int(prediction[0]):,}")

elif choice == "Upload":
    st.title("📄 Bulk Upload for Predictions")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        data = pd.read_csv(file)
        try:
            scaled_data = scaler.transform(data[features])
            predictions = model.predict(scaled_data)
            data['PredictedPrice'] = predictions
            st.write(data)
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Error in processing the file. Make sure it has the correct format.")

# ========== Footer ==========
st.markdown("""
    <hr/>
    <div style='text-align: center; color: grey;'>
        Made By SEIS_TEAM
        <br>Smart House Price Predictor - 2025
    </div>
""", unsafe_allow_html=True)
