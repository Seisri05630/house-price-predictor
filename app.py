import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# ===== Load Lottie Animation with Fallback =====
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.warning(f"Could not load animation: {e}")
    return None

# Purple-themed animation
lottie_animation = load_lottie_url("https://lottie.host/4e8aecec-c3ef-4f4e-94d5-210a50519557/Vzls1t98Ht.json")

# ===== Load Model & Transformers =====
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
except Exception as e:
    st.error(f"🔴 Error loading model components: {e}")
    st.stop()

# ===== Custom Styling: White with Purple Accents =====
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #7e57c2;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5e35b1;
    }
    .stDownloadButton>button {
        background-color: #7e57c2;
        color: white;
        border-radius: 6px;
        padding: 8px 18px;
    }
    .css-1d391kg {
        color: #5e35b1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Sidebar Navigation =====
with st.sidebar:
    choice = option_menu("Navigation", ["Home", "Predict", "Upload"],
                         icons=["house", "graph-up", "cloud-upload"], menu_icon="cast", default_index=0)

# ===== Pages =====
if choice == "Home":
    st.title("🏠 Smart House Price Predictor - India Edition")
    st.write("Predict residential property prices across Indian cities using ML.")

    if lottie_animation:
        st_lottie(lottie_animation, height=250)
    else:
        st.info("🎬 Animation could not be loaded.")

elif choice == "Predict":
    st.title("🔍 Predict House Price (India)")

    with st.expander("🏗️ Structural Details"):
        overall_qual = st.slider("Overall Quality (1 - 10)", 1, 10, 7)
        garage_cars = st.slider("Garage Capacity (No. of Cars)", 0, 4, 2)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2010)
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=100, value=1800)

    with st.expander("🚰 Basic Utilities"):
        drinking_water = st.selectbox("Drinking Water Connection Available?", ["Yes", "No"])
        water_conn = 1 if drinking_water == "Yes" else 0

    with st.expander("🏙️ Location & Social Infrastructure"):
        social_infra = st.slider("Social Infrastructure Score (1 - 10)", 1, 10, 6)
        amenities = st.multiselect(
            "Nearby Amenities",
            ["School", "Hospital", "Park", "Mall", "Metro Station"]
        )
        amenity_score = len(amenities)

    # Prepare input for model
    input_dict = {
        "OverallQual": overall_qual,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "GrLivArea": gr_liv_area,
        "DrinkingWater": water_conn,
        "SocialInfra": social_infra,
        "AmenityScore": amenity_score
    }

    for f in features:
        if f not in input_dict:
            input_dict[f] = 0  # Fill missing features

    df = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df)

    if st.button("Predict Price 💰"):
        prediction = model.predict(df_scaled)
        st.success(f"🏷️ Estimated House Price: ₹{int(prediction[0]):,}")

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
            st.error("❌ Error processing the file. Make sure it includes all required columns.")

# ===== Footer =====
st.markdown("""
    <hr/>
    <div style='text-align: center; color: grey; font-size: 14px;'>
        Made with ❤️ by SEIS_TEAM <br>
        Smart House Price Predictor - India 🇮🇳 | 2025
    </div>
""", unsafe_allow_html=True)
