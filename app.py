import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os

# Load the model
model_path = "xgboost_model"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found! Ensure it's in the correct directory.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="California House Price Prediction", page_icon="🏡", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #004d99;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-header'>California House Price Prediction</h1>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["🏡 Prediction", "🗺️ Map", "ℹ️ About"])

with tabs[0]:
    st.image("california.jpg", use_container_width=True)
    st.sidebar.header("Enter House Details")
    
    # User Inputs
    median_income = st.sidebar.slider("Median Income ($1000s)", 0.0, 15.0, 5.0)
    house_age = st.sidebar.slider("House Age (Years)", 0, 50, 25)
    avg_rooms = st.sidebar.slider("Average Rooms per Household", 1.0, 10.0, 5.0)
    avg_bedrooms = st.sidebar.slider("Average Bedrooms per Household", 0.5, 5.0, 1.5)
    population = st.sidebar.number_input("Population", min_value=1, step=1, value=1000)
    households = st.sidebar.number_input("Number of Households", min_value=1, step=1, value=500)
    latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 36.0)
    longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.5)
    
    if st.sidebar.button("Predict Price"):
        input_data = np.array([[median_income, house_age, avg_rooms, avg_bedrooms, population, households, latitude, longitude]])
        prediction = model.predict(input_data)[0]
        st.metric(label="🏠 Estimated House Price", value=f"${prediction * 100000:,.2f}", delta=None)

with tabs[1]:
    st.subheader("House Distribution in California")
    df = pd.read_csv("raw_df")
    df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)
    map_center = [df.latitude.mean(), df.longitude.mean()]
    california_map = folium.Map(location=map_center, zoom_start=6)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(california_map)
    st_folium(california_map, width=700, height=500)

with tabs[2]:
    st.subheader("About the Model")
    st.write("This model predicts house prices in California using features like median income, house age, and location.")
    st.write("Dataset: California Housing Prices (from the 1990 U.S. Census data)")
    st.write("Algorithm: XGBoost Regressor")
    