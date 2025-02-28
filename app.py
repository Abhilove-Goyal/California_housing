import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os

# Load the model
model_path = "xgboost_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found! Ensure it's in the correct directory.")
    st.stop()

# Load the scaler
scaler_path = "my_scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("Scaler file not found! Ensure it's in the correct directory.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="California House Price Prediction", page_icon="🏡", layout="wide")

st.markdown("""
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
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>California House Price Prediction</h1>", unsafe_allow_html=True)

# Create tabs
tabs = st.tabs(["🏡 Prediction", "🗺️ Map", "ℹ️ About"])

with tabs[0]:
    st.image("california.jpg", use_container_width=True)
    st.sidebar.header("Enter House Details")

    # Sidebar Input Fields
    longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-114.0, step=0.01)
    latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.01)
    housing_median_age = st.sidebar.number_input("Housing Median Age (Years)", min_value=0.0, step=0.1)
    total_rooms = st.sidebar.number_input("Total Rooms", min_value=1, step=1)
    total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1, step=1)
    population = st.sidebar.number_input("Population", min_value=1, step=1)
    households = st.sidebar.number_input("Number of Households", min_value=1, step=1)
    median_income = st.sidebar.number_input("Median Income ($1000s)", min_value=0.0, step=0.1)
    
    # Compute derived features safely
    rooms_per_household = total_rooms / households
    bedroom_per_room = total_bedrooms / total_rooms
    popu_per_household = population / households

    # Predict Button
    if st.sidebar.button("Predict Price"):
        try:
            # Ensure feature names match training data
            feature_names = [
                "Longitude", "Latitude", "housingMedianAge", "totalRooms", "totalBedrooms", 
                "population", "households", "medianIncome", "rooms_per_household", 
                "popu_per_household", "bedroom_per_room"
            ]

            # Create a DataFrame with correct feature names
            input_data = pd.DataFrame([[
                longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
                population, households, median_income, rooms_per_household, 
                popu_per_household, bedroom_per_room
            ]], columns=feature_names)

            # Apply scaler
            scaled_data = scaler.transform(input_data)

            # Predict using the model
            prediction = model.predict(scaled_data)[0]
            st.success(f"Estimated House Price: ${prediction :.2f}")

        except ZeroDivisionError:
            st.error("Invalid input: Ensure no zero values in total rooms or households.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with tabs[1]:
    st.subheader("House Distribution in California")
    try:
        df = pd.read_csv("raw_df.csv")
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
    except FileNotFoundError:
        st.error("Dataset file 'raw_df.csv' not found! Ensure it's in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the map: {e}")

with tabs[2]:
    st.subheader("About the Model")
    st.write("This model predicts house prices in California using features like median income, house age, and location.")
    st.write("Dataset: California Housing Prices (from the 1990 U.S. Census data)")
    st.write("Algorithm: XGBoost Regressor")
