import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("California Housing Price Prediction")

st.write("Enter housing details to predict house price")

longitude = st.number_input("Longitude", value=-118.0)
latitude = st.number_input("Latitude", value=34.0)
housing_median_age = st.number_input("Housing Median Age", value=20)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=400)
median_income = st.number_input("Median Income", value=4.0)


ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)


if st.button("Predict Price"):
    input_data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,

        
        "ocean_proximity_<1H OCEAN": 0,
        "ocean_proximity_INLAND": 0,
        "ocean_proximity_ISLAND": 0,
        "ocean_proximity_NEAR BAY": 0,
        "ocean_proximity_NEAR OCEAN": 0
    }

    input_data[f"ocean_proximity_{ocean_proximity}"] = 1

    input_df = pd.DataFrame([input_data])


    prediction = model.predict(input_df)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")