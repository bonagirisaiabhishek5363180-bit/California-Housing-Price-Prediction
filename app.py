import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

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
    st.session_state["show"] = True
    
if "show" in st.session_state:

    input_data = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
    }

    input_df = pd.DataFrame([input_data])

    input_prepared = pipeline.transform(input_df)
    prediction = model.predict(input_prepared)
    st.success(f"Predicted House Price: ${prediction[0]:.2f}")
    st.subheader("📊 Feature Importance")

    important_features = model.feature_importances_
    feature_names = pipeline.get_feature_names_out()

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": important_features
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax)

    st.pyplot(fig)