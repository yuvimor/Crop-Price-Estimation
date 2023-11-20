import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Loading the saved model and historical data
model = joblib.load('model.pkl')
historical_data = pd.read_csv('crop_data.csv') 

# Recommendation function
def crop_recommendation(inputs):
    # Identify the crop with the highest price in historical data
    highest_price_crop = historical_data.groupby('CROP')['CROP_PRICE'].idxmax()
    best_crop_states = historical_data.loc[highest_price_crop]['STATE'].unique()

    return best_crop_states

# Streamlit app
st.title("Crop Price Estimation and Planting Recommendation App")

# Sidebar for user input
st.sidebar.header("Input Features")

inputs = {
    'N_SOIL': st.sidebar.slider('Nitrogen in Soil', min_value=0, max_value=100, value=50),
    'P_SOIL': st.sidebar.slider('Phosphorous in Soil', min_value=0, max_value=100, value=50),
    'K_SOIL': st.sidebar.slider('Potassium in Soil', min_value=0, max_value=100, value=50),
    'TEMPERATURE': st.sidebar.slider('Temperature', min_value=0.0, max_value=40.0, value=25.0),
    'HUMIDITY': st.sidebar.slider('Humidity', min_value=0.0, max_value=100.0, value=50.0),
    'PH': st.sidebar.slider('Soil pH', min_value=0.0, max_value=14.0, value=7.0),
    'RAINFALL': st.sidebar.slider('Rainfall', min_value=0.0, max_value=500.0, value=250.0),
}

# Convert the input features to a DataFrame
input_df = pd.DataFrame([inputs])

# Make predictions
prediction = model.predict(input_df)

# Display prediction
st.write(f"Predicted Crop Price: ${prediction[0]:,.2f}")

# Display planting recommendations
recommended_states = crop_recommendation(inputs)
st.write(f"Recommended States for Planting: {', '.join(recommended_states)}")
