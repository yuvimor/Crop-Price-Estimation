import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Load dataset
df = pd.read_csv('crop_data.csv')

# Data Cleaning
df = df.dropna()
df['Price Date'] = pd.to_datetime(df['Price Date'], format='%b-%y')

# Feature Engineering
df['Month'] = df['Price Date'].dt.month
df['Year'] = df['Price Date'].dt.year

# Average price per month
avg_price = df.groupby(['District', 'Crop', 'Year', 'Month'])['Crop Price (Rs per quintal)'].mean().reset_index()

# Function to forecast prices
def forecast_prices(district, crop):
    subset = avg_price[(avg_price['District'] == district) & (avg_price['Crop'] == crop)]

    if len(subset) >= 2:
        X_train = subset[['Month', 'Year']]
        y_train = subset['Crop Price (Rs per quintal)']

        model = LinearRegression()
        model.fit(X_train, y_train)

        latest_month = subset[['Month', 'Year']].iloc[-1].copy()

        forecasted_prices = []
        for i in range(1, 4):
            # Increment the month and adjust the year accordingly
            latest_month['Month'] += 1
            if latest_month['Month'] > 12:
                latest_month['Month'] = 1
                latest_month['Year'] += 1

            next_month_data = pd.DataFrame([latest_month], columns=['Month', 'Year'])
            forecast = model.predict(next_month_data)
            forecasted_prices.append({'Year': latest_month['Year'], 'Month': latest_month['Month'], 'Forecasted Price': forecast[0]})
        
        return forecasted_prices
    else:
        return []

# Function to recommend crops based on the highest average prices in the current month
def recommend_crops(district):
    current_month = datetime.now().month
    top_crops = avg_price[(avg_price['District'] == district) & (avg_price['Month'] == current_month)].groupby('Crop')['Crop Price (Rs per quintal)'].mean().nlargest(3).reset_index()
    return top_crops[['Crop', 'Crop Price (Rs per quintal)']].round(2)

# Streamlit App
st.title('Crop Price Estimation App')

# User Input: Dropdowns for District and Crop
district_options = avg_price['District'].unique()
crop_options = avg_price['Crop'].unique()

selected_district = st.selectbox('Select District:', district_options)
selected_crop = st.selectbox('Select Crop:', crop_options)

# Forecast and Display Results
if selected_crop and selected_district:
    # Forecasted Prices
    forecasted_prices = forecast_prices(selected_district, selected_crop)
    
    if forecasted_prices:
        st.write(f'Forecasted Prices for {selected_crop} in {selected_district} for the next 3 months:')
        st.write(pd.DataFrame(forecasted_prices)['Forecasted Price'])
        
    else:
        st.write(f'Not enough data for {selected_crop} in {selected_district}. Unable to make predictions.')

    st.markdown('---')  # Separator

    # Recommended Crops
    st.write(f'Recommended Crops in {selected_district} for the Current Month ({datetime.now().strftime("%B %Y")}):')
    recommended_crops = recommend_crops(selected_district)
    st.write(recommended_crops)
