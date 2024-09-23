import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to fetch gold prices
def fetch_gold_prices():
    gold_data = yf.download('GC=F', start='2014-02-01', end='2024-07-01')
    return gold_data[['Close']].dropna()

# Function to train the model
def train_model(data):
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)
    X = data[['Date']]
    Y = data['Close']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# Streamlit interface
st.title("Gold Price Prediction")
st.sidebar.header("User Input")

# Date input for prediction
input_date = st.sidebar.date_input("Select a date for prediction", value=pd.to_datetime("2024-09-24"))

# Fetch and train model
gold_data = fetch_gold_prices()
model = train_model(gold_data)

# Predict for the selected date
ordinal_date = np.array(pd.Timestamp(input_date).toordinal()).reshape(-1, 1)
predicted_price = model.predict(ordinal_date)

# Display the prediction
st.subheader(f"Predicted Gold Price on {input_date}: ₹{predicted_price[0]:.2f}")

# Plot the actual vs predicted prices
st.subheader("Gold Price History")
plt.figure(figsize=(10, 6))
plt.plot(gold_data.index, gold_data['Close'], label='Actual Prices', color='blue')
plt.axvline(x=input_date, color='red', linestyle='--', label='Prediction Date')
plt.title("Historical Gold Prices")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.legend()
st.pyplot(plt)

# Additional features
if st.sidebar.checkbox("Show historical data"):
    st.subheader("Historical Gold Prices")
    st.write(gold_data)

# Option to set alerts
alert_price = st.sidebar.number_input("Set an alert for price:", min_value=0.0)
if predicted_price[0] >= alert_price:
    st.warning("Alert: Predicted price exceeds your set alert!")

