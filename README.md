# Gold Price Prediction App

This is a Streamlit web application that predicts gold prices based on historical data using a linear regression model. Users can input a specific date to retrieve the predicted gold price.

## Features

- **Date Picker**: Select a date for which you want to predict the gold price.
- **Prediction Display**: View the predicted gold price for the selected date.
- **Historical Price Plot**: Visualize historical gold prices with a line indicating the prediction date.
- **Historical Data Display**: Option to view historical gold price data.
- **Price Alert**: Set a price alert to receive a warning when the predicted price exceeds a specified threshold.

## Technologies Used

- **Python**: The main programming language used for this project.
- **Streamlit**: Framework for building the web application.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For building and evaluating the linear regression model.
- **Matplotlib**: For data visualization.
- **yfinance**: To fetch historical gold price data from Yahoo Finance.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/gold-price-prediction.git

2. Navigate to the project directory:
   cd gold-price-prediction

3. Install the required packages
   pip install -r requirements.txt


   Run the Streamlit APP:
   streamlit run strm_web.py
