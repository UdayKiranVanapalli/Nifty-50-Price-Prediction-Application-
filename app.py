import numpy as np
import yfinance as yf
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta

# Load the model
model_path = r"C:\Users\udayk\Downloads\Career\python\BitCoin Price Prediction\nifty_prediction_model.keras"
model_path = "nifty_prediction_model.keras"

if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded successfully!")
else:
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Streamlit Header
st.title("Nifty 50 Price Prediction")
st.subheader("Historical Data Analysis and 6-Month Prediction")

# Download data using yfinance
try:
    data = yf.download("^NSEI", start="2020-01-27", end=datetime.today().strftime("%Y-%m-%d"))
    if data.empty:
        st.error("No data retrieved. Please check the ticker symbol or date range.")
        st.stop()
    data.reset_index(inplace=True)
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

# Display raw data
st.write("### Historical Data", data)

# Extract the Close column
close_data = data["Close"].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Sequence creation
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# Prepare last sequence for future prediction
seq_length = 100
if len(scaled_data) < seq_length:
    st.error("Insufficient data for sequence creation. Please use a larger dataset.")
    st.stop()

last_sequence = scaled_data[-seq_length:].reshape((1, seq_length, 1))

# Generate predictions
future_predictions = []
current_sequence = last_sequence.copy()

try:
    for _ in range(126):  # 21 trading days * 6 months = 126 days
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# Convert predictions back to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Create future dates
last_date = data["Date"].iloc[-1]
future_dates = []
current_date = last_date

for _ in range(126):
    current_date += timedelta(days=1)
    while current_date.weekday() > 4:  # Skip weekends
        current_date += timedelta(days=1)
    future_dates.append(current_date)

# Create DataFrame for future predictions
future_df = pd.DataFrame({"Date": future_dates, "Predicted_Price": future_predictions.flatten()})

# Display historical closing price trend
st.subheader("Historical Nifty 50 Closing Price Trend")
close_price_df = pd.DataFrame(close_data, index=data["Date"], columns=["Close"])
st.line_chart(close_price_df)

# Display future predictions
st.subheader("6-Month Future Price Predictions")
st.line_chart(future_df.set_index("Date"))

# Display prediction statistics
st.subheader("Prediction Statistics")
stats = {
    "Current Price": close_data[-1][0],
    "Predicted Price (After 1 Month)": future_predictions[20][0],
    "Predicted Price (After 3 Months)": future_predictions[62][0],
    "Predicted Price (After 6 Months)": future_predictions[-1][0],
    "Predicted Highest Price": np.max(future_predictions),
    "Predicted Lowest Price": np.min(future_predictions),
}

stats_df = pd.DataFrame([stats]).T.rename(columns={0: "Value"})
st.write(stats_df)

# Display monthly predictions
st.subheader("Monthly Predicted Prices")
monthly_indices = [0, 20, 41, 62, 83, 104, -1]  # Approximately monthly intervals
monthly_predictions = future_df.iloc[monthly_indices]
st.write(monthly_predictions)

# Calculate potential returns
returns = ((future_predictions[-1][0] - close_data[-1][0]) / close_data[-1][0]) * 100
st.subheader("Potential 6-Month Return")
st.write(f"{returns:.2f}%")

# Add risk warning
st.warning(
    """
    Please note: These predictions are based on historical data and technical analysis.
    Market conditions can change rapidly, and actual results may vary significantly.
    This tool should not be used as the sole basis for investment decisions.
    Always conduct thorough research and consider consulting with financial advisors.
    """
)
