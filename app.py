import numpy as np 
import pandas as pd 
import yfinance as yf 
from keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler 
import streamlit as st 

# Load Model  
model = load_model(r"C:\Users\udayk\Downloads\Career\python\BitCoin Price Prediction\nifty_prediction_model.keras")  

st.header('Nifty 50 Price Prediction Model')
st.subheader('Nifty 50 Price Data')

# Download data
data = pd.DataFrame(yf.download('^NSEI', '2020-01-01', '2024-11-21'))
data = data.reset_index()
st.write(data)

# Display line chart of closing prices
st.subheader('Nifty 50 Line Chart')
data_display = data[['Close']].copy()  # Create a copy for display
st.line_chart(data_display)

# Prepare data for prediction
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)

# Split data
train_data = data[:-100]
test_data = data[-200:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

# Create sequences
base_days = 100
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i,0])

# Convert to numpy arrays and reshape
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Make predictions
st.subheader('Predicted vs Original Prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1,1)
ys = scaler.inverse_transform(y.reshape(-1,1))

# Create DataFrames for visualization
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)
st.write(chart_data)

# Display prediction chart
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(chart_data)

# Future predictions
st.subheader('Future Nifty 50 Price Predictions')
m = y
z = []
future_days = 5

for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1,1)
    inter = [m[-base_days:,0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    m = np.append(m, pred)
    z = np.append(z, pred)

# Transform and display future predictions
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1,1))
future_predictions = pd.DataFrame(z, columns=['Future Predicted Price'])
st.line_chart(future_predictions)

# Display the last 5 predicted values
st.subheader('Next 5 Days Predicted Prices')
st.write(future_predictions.tail())