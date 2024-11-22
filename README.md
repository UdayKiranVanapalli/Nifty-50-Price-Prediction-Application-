# Nifty 50 Price Prediction Application 🚀
![image](https://github.com/user-attachments/assets/e1a0ca46-70f1-42a5-adab-d66f23287805)

## 📊 Predict Future Nifty 50 Prices Using Deep Learning

This project analyzes the performance of the **Nifty 50 index in India** following the onset of COVID-19. Leveraging historical data and deep learning techniques, it provides insights into market trends, recovery phases, and future price predictions, highlighting the resilience of the Indian stock market.

---

## 📌 Features
1. **Historical Data Visualization**:
   - Displays historical Nifty 50 prices from 2020 to the present.
   - Allows users to analyze price trends over time.

2. **6-Month Future Predictions**:
   - Predicts Nifty 50 prices for approximately 126 trading days (~6 months).
   - Shows predicted price trends via interactive charts.

3. **Statistics and Insights**:
   - Displays key statistics like predicted monthly prices, highest and lowest prices, and potential returns.

4. **Monthly Predictions**:
   - Provides specific predicted prices at approximately monthly intervals.

5. **Risk Warnings**:
   - Alerts users about the limitations of predictions and the need for financial research.

6. **Interactive Dashboard**:
   - Built with **Streamlit** for real-time updates and data visualization.

---

## 🛠️ Technology Stack
- **Programming Language**: Python 3.8+
- **Frameworks & Libraries**:
  - TensorFlow/Keras: For model inference.
  - yFinance: For downloading stock data.
  - Streamlit: For creating an interactive dashboard.
  - Pandas & NumPy: For data manipulation.
  - Scikit-learn: For data scaling (Min-Max Normalization).

---

## 🧠 Model Overview
- The model used is a trained **Long Short-Term Memory (LSTM)** neural network designed for time series forecasting.
- It has been trained on Nifty 50 historical data for robust and accurate predictions.

---

## 📈 Use Case
This project is beneficial for:
- **Investors**: To gain insights into potential future trends.
- **Traders**: To analyze potential returns over a 6-month period.
- **Researchers & Analysts**: To explore deep learning applications in financial markets.

---

## 🚀 Installation & Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/UdayKiranVanapalli/nifty50-price-prediction.git


## 🖼️ Application Walkthrough

### **1. Dashboard Overview**
- **Historical Data Trends:**
  
  - Analyze Nifty 50's historical performance starting from January 2020, including its recovery and growth post-COVID-19.
  - Interactive visualizations reveal the market's volatility during the pandemic and subsequent rebounds.
- **Post-COVID Analysis:**
  
  - Insights on how Nifty 50 responded after COVID-19 hit India, focusing on:
  - Initial decline: The sharp drop due to pandemic-induced uncertainty.
  - Recovery phase: Market trends as the economy adapted to the "new normal."
  - Growth trajectory: Long-term trends and emerging opportunities post-pandemic.

### **2. Prediction Outputs**
- **6-Month Price Chart:**
  - Predict Nifty 50's future performance for 126 trading days (~6 months), considering its resilience after the COVID-19 downturn.
- Monthly Predictions:
  - Monthly interval predictions showcase price stabilization and recovery trends.
- Statistics:
  - Insights into the highest, lowest, and potential returns, emphasizing how the market regains momentum in a post-COVID world.
### **3. Insights on Post-COVID Trends**
- **Recovery Trends:**
  - Visualization of Nifty 50's "V-shaped" recovery after its March 2020 crash.
  - Highlights sectors driving growth, such as technology, pharma, and consumer goods.
- **Potential Returns:**
  - Predicted returns and volatility give a better understanding of future market opportunities.
### **🔑 Key Components**
### **Input**

  - Historical Data: Downloaded via yFinance (^NSEI), capturing pre- and post-COVID price movements.
### **Processing**

  - Data Scaling: Prices normalized with MinMaxScaler for efficient model training.
  - LSTM Sequences: Deep learning model trained on sequences of 100-day windows.
### **Output**

- **Predictions:**
  - 6-month forecast showcasing Nifty 50's trajectory post-pandemic recovery.
  - Monthly intervals for practical tracking.  
## ⚠️ Risk Warning
### **Disclaimer:**
This tool is for educational purposes and should not be considered financial advice. Predictions are based on historical data and trained models, which cannot guarantee future performance. Always consult a financial advisor before making investment decisions.   
## 🖥️ Screenshots
![image](https://github.com/user-attachments/assets/a186a086-cb60-41fc-a161-171e513db799)
![image](https://github.com/user-attachments/assets/8b947e37-a0e7-436b-a33d-f9b13f77c253)

## 👥 Acknowledgements

- **TensorFlow:** Deep learning libraries for building the LSTM model.
- **Streamlit:** Framework for interactive app development.
- **Yahoo Finance (yFinance):** Source for historical stock data.
