# 📈 StockMindX

### AI-Powered Multi-Horizon Stock Price & Return Forecasting Platform

StockMindX is a production-ready Streamlit application that forecasts
short-term stock **returns** at multiple horizons using deep learning
(TensorFlow/Keras), derives **future prices** from those returns, and presents both
alongside technical indicators, model evaluation, a rule-based
trading signal, and a portfolio calculator.

Models are **trained** on a historical snapshot; every forecast and chart
in the running dashboard is computed from **live Yahoo Finance data**
fetched at request time.

---

## Features

- Multi-horizon forecasting (1, 5, and 20 trading days)
- Predicts both stock prices and log returns
- LSTM, GRU, and Transformer (TensorFlow/Keras)
- Live Yahoo Finance inference
- Technical indicators (RSI, MACD, Moving Averages, Volume)
- 95% Confidence Intervals
- Model comparison
- Portfolio calculator
- Buy / Hold / Sell recommendation
- Interactive Streamlit dashboard

---

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn
- yfinance

---

## Dataset

**Training**
- Yahoo Finance
- 20 Popular US Stocks
- Historical data (2015–Present)

**Inference**
- Live Yahoo Finance data

---

## Models

- LSTM
- GRU
- Transformer

Models are trained using walk-forward validation to prevent data leakage.

---

## Evaluation

- RMSE
- MAE
- MAPE
- Directional Accuracy

---

## Dashboard

- Live Forecasting
- Multi-Horizon Predictions
- Technical Indicators
- Model Comparison
- Portfolio Analytics
- Buy / Hold / Sell Recommendation
- CSV Export

---

## Exploratory Data Analysis

`notebooks/EDA.ipynb` includes:

- Price Trends
- Feature Engineering
- Correlation Analysis
- Technical Indicators
- Data Visualization

---

## Screenshots

Add dashboard screenshots here.

---

## Key Highlights

- Deep Learning-based financial forecasting
- Real-time Yahoo Finance inference
- Predicts stock prices and log returns
- Multi-model comparison
- Professional Streamlit dashboard
- Modular architecture

---

## Installation

```bash
git clone <repository-url>
cd StockMindX

pip install -r requirements.txt

python src/fetch_data.py
python src/features.py
python src/train.py

streamlit run app/app.py
```
