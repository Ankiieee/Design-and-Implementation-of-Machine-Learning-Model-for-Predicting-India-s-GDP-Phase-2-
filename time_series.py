from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def run_arima(df, steps=10):
    df = df.sort_values("Year")
    series = df["GDP (current US$)"]

    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()

    # ✅ Get forecast with confidence interval
    forecast_obj = model_fit.get_forecast(steps=steps)

    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    future_years = list(range(df["Year"].max()+1, df["Year"].max()+1+steps))

    return pd.DataFrame({
        "Year": future_years,
        "Forecast": forecast.values,
        "Lower CI": conf_int.iloc[:, 0].values,
        "Upper CI": conf_int.iloc[:, 1].values
    })
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def run_lstm(df, steps=10):
    df = df.sort_values("Year")
    data = df["GDP (current US$)"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    window = 5

    for i in range(len(data_scaled)-window):
        X.append(data_scaled[i:i+window])
        y.append(data_scaled[i+window])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(window,1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    # Forecast future
    last_seq = data_scaled[-window:]
    preds = []

    for _ in range(steps):
        pred = model.predict(last_seq.reshape(1,window,1), verbose=0)
        preds.append(pred[0][0])
        last_seq = np.append(last_seq[1:], pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    future_years = list(range(df["Year"].max()+1, df["Year"].max()+1+steps))

    return pd.DataFrame({
        "Year": future_years,
        "Forecast": preds.flatten()
    })