import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# ======================================================
# 🔹 Preprocess Data
# ======================================================
def preprocess_df(df):
    df = df.copy()
    df = df.dropna()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ======================================================
# 🔹 Train and Evaluate ML Models
# ======================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Linear Regression"):
    if model_name == "Linear Regression":
        model = Ridge(alpha=20)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_name == "XGBoost":
        if xgb is None:
            raise ValueError("XGBoost is not installed. Run: pip install xgboost")
        model =xgb.XGBRegressor(n_estimators=600,max_depth=5,learning_rate=0.03,subsample=0.9,colsample_bytree=0.9,random_state=42,objective="reg:squarederror",verbosity=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return model, preds, {"MAE": mae, "RMSE": rmse, "R2": r2}, scaler


# ======================================================
# 🔹 Plot Actual vs Predicted
# ======================================================
def plot_actual_vs_pred(y_test, preds, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(y=preds, mode='lines+markers', name=model_name))

    fig.update_layout(
        title=f"Actual vs Predicted GDP - {model_name}",
        xaxis_title="Index",
        yaxis_title="GDP (current US$)",
        template="plotly_white"
    )
    return fig


# ======================================================
# 🔹 LSTM SUPPORT
# ======================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def prepare_lstm_data(series, time_steps=3):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps])
    return np.array(X), np.array(y)


def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, activation="relu", return_sequences=False, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast_lstm(df, target_col="GDP (current US$)", future_years=10):

    values = df[target_col].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    X, y = prepare_lstm_data(scaled, 3)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_lstm((3, 1))

    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=150, batch_size=8, verbose=0, callbacks=[es])

    last_seq = scaled[-3:].reshape(1, 3, 1)
    preds_scaled = []

    for _ in range(future_years):
        nxt = model.predict(last_seq, verbose=0)[0][0]
        preds_scaled.append(nxt)
        last_seq = np.append(last_seq[:, 1:, :], [[[nxt]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    last_year = int(df["Year"].max())
    years = list(range(last_year + 1, last_year + future_years + 1))

    return pd.DataFrame({"Year": years, "Forecasted GDP (LSTM)": preds})


# ======================================================
# 🔹 ARIMA SUPPORT
# ======================================================
from statsmodels.tsa.arima.model import ARIMA


def train_arima(df, target_col="GDP (current US$)"):
    series = df[target_col].astype(float).values
    model = ARIMA(series, order=(2, 1, 2))
    return model.fit()


def forecast_arima(df, model_fit, future_years=10, target_col="GDP (current US$)"):

    last_year = int(df["Year"].max())
    forecast = model_fit.forecast(steps=future_years)

    years = list(range(last_year + 1, last_year + future_years + 1))

    return pd.DataFrame({
        "Year": years,
        "Forecasted GDP (ARIMA)": forecast
    })