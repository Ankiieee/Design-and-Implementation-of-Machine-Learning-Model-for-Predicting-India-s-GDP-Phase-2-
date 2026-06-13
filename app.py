import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from time_series import run_arima, run_lstm
from utils import (
    preprocess_df,
    train_and_evaluate,
    plot_actual_vs_pred,
    forecast_lstm,
    train_arima,
    forecast_arima 
)
import os
import random
import numpy as np
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
  
def lstm_test_prediction(train, test, window=5):
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # Combine for proper scaling
    full = np.concatenate([
        train["GDP (current US$)"].values,
        test["GDP (current US$)"].values
    ]).reshape(-1, 1)

    scaler = MinMaxScaler()
    full_scaled = scaler.fit_transform(full)

    train_scaled = full_scaled[:len(train)]
    test_scaled = full_scaled[len(train):]

    # Prepare training data
    X_train, y_train = [], []
    for i in range(window, len(train_scaled)):
        X_train.append(train_scaled[i-window:i])
        y_train.append(train_scaled[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Build model (simple & stable)
    model = Sequential([
        LSTM(16, activation='tanh', recurrent_dropout=0),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(
    X_train, 
    y_train, 
    epochs=60, 
    verbose=0, 
    shuffle=False
)
    # Prepare test inputs (IMPORTANT FIX)
    X_test = []
    for i in range(len(train_scaled), len(full_scaled)):
        X_test.append(full_scaled[i-window:i])

    X_test = np.array(X_test)

    # Predict test directly (NO recursive drift)
    preds_scaled = model.predict(X_test, verbose=0)

    preds = scaler.inverse_transform(preds_scaled).flatten()

    return preds
  
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_models(df, steps=10):

    df = df.sort_values("Year")

    train = df.iloc[:-steps]
    test = df.iloc[-steps:]

    actual = test["GDP (current US$)"].values

    # ARIMA
    arima_pred = run_arima(train, steps)["Forecast"].values

    # LSTM
    lstm_pred = lstm_test_prediction(train, test)

    # Metrics
    arima_mae = mean_absolute_error(actual, arima_pred)
    arima_rmse = np.sqrt(mean_squared_error(actual, arima_pred))
    arima_r2 = r2_score(actual, arima_pred)

    lstm_mae = mean_absolute_error(actual, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(actual, lstm_pred))
    lstm_r2 = r2_score(actual, lstm_pred)

    return arima_mae, arima_rmse, arima_r2, lstm_mae, lstm_rmse, lstm_r2  


# =====================================================
# Streamlit Page Settings
# =====================================================
st.set_page_config(
    page_title="GDP Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv("IndData.csv")
df = preprocess_df(df) 


target_col = "GDP (current US$)"
X = df.drop(columns=[target_col])
y = df[target_col]

# =====================================================
# Sidebar Navigation
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Analysis", "Model Comparison", "Forecasting", "Recommendation"]
)

# =====================================================
# 1) PREDICTION PAGE
# =====================================================
if page == "Prediction":
    st.header("📈 GDP Prediction Dashboard (ML Models)")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = ["Linear Regression", "Random Forest", "XGBoost"]
    results = {}

    st.subheader("Model Results on Test Data")

    for model_name in models:
        try:
            model, preds, metrics, scaler = train_and_evaluate(
                X_train, X_test, y_train, y_test, model_name
            )
            results[model_name] = {"model": model, "preds": preds,
                                   "scaler": scaler, "metrics": metrics}

            st.success(
                f"**{model_name}** → R² = {metrics['R2']:.3f}, "
                f"RMSE = {metrics['RMSE']:.0f}, MAE = {metrics['MAE']:.0f}"
            )

        except Exception as e:
            st.warning(f"{model_name} failed: {e}")

    st.subheader("Manual Input for Prediction")

    input_vals = {}
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        mn, mx = float(X[col].min()), float(X[col].max())
        val = float(X[col].median())

        if col.lower() == "year":
            input_vals[col] = cols[i % 3].number_input(
                col, value=2025, min_value=int(df["Year"].min()), max_value=2050
            )
        else:
            input_vals[col] = cols[i % 3].number_input(
                col, value=val, min_value=mn, max_value=mx
            )

    if st.button("Predict GDP"):
        sample = pd.DataFrame([input_vals])[X.columns]

        st.success("Predictions:")
        for name, r in results.items():
            try:
                pred = r["model"].predict(r["scaler"].transform(sample))
                st.write(f"- **{name}** → ${float(pred[0]):,.2f}")
            except:
                st.write(f"- **{name}** → error in prediction")

# =====================================================
# 2) ANALYSIS PAGE
# =====================================================
elif page == "Analysis":

    st.header("📊 GDP Analysis (Indian Economy)")

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # -------------------------------
    # ROW 1
    # -------------------------------
    col1, col2 = st.columns(2)

    # Correlation Heatmap
    with col1:
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("GDP has strong positive correlation with population and exports.")
        
    # GDP Trend
    with col2:
        st.subheader("GDP Trend Over Years")

        fig_gdp = px.line(
            df,
            x='Year',
            y='GDP (current US$)',
            markers=True
        )
        st.plotly_chart(fig_gdp, use_container_width=True)
        st.info("GDP shows a consistent upward trend over the years.")
    st.divider()

    # -------------------------------
    # ROW 2
    # -------------------------------
    col3, col4 = st.columns(2)

    # GDP Growth
    with col3:
        st.subheader("Annual GDP Growth (%)")

        fig_growth = px.line(
            df,
            x='Year',
            y='GDP_growth',
            markers=True
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        st.info("GDP growth fluctuates, indicating periods of economic expansion and slowdown.")

    # Population vs Life Expectancy
    with col4:
        st.subheader("Population vs Life Expectancy")

        fig_pop = go.Figure()

        fig_pop.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Population'],
            mode='lines+markers',
            name='Population'
        ))

        fig_pop.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Life_expectancy'],
            mode='lines+markers',
            name='Life Expectancy'
        ))

        st.plotly_chart(fig_pop, use_container_width=True)
        st.info("Population and life expectancy both show steady improvement over time.")
    st.divider()

    # -------------------------------
    # ROW 3
    # -------------------------------
    col5, col6 = st.columns(2)

    # Sector Contributions
    with col5:
        st.subheader("Sector Contributions to GDP (%)")

        sectors = [
            'Agriculture',
            'Industry',
            'Exports',
            'Imports',
            'Military',
            'Merchandise_trade'
        ]

        fig_sector = px.bar(
            df,
            x='Year',
            y=sectors
        )

        st.plotly_chart(fig_sector, use_container_width=True)
        st.info("Industry and services contribute more significantly to GDP than other sectors.")

    # Exports vs Imports
    with col6:
        st.subheader("Exports vs Imports")

        fig_trade = go.Figure()

        fig_trade.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Exports'],
            name='Exports',
            line=dict(color='green')
        ))

        fig_trade.add_trace(go.Scatter(
            x=df['Year'],
            y=df['Imports'],
            name='Imports',
            line=dict(color='red')
        ))

        st.plotly_chart(fig_trade, use_container_width=True)
        st.info("Exports and imports move closely, reflecting active trade dynamics.")
    st.divider()

    # -------------------------------
    # ROW 4
    # -------------------------------
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("### GDP vs Inflation")

        fig_inf = px.scatter(
            df,
            x="Inflation",
            y="GDP (current US$)",
            size="Population",
            color="GDP_growth"
        )
        st.plotly_chart(fig_inf, use_container_width=True)
        st.info("Inflation shows a moderate impact on GDP performance.")
    with col8:
        st.markdown("### Feature Impact on GDP")

        corr_target = df.corr(numeric_only=True)['GDP (current US$)'].sort_values()

        fig_corr_bar = px.bar(
            corr_target,
            orientation='h'
        )
        st.plotly_chart(fig_corr_bar, use_container_width=True)
        st.info("A few key factors have a strong influence on GDP compared to others.")
       

        
    st.divider()

    # -------------------------------
    # ROW 5
    # -------------------------------
    col9, col10 = st.columns(2)

    # FDI vs GDP
    with col9:
        st.markdown("### Foreign Investment vs GDP")

        fig_fdi = px.line(
            df,
            x='Year',
            y=['FDI', 'GDP (current US$)']
        )
        st.plotly_chart(fig_fdi, use_container_width=True)
        st.info("Higher FDI is associated with increased GDP levels.")
    # Trade Balance
    with col10:
        st.markdown("### Trade Balance")

        df['Trade_Balance'] = df['Exports'] - df['Imports']

        fig_tb = px.line(
            df,
            x='Year',
            y='Trade_Balance'
        )
        st.plotly_chart(fig_tb, use_container_width=True)
        st.info("Trade balance fluctuates, indicating changing export-import gaps.")
    st.divider()

  

    # -------------------------------
    # ROW 7 (3D SCATTER)
    # -------------------------------
    st.markdown("### 🌐 3D GDP Relationship")

    if all(c in df.columns for c in ["GDP (current US$)", "Population", "Inflation"]):

      fig3d = px.scatter_3d(
        df,
        x="Population",
        y="Inflation",
        z="GDP (current US$)",
        color="Year",
        size=df["GDP_growth"].abs() + 1   # ✅ FIXED (no negative issue)
    )

    st.plotly_chart(fig3d, use_container_width=True)

    st.info("GDP increases with population, while inflation shows mixed impact.")

# -------------------------------
# 3)Model Comparison Page
# -------------------------------
elif page == "Model Comparison":
    st.title("📊 Model Comparison: Actual vs Predicted GDP")

    X = df.drop(columns=["GDP (current US$)", "Year", "Unnamed: 0"], errors="ignore")
    y = df["GDP (current US$)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models_to_compare = ["Linear Regression", "Random Forest", "XGBoost"]
    all_metrics = {}

    for model_name in models_to_compare:
        try:
            model, preds, metrics, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, model_name)
            all_metrics[model_name] = metrics

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values, mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=preds, mode="lines+markers", name=model_name))
            fig.update_layout(title=f"Actual vs Predicted GDP ({model_name})", xaxis_title="Index", yaxis_title="GDP", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            
        except Exception as e:
            st.error(f"{model_name} failed: {e}")

    # Summary Chart
    if all_metrics:
        st.subheader("📊 Overall Model Performance (R² Score)")
        summary_df = pd.DataFrame({
            "Model": list(all_metrics.keys()),
            "R2 Score": [m["R2"] for m in all_metrics.values()]
        })
        fig_summary = px.bar(summary_df, x="Model", y="R2 Score", color="Model", text="R2 Score", title="Model Comparison by R² Score", template="plotly_white")
        fig_summary.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_summary, use_container_width=True)
        

# =====================================================
# 4) FORECASTING PAGE (LSTM + ARIMA)
# =====================================================
elif page == "Forecasting":
    st.header("🔮 GDP Forecasting (Next 10 Years)")

    from time_series import run_arima, run_lstm

    steps = st.slider("Years to Forecast", 5, 20, 10)

    arima_df = run_arima(df, steps)
    lstm_df = run_lstm(df, steps)

    st.subheader("Forecast Results")

    import plotly.graph_objects as go

    # =========================
    # ARIMA GRAPH
    # =========================
    st.subheader("🔵 ARIMA Forecast")

    fig_arima = go.Figure()

    # Actual
    fig_arima.add_trace(go.Scatter(
    x=df["Year"], y=df["GDP (current US$)"],
    name="Actual"
    ))

    # Forecast
    fig_arima.add_trace(go.Scatter(
    x=arima_df["Year"], y=arima_df["Forecast"],
    name="ARIMA Forecast"
    ))

    # Upper CI line
    fig_arima.add_trace(go.Scatter(
    x=arima_df["Year"],
    y=arima_df["Upper CI"],
    name="Upper CI",
    line=dict(dash='dash')
    ))

    # Lower CI line
    fig_arima.add_trace(go.Scatter(
    x=arima_df["Year"],
    y=arima_df["Lower CI"],
    name="Lower CI",
    line=dict(dash='dash')
    ))

    # Layout
    fig_arima.update_layout(
    title="ARIMA Forecast vs Actual",
    xaxis_title="Year",
    yaxis_title="GDP"
    )
    st.plotly_chart(fig_arima, use_container_width=True)


   # =========================
   # LSTM GRAPH
   # =========================
    st.subheader("🟢 LSTM Forecast")

    fig_lstm = go.Figure()

    fig_lstm.add_trace(go.Scatter(
    x=df["Year"], y=df["GDP (current US$)"],
    name="Actual"
    ))

    fig_lstm.add_trace(go.Scatter(
    x=lstm_df["Year"], y=lstm_df["Forecast"],
    name="LSTM Forecast"
    ))

    fig_lstm.update_layout(
    title="LSTM Forecast vs Actual",
    xaxis_title="Year",
    yaxis_title="GDP"
    )

    st.plotly_chart(fig_lstm, use_container_width=True)
    
    # =========================
    # METRICS
    # =========================
    st.subheader("📊 Evaluation Metrics")

    arima_mae, arima_rmse, arima_r2, lstm_mae, lstm_rmse, lstm_r2 = evaluate_models(df, steps)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 ARIMA")
        st.metric("R² Score", f"{arima_r2:.3f}")
        st.metric("MAE", f"{arima_mae:.2f}")
        st.metric("RMSE", f"{arima_rmse:.2f}")

    with col2:
        st.markdown("### 🟢 LSTM")
        st.metric("R² Score", f"{lstm_r2:.3f}")
        st.metric("MAE", f"{lstm_mae:.2f}")
        st.metric("RMSE", f"{lstm_rmse:.2f}")    
    #Store ARIMA and LSTM metrics
    forecast_metrics = {
    "ARIMA": {"R2": arima_r2},
    "LSTM": {"R2": lstm_r2}
    }
    # Summary Chart
    if forecast_metrics:
      st.subheader("📊 Forecast Model Performance (R² Score)")

    summary_df = pd.DataFrame({
        "Model": list(forecast_metrics.keys()),
        "R2 Score": [m["R2"] for m in forecast_metrics.values()]
    })

    fig_summary = px.bar(
        summary_df,
        x="Model",
        y="R2 Score",
        color="Model",
        text="R2 Score",
        title="ARIMA vs LSTM Comparison (R² Score)",
        template="plotly_white"
    )

    fig_summary.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(fig_summary, use_container_width=True)

elif page == 'Recommendation':

    # ---------------------------------
    # HEADER
    # ---------------------------------
    st.title("📊 India GDP Recommendation Dashboard")
    st.markdown("### Smart policy suggestions to achieve target GDP")

    # ---------------------------------
    # LOAD DATA
    # ---------------------------------
    df = pd.read_csv('IndData.csv')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # ---------------------------------
    # CORRELATION
    # ---------------------------------
    corr = df.corr(numeric_only=True)['GDP (current US$)']
    corr_sorted = corr.sort_values(ascending=False)

    positive_features = corr_sorted[corr_sorted > 0].index.tolist()
    negative_features = corr_sorted[corr_sorted < 0].index.tolist()

    if 'Year' in positive_features:
        positive_features.remove('Year')

    # ---------------------------------
    # LATEST DATA
    # ---------------------------------
    latest = df[df['Year'] == 2020]
    gdp = float(latest['GDP (current US$)'])
    gdp_trillion = gdp / 1e12

    # ---------------------------------
    # TARGET SLIDER
    # ---------------------------------
    target = st.slider(
        "🎯 Target GDP (Trillion $)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

    # ---------------------------------
    # KPI SECTION
    # ---------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📌 Current GDP", f"{gdp_trillion:.2f} T")
    col2.metric("🎯 Target GDP", f"{target:.2f} T")

    growth = ((target - gdp_trillion) / gdp_trillion) * 100 if target > gdp_trillion else 0
    col3.metric("📈 Required Growth", f"{growth:.2f}%")

    st.divider()

    # ---------------------------------
    # PROGRESS BAR
    # ---------------------------------
    if target > gdp_trillion:
        progress = min(gdp_trillion / target, 1.0)
        st.progress(progress)
        st.caption(f"Progress toward target: {progress*100:.1f}%")

    # ---------------------------------
    # CURRENT DATA
    # ---------------------------------
    with st.expander("📊 Current Economic Indicators (2020)", expanded=False):
        st.dataframe(latest, use_container_width=True)

   # ---------------------------------
   # 🤖 AI RECOMMENDATION ENGINE + SIMULATOR + OPTIMIZER
   # ---------------------------------
    if target > gdp_trillion:
      percent = ((target - gdp_trillion) / gdp_trillion) * 100

    st.markdown("# 🤖 Economic Recommendation Engine")
    st.markdown("### Policy Optimization & Growth Strategy")

    # 🎯 TARGET GAP
    st.warning(f"📉 GDP needs to grow by **{percent:.2f}%** to reach the target.")

    col1, col2 = st.columns(2)

    # 🟢 IMPROVE
    with col1:
        st.markdown("### 🟢 Growth Drivers")

        for feature in positive_features[1:6]:
            val = float(latest[feature])
            new_val = val * (1 + percent / 100)

            st.markdown(f"""
            🔹 **{feature}**  
            Current: `{val:.2f}` → Target: **{new_val:.2f}**
            """)

    # 🔴 REDUCE
    with col2:
        st.markdown("### 🔴 Risk Factors")

        for feature in negative_features[:5]:
            val = float(latest[feature])
            new_val = val * (1 - percent / 100)

            st.markdown(f"""
            🔸 **{feature}**  
            Current: `{val:.2f}` → Target: **{abs(new_val):.2f}**
            """)

    # ---------------------------------
    # 📊 IMPACT CHART
    # ---------------------------------
    import plotly.express as px
    import pandas as pd

    impact_data = []

    for f in positive_features[1:6]:
        impact_data.append({"Feature": f, "Impact": corr_sorted[f], "Type": "Positive"})

    for f in negative_features[:5]:
        impact_data.append({"Feature": f, "Impact": corr_sorted[f], "Type": "Negative"})

    impact_df = pd.DataFrame(impact_data)

    fig = px.bar(
        impact_df,
        x="Impact",
        y="Feature",
        color="Type",
        orientation="h",
        title="📊 Economic Drivers Impact Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # 🧠 AI INSIGHTS
    # ---------------------------------
    st.markdown("### 🧠 AI Insights")

    st.success("""
    ✔ Strengthening growth drivers increases GDP  
    ✔ Reducing risk factors stabilizes economy  
    ✔ Balanced strategy ensures sustainable growth  
    """)

    # ---------------------------------
    # 🎮 GDP SIMULATOR
    # ---------------------------------
    st.markdown("---")
    st.markdown("## 🎮 GDP Growth Simulator")

    sim_data = {}

    # sliders
    for feature in positive_features[1:4]:
        sim_data[feature] = st.slider(f"Increase {feature}", 0.0, 50.0, 10.0)

    for feature in negative_features[:3]:
        sim_data[feature] = st.slider(f"Reduce {feature}", 0.0, 50.0, 10.0)

    # ---------------------------------
    # ⚡ SCENARIOS
    # ---------------------------------
    st.markdown("### ⚡ Quick Scenarios")

    colA, colB, colC = st.columns(3)

    scenario_data = {}

    with colA:
        if st.button("📉 Recession"):
            for f in positive_features[1:4]:
                scenario_data[f] = 5
            for f in negative_features[:3]:
                scenario_data[f] = 30

    with colB:
        if st.button("📈 Growth"):
            for f in positive_features[1:4]:
                scenario_data[f] = 20
            for f in negative_features[:3]:
                scenario_data[f] = 10

    with colC:
        if st.button("🚀 Aggressive"):
            for f in positive_features[1:4]:
                scenario_data[f] = 40
            for f in negative_features[:3]:
                scenario_data[f] = 5

    if scenario_data:
        sim_data.update(scenario_data)

    # ---------------------------------
    # 🤖 AUTO OPTIMIZATION
    # ---------------------------------
    st.markdown("### 🤖 Auto Optimize")

    if st.button("⚡ Optimize to Reach Target"):

        optimized = {}
        required_growth = (target - gdp_trillion) / gdp_trillion

        total_weight = sum([abs(corr_sorted[f]) for f in positive_features[1:4]])

        for f in positive_features[1:4]:
            weight = abs(corr_sorted[f])
            optimized[f] = (required_growth * (weight / total_weight)) * 100

        for f in negative_features[:3]:
            optimized[f] = 5

        sim_data.update(optimized)

        st.success("✅ Optimal strategy applied!")

    # ---------------------------------
    # 📊 SIMULATION CALCULATION
    # ---------------------------------
    impact_score = 0

    for feature, change in sim_data.items():
        weight = corr_sorted.get(feature, 0)

        if feature in positive_features:
            impact_score += weight * change
        else:
            impact_score -= abs(weight) * change

    estimated_growth = impact_score / 100
    new_gdp = gdp_trillion * (1 + estimated_growth)

    # ---------------------------------
    # 📊 RESULTS
    # ---------------------------------
    st.markdown("### 📊 Simulation Results")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Estimated GDP", f"{new_gdp:.2f} Trillion USD")

    with col4:
        st.metric("Growth Change", f"{estimated_growth*100:.2f}%")

    progress = min(new_gdp / target, 1.0)
    st.progress(progress)
    st.write(f"🎯 Target Progress: {progress*100:.2f}%")

    # ---------------------------------
    # 🤖 AI FEEDBACK
    # ---------------------------------
    st.markdown("### 🤖 AI Feedback")

    if new_gdp >= target:
        st.success("🎉 Target achieved! Strategy is highly effective.")
    elif new_gdp > gdp_trillion:
        st.info("📈 Positive growth but target not reached.")
    else:
        st.warning("⚠️ Strategy ineffective. Adjust factors.")

    # ---------------------------------
    # 📈 LIVE CHART
    # ---------------------------------
    sim_df = pd.DataFrame({
        "Factor": list(sim_data.keys()),
        "Change (%)": list(sim_data.values())
    })

    fig2 = px.bar(sim_df, x="Factor", y="Change (%)", title="📈 Simulation Adjustments")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------
    # 🎯 FINAL DECISION (FIXED)
    # ---------------------------------
    st.markdown("---")
    st.markdown("## 🎯 Final Recommendation")
    if 'new_gdp' not in locals():
      new_gdp = gdp_trillion

    if new_gdp >= target:
        st.success("🎉 Target Achieved!")

        st.markdown("""
        ✅ Your current strategy is strong and effective.  
        📈 Maintain momentum in key growth drivers.  
        🛡️ Keep risk factors under control to sustain growth.  
        """)

    elif new_gdp > gdp_trillion:
        gap = target - new_gdp

        st.info("📈 Positive Growth, but Target Not Yet Reached")

        st.markdown(f"""
        🔍 **Gap Remaining:** {gap:.2f} Trillion USD  

        💡 **Suggestions:**
        - Increase investment in top-performing sectors  
        - Improve exports and industrial output  
        - Gradually reduce high-impact risk factors  
        """)

    else:
        st.error("⚠️ Strategy Ineffective")

        st.markdown("""
        🚨 Current adjustments are not driving growth.
        """)
