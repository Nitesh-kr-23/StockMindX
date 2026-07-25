# --------------------------------------------------------------------------- #
# Streamlit dashboard
# --------------------------------------------------------------------------- #
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import FEATURE_COLUMNS, HORIZONS 
from fetch_data import TICKERS  
from live_data import get_latest_window 
from models import MODEL_BUILDERS 
from recommendation import recommend, project_portfolio_value 

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
Z_SCORE_95 = 1.96  # normal-approx z for a 95% confidence interval
MODEL_NAMES = ["LSTM", "GRU", "Transformer"]

st.set_page_config(page_title="StockMindX", layout="wide", page_icon="📈")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    """Loads all trained Keras models, the fitted scaler, evaluation
    metrics, and per-model/per-horizon residual std (for confidence
    intervals) once per session. These were produced by training on the
    static historical snapshot -- unrelated to the live data used below."""
    bundle = joblib.load(MODELS_DIR / "scaler.joblib")
    models = {name: tf.keras.models.load_model(MODELS_DIR / f"{name}.keras") for name in MODEL_NAMES}

    with open(MODELS_DIR / "metrics.json") as f:
        metrics_bundle = json.load(f)
    with open(MODELS_DIR / "residual_std.json") as f:
        residual_std = json.load(f)

    return models, bundle["scaler"], bundle["seq_len"], metrics_bundle["metrics"], metrics_bundle["best_model"], residual_std


@st.cache_data(ttl=900, show_spinner="Fetching live market data...")
def load_live_ticker_data(ticker: str, seq_len: int):
    """Live yfinance fetch for the selected ticker, cached for 15 minutes
    so rapid widget interactions don't hammer Yahoo Finance. This is the
    ONLY source of per-ticker data used anywhere in the app."""
    return get_latest_window(ticker, seq_len=seq_len)


# --------------------------------------------------------------------------- #
# Core forecasting logic
# --------------------------------------------------------------------------- #
def predict_returns(model, window_df: pd.DataFrame, scaler) -> np.ndarray:
    """Runs one forward pass and returns predicted log returns for each
    horizon, shape (n_horizons,)."""
    X = scaler.transform(window_df[FEATURE_COLUMNS].values.astype(np.float32))
    x = np.expand_dims(X, axis=0).astype(np.float32)
    pred = model.predict(x, verbose=0)
    return pred[0]


def derive_future_price(current_price: float, predicted_log_return: float) -> float:
    """Future Price = Current Price x exp(Predicted Log Return).
    Deterministic transform -- no separate price model is trained."""
    return float(current_price * np.exp(predicted_log_return))


def confidence_interval(predicted_return: float, residual_std: float, z: float = Z_SCORE_95):
    """Simple normal-approximation confidence interval around the point
    forecast, using the model's own validation-set residual std-dev as the
    uncertainty estimate (no Monte Carlo sampling required)."""
    return predicted_return - z * residual_std, predicted_return + z * residual_std


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_sidebar(models):
    with st.sidebar:
        st.header("⚙️ Controls")
        ticker = st.selectbox("Stock ticker", TICKERS)
        model_name = st.selectbox("Model", list(models.keys()),
                                   index=list(models.keys()).index("Transformer"))
        st.markdown("---")
        st.caption(
            "**Live inference:** every forecast and chart below is "
            "computed from data fetched live from Yahoo Finance for the "
            "selected ticker (cached for 15 minutes). Models were trained "
            "on a historical snapshot (2015-present); see README for details."
        )
        investment = st.number_input("Portfolio calculator: investment amount ($)",
                                      min_value=0.0, value=10000.0, step=500.0)
        return ticker, model_name, investment


def render_price_and_forecast_cards(current_price, predicted_returns, ci_bounds):
    """Top-of-dashboard metric cards: current price, predicted return, and
    predicted price for each horizon, shown side by side."""
    st.subheader("Forecast summary")
    cols = st.columns(len(HORIZONS) + 1)
    with cols[0]:
        st.metric("Current price (live)", f"${current_price:,.2f}")

    for i, h in enumerate(HORIZONS):
        future_price = derive_future_price(current_price, predicted_returns[i])
        lo, hi = ci_bounds[i]
        with cols[i + 1]:
            st.metric(
                f"{h}-day forecast",
                f"${future_price:,.2f}",
                delta=f"{predicted_returns[i]:+.2%}",
            )
            st.caption(f"95% CI: {lo:+.2%} to {hi:+.2%}")


def render_forecast_chart(ticker_hist, current_price, predicted_returns, ci_bounds, chart_key):
    """Live historical close price with a forecast extension and shaded
    confidence band.

    `chart_key` is derived from the ticker/model so Streamlit tears down
    and rebuilds the chart element cleanly on change, rather than
    transitioning an existing element in place -- this avoids the old
    chart briefly overlapping the new one during a rerun.
    """
    last_date = ticker_hist["Date"].iloc[-1]
    hist_plot = ticker_hist.tail(120)

    proj_dates = [last_date + pd.tseries.offsets.BDay(h) for h in HORIZONS]
    mean_prices = [derive_future_price(current_price, r) for r in predicted_returns]
    lower_prices = [derive_future_price(current_price, ci[0]) for ci in ci_bounds]
    upper_prices = [derive_future_price(current_price, ci[1]) for ci in ci_bounds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_plot["Date"], y=hist_plot["Close"],
                              name="Historical close (live)", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=[last_date] + proj_dates, y=[current_price] + mean_prices,
                              name="Forecast (mean)", line=dict(color="#F58518", dash="dash")))
    fig.add_trace(go.Scatter(x=[last_date] + proj_dates, y=[current_price] + upper_prices,
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[last_date] + proj_dates, y=[current_price] + lower_prices,
                              fill="tonexty", fillcolor="rgba(245,133,24,0.18)",
                              line=dict(width=0), name="95% confidence band"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                       legend=dict(orientation="h", y=1.08),
                       title="Historical price (live) with forecast extension",
                       transition_duration=0)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_indicator_tabs(ticker_hist, key_prefix):
    """Technical indicator charts computed from live data: RSI, MACD,
    Volume, Moving Averages -- each in its own tab."""
    recent = ticker_hist.tail(180)
    tab_ma, tab_rsi, tab_macd, tab_vol = st.tabs(
        ["Moving Averages", "RSI", "MACD", "Volume"]
    )

    with tab_ma:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"], name="Close", line=dict(color="#4C78A8")))
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["sma_20"], name="SMA 20", line=dict(color="#F58518")))
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["sma_50"], name="SMA 50", line=dict(color="#54A24B")))
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["ema_12"], name="EMA 12", line=dict(color="#B279A2", dash="dot")))
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), transition_duration=0)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ma")

    with tab_rsi:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["rsi_14"], name="RSI 14", line=dict(color="#E45756")))
        fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 100], transition_duration=0)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_rsi")

    with tab_macd:
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["macd"], name="MACD", line=dict(color="#4C78A8")))
        fig.add_trace(go.Bar(x=recent["Date"], y=recent["macd_hist"], name="Histogram", marker_color="#B279A2"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), transition_duration=0)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_macd")

    with tab_vol:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=recent["Date"], y=recent["Volume"], name="Volume", marker_color="#72B7B2"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), transition_duration=0)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_vol")


def render_model_comparison(all_metrics, best_model):
    """Comparison table across all models + naive baseline, computed once
    during training on the historical held-out test set (not live data --
    evaluation needs a fixed, reproducible benchmark)."""
    st.subheader("Model comparison (held-out test set, walk-forward split)")
    st.caption(
        f"🏆 **Best-performing model: {best_model}** (lowest mean RMSE across horizons on the test set). "
        "The Naive row predicts zero return and is the bar every model must clear. "
        "Computed once during training on historical data, not live data."
    )
    rows = []
    for name, hmetrics in all_metrics.items():
        for h, vals in hmetrics.items():
            rows.append({
                "Model": name, "Horizon": h,
                "RMSE": vals["rmse"], "MAE": vals["mae"],
                "MAPE (%)": vals["mape"], "Directional Accuracy (%)": vals["directional_accuracy"],
            })
    comp_df = pd.DataFrame(rows)

    for metric in ["RMSE", "MAE", "MAPE (%)"]:
        pivot = comp_df.pivot(index="Model", columns="Horizon", values=metric).round(4)
        st.markdown(f"**{metric}** (lower is better)")
        st.dataframe(pivot.style.highlight_min(axis=0, color="#2E7D32"), use_container_width=True)

    pivot_da = comp_df.pivot(index="Model", columns="Horizon", values="Directional Accuracy (%)").round(2)
    st.markdown("**Directional Accuracy (%)** (higher is better; 50% = coin flip)")
    st.dataframe(pivot_da.style.highlight_max(axis=0, color="#2E7D32"), use_container_width=True)


def render_recommendation(ticker_hist, predicted_returns, ci_bounds):
    """Rule-based Buy/Hold/Sell card using the 5-day forecast horizon,
    its confidence interval, and live RSI/MACD."""
    st.subheader("Rule-based recommendation")
    idx_5d = HORIZONS.index(5)
    latest = ticker_hist.iloc[-1]

    rec = recommend(
        predicted_return=predicted_returns[idx_5d],
        ci_lower=ci_bounds[idx_5d][0],
        ci_upper=ci_bounds[idx_5d][1],
        rsi=latest["rsi_14"],
        macd_hist=latest["macd_hist"],
    )

    color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[rec.action]
    st.markdown(f"### {color} {rec.action}  (signal score: {rec.score:+d})")
    for reason in rec.reasons:
        st.write(f"- {reason}")
    st.caption(
        "This is a simple, transparent rule-based signal for educational/portfolio "
        "purposes based on the 5-day forecast -- not financial advice."
    )


def render_portfolio_calculator(current_price, predicted_returns, investment):
    """Lets the user see the projected value of a hypothetical investment
    at each forecast horizon, derived from the predicted future price."""
    st.subheader("Portfolio calculator")
    st.caption(f"Projected value of a ${investment:,.2f} investment at today's live price, "
               "using each horizon's predicted future price.")
    cols = st.columns(len(HORIZONS))
    for i, h in enumerate(HORIZONS):
        future_price = derive_future_price(current_price, predicted_returns[i])
        result = project_portfolio_value(investment, current_price, future_price)
        with cols[i]:
            st.metric(f"{h}-day projected value",
                      f"${result['projected_value']:,.2f}",
                      delta=f"{result['gain_pct']:+.2%}")


def render_download(ticker, model_name, current_price, predicted_returns, ci_bounds):
    """Builds a small predictions table and offers it as a CSV download."""
    rows = []
    for i, h in enumerate(HORIZONS):
        future_price = derive_future_price(current_price, predicted_returns[i])
        lo, hi = ci_bounds[i]
        rows.append({
            "ticker": ticker, "model": model_name, "horizon_days": h,
            "current_price": current_price,
            "predicted_log_return": predicted_returns[i],
            "predicted_price": future_price,
            "ci_lower_return": lo, "ci_upper_return": hi,
            "ci_lower_price": derive_future_price(current_price, lo),
            "ci_upper_price": derive_future_price(current_price, hi),
        })
    out_df = pd.DataFrame(rows)
    st.download_button(
        "⬇️ Download predictions as CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_{model_name}_predictions.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("📈 StockMindX")
    st.caption("AI-Powered Multi-Horizon Stock Price & Return Forecasting ")

    models, scaler, seq_len, all_metrics, best_model, residual_std = load_models()
    ticker, model_name, investment = render_sidebar(models)

    try:
        ticker_hist, window_df = load_live_ticker_data(ticker, seq_len)
    except (ConnectionError, ValueError) as e:
        st.error(f"⚠️ Could not fetch live data for {ticker}: {e}")
        st.stop()

    current_price = float(ticker_hist["Close"].iloc[-1])

    model = models[model_name]
    predicted_returns = predict_returns(model, window_df, scaler)
    ci_bounds = [
        confidence_interval(predicted_returns[i], residual_std[model_name][f"{h}d"])
        for i, h in enumerate(HORIZONS)
    ]

    render_price_and_forecast_cards(current_price, predicted_returns, ci_bounds)
    st.markdown("---")

    chart_key_base = f"{ticker}_{model_name}"

    tab_forecast, tab_indicators, tab_models, tab_portfolio = st.tabs(
        ["📊 Forecast", "📉 Technical Indicators", "🏆 Model Comparison", "💰 Portfolio & Signal"]
    )

    with tab_forecast:
        render_forecast_chart(ticker_hist, current_price, predicted_returns, ci_bounds,
                               chart_key=f"{chart_key_base}_forecast")
        render_download(ticker, model_name, current_price, predicted_returns, ci_bounds)

    with tab_indicators:
        render_indicator_tabs(ticker_hist, key_prefix=chart_key_base)

    with tab_models:
        render_model_comparison(all_metrics, best_model)

    with tab_portfolio:
        col_a, col_b = st.columns(2)
        with col_a:
            render_recommendation(ticker_hist, predicted_returns, ci_bounds)
        with col_b:
            render_portfolio_calculator(current_price, predicted_returns, investment)


if __name__ == "__main__":
    main()
