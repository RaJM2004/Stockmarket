import streamlit as st
import plotly.graph_objs as go
from stock_predictor import StockPredictor

st.set_page_config(page_title="📊 Stock Predictor", layout="wide")
st.title("🔮 Stock Price Prediction with Technical Analysis")

symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "5m"], index=0)

predictor = StockPredictor()

# Load data
data_load_state = st.text("Loading stock data...")
df = predictor.fetch_data(symbol, period, interval)
df = predictor.add_indicators(df)
df = predictor.generate_signals(df)
data_load_state.text("")

# Chart
st.subheader(f"📈 {symbol.upper()} Stock Chart with Indicators")

fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name="MA 20", line=dict(color='orange')))
st.plotly_chart(fig, use_container_width=True)

# RSI/MACD
col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 RSI")
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple')))
    rsi_fig.update_layout(height=300)
    st.plotly_chart(rsi_fig, use_container_width=True)

with col2:
    st.subheader("📊 MACD")
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal'))
    macd_fig.update_layout(height=300)
    st.plotly_chart(macd_fig, use_container_width=True)

# Prediction
st.subheader("🤖 Predict Next Price & Recommendation")

X, y = predictor.prepare_features(df)
predictor.train_model(X, y)
predicted_price = predictor.predict_next(df)
current_price = df['Close'].iloc[-1]
recommendation = predictor.get_recommendation(current_price, predicted_price)

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price.item():.2f}")
col2.metric("Predicted Price", f"${predicted_price.item():.2f}")

col3.metric("Recommendation", recommendation)
