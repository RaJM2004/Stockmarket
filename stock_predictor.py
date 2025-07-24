import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

class StockPredictor:
    def __init__(self):
        self.model = None

    def fetch_data(self, symbol, period="6mo", interval="1d"):
        df = yf.download(symbol, period=period, interval=interval)
        df.dropna(inplace=True)
        return df

    def add_indicators(self, df):
        df['MA_20'] = df['Close'].rolling(window=20).mean()

        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0).ravel()
        loss = np.where(delta < 0, -delta, 0).ravel()

        avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df.dropna(inplace=True)
        return df


    def generate_signals(self, df):
        df['Signal_BuySell'] = np.where(df['MACD'] > df['Signal'], 1, 0)
        return df

    def prepare_features(self, df):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'RSI', 'MACD', 'Signal']
        X = df[features]
        y = df['Close'].shift(-1)
        return X[:-1], y[:-1]  # remove last row for training

    def train_model(self, X, y):
        self.model = XGBRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def predict_next(self, df):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'RSI', 'MACD', 'Signal']
        X_pred = df[features].tail(1)
        return self.model.predict(X_pred)[0]

    def get_recommendation(self, current, predicted):
    # Ensure both are scalar floats
        if isinstance(current, (pd.Series, np.ndarray)):
            current = current.item()
        if isinstance(predicted, (pd.Series, np.ndarray)):
            predicted = predicted.item()

        if predicted > current * 1.01:
            return "📈 BUY"
        elif predicted < current * 0.99:
            return "🔻 SELL"
        else:
            return "🟡 HOLD"
