import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("綜合策略回測平台")

START_EQUITY = 10000

# 股票清單
tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AMD", "NFLX", "BAC",
    "JPM", "INTC", "XOM", "PFE", "T", "CSCO", "VZ", "WFC", "CVX", "QCOM",
    "KO", "PEP", "MRNA", "C", "DIS", "F", "GM", "PYPL", "ABNB", "CRM",
    "NKE", "BA", "BABA", "UBER", "ORCL", "ADBE", "SQ", "PLTR", "SNAP", "LYFT",
    "GOOG", "SHOP", "TSM", "MU", "GE", "SOFI", "ROKU", "TWLO", "TWTR", "ZM"
]

# Sidebar 選項
st.sidebar.header("參數設定")
selected_tickers = st.sidebar.multiselect("選擇股票（最多5隻）", tickers, default=["PLTR"], max_selections=5)
interval = st.sidebar.selectbox("頻率", ["1d", "1h"])
start_date = st.sidebar.date_input("開始日期", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("結束日期", datetime.today())

# 策略選擇
selected_strategies = st.sidebar.multiselect(
    "選擇策略模組", ["SMA", "RSI", "MACD", "Bollinger Bands", "Trendline"],
    default=["SMA", "RSI", "MACD"]
)

# 策略參數
st.sidebar.subheader("參數範圍")
short_ma = st.sidebar.slider("短期 MA", 5, 50, 20)
long_ma = st.sidebar.slider("長期 MA", 20, 200, 50)
rsi_period = st.sidebar.slider("RSI 週期", 5, 30, 14)
macd_fast = st.sidebar.slider("MACD 快線", 5, 20, 12)
macd_slow = st.sidebar.slider("MACD 慢線", 10, 40, 26)
macd_signal = st.sidebar.slider("MACD 信號線", 5, 20, 9)
boll_window = st.sidebar.slider("布林通道週期", 10, 40, 20)
boll_std = st.sidebar.slider("布林標準差", 1.0, 3.0, 2.0)

run_button = st.sidebar.button("提交回測")

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty and interval == "1h":
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.dropna(inplace=True)
    return df

def run_strategy(df):
    df = df.copy()
    if "SMA" in selected_strategies:
        df["short_ma"] = df["Close"].rolling(window=short_ma).mean()
        df["long_ma"] = df["Close"].rolling(window=long_ma).mean()
    if "RSI" in selected_strategies:
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
    if "MACD" in selected_strategies:
        ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
    if "Bollinger Bands" in selected_strategies:
        df["boll_mid"] = df["Close"].rolling(window=boll_window).mean()
        df["boll_std"] = df["Close"].rolling(window=boll_window).std()
        df["boll_upper"] = df["boll_mid"] + boll_std * df["boll_std"]
        df["boll_lower"] = df["boll_mid"] - boll_std * df["boll_std"]
    if "Trendline" in selected_strategies:
        x = np.arange(len(df)).reshape(-1, 1)
        y = df["Close"].values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        df["trend"] = model.predict(x)

    # 組合策略條件
    conditions = []
    if "SMA" in selected_strategies:
        conditions.append(df["short_ma"] > df["long_ma"])
    if "RSI" in selected_strategies:
        conditions.append(df["RSI"] < 70)
    if "MACD" in selected_strategies:
        conditions.append(df["MACD"] > df["MACD_signal"])
    if "Bollinger Bands" in selected_strategies:
        conditions.append(df["Close"] > df["boll_mid"])
    if "Trendline" in selected_strategies:
        conditions.append(df["Close"] > df["trend"])

    df["signal"] = np.where(np.logical_and.reduce(conditions), 1, 0) if conditions else 0
    df["returns"] = df["Close"].pct_change()
    df["strategy"] = df["returns"] * df["signal"].shift(1)
    df["equity"] = START_EQUITY * (1 + df["strategy"]).cumprod()
    return df

def benchmark_df():
    df = yf.download("SPY", start=start_date, end=end_date, interval=interval)
    if df.empty and interval == "1h":
        df = yf.download("SPY", start=start_date, end=end_date, interval="1d")
    df.dropna(inplace=True)
    df["spy_return"] = df["Close"].pct_change()
    df["spy_equity"] = START_EQUITY * (1 + df["spy_return"]).cumprod()
    return df

if run_button:
    for ticker in selected_tickers:
        st.subheader(f"{ticker} 策略回測")
        st.info("本次回測使用的策略模組包括：" + ", ".join(selected_strategies))

        df = load_data(ticker)
        bt = run_strategy(df)
        spy = benchmark_df()

        combined = pd.DataFrame({
            "策略資產": bt["equity"],
            "SPY 持有": spy["spy_equity"]
        }).dropna()

        # 指標計算
        final = bt["equity"].iloc[-1]
        max_dd = (bt["equity"].cummax() - bt["equity"]).max() / bt["equity"].cummax().max()
        ann_ret = bt["strategy"].mean() * 252
        ann_vol = bt["strategy"].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("總報酬率", f"{(final - START_EQUITY)/START_EQUITY:.2%}")
        col2.metric("最大回撤", f"{max_dd:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.line_chart(combined, height=350, use_container_width=True)
