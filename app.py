import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from itertools import product
from sklearn.linear_model import LinearRegression

st.title("綜合策略參數優化平台")

START_EQUITY = 10000

most_active_stocks = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AMD", "NFLX", "BAC",
    "JPM", "INTC", "XOM", "PFE", "T", "CSCO", "VZ", "WFC", "CVX", "QCOM",
    "KO", "PEP", "MRNA", "C", "DIS", "F", "GM", "PYPL", "ABNB", "CRM",
    "NKE", "BA", "BABA", "UBER", "ORCL", "ADBE", "SQ", "PLTR", "SNAP", "LYFT",
    "GOOG", "SHOP", "TSM", "MU", "GE", "SOFI", "ROKU", "TWLO", "TWTR", "ZM"
]

# Sidebar
st.sidebar.header("參數設定")
selected_tickers = st.sidebar.multiselect("選擇最多 5 隻股票", most_active_stocks, default=["PLTR"], max_selections=5)
interval = st.sidebar.selectbox("數據頻率", ["1d", "1h"], index=0)
start_date = st.sidebar.date_input("開始日期", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("結束日期", value=datetime.now())

st.sidebar.subheader("策略選擇")
selected_strategies = st.sidebar.multiselect(
    "選擇要納入的策略", ["SMA", "RSI", "MACD", "Bollinger Bands", "Trendline"], default=["SMA", "RSI", "MACD"])

st.sidebar.subheader("參數範圍")
short_ma_range = st.sidebar.slider("短期 MA", 5, 100, (10, 50), step=10)
long_ma_range = st.sidebar.slider("長期 MA", 20, 300, (60, 200), step=20)
rsi_range = st.sidebar.slider("RSI 週期", 5, 30, (10, 20), step=5)
macd_fast_range = st.sidebar.slider("MACD 快線", 5, 20, (10, 12), step=2)
macd_slow_range = st.sidebar.slider("MACD 慢線", 10, 40, (20, 26), step=2)
macd_signal_range = st.sidebar.slider("MACD 信號線", 5, 20, (9, 9), step=1)
boll_window_range = st.sidebar.slider("布林週期", 10, 40, (20, 20), step=1)
boll_std_range = st.sidebar.slider("布林標準差", 1.0, 3.0, (2.0, 2.0), step=0.5)

run_button = st.sidebar.button("提交開始回測")

@st.cache_data

def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty and interval == "1h":
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.dropna(inplace=True)
    return df

def backtest(df, short_ma, long_ma, rsi_period, macd_fast, macd_slow, macd_signal, boll_window, boll_std):
    df = df.copy()

    if "SMA" in selected_strategies:
        df['short_ma'] = df['Close'].rolling(window=short_ma).mean()
        df['long_ma'] = df['Close'].rolling(window=long_ma).mean()
    if "RSI" in selected_strategies:
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
        avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    if "MACD" in selected_strategies:
        ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    if "Bollinger Bands" in selected_strategies:
        df['boll_mid'] = df['Close'].rolling(window=boll_window).mean()
        df['boll_std'] = df['Close'].rolling(window=boll_window).std()
        df['boll_upper'] = df['boll_mid'] + boll_std * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - boll_std * df['boll_std']
    if "Trendline" in selected_strategies:
        x = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        df['trend'] = model.predict(x)

    # 建立信號條件
    conditions = []
    if "SMA" in selected_strategies:
        conditions.append(df['short_ma'] > df['long_ma'])
    if "RSI" in selected_strategies:
        conditions.append(df['RSI'] < 70)
    if "MACD" in selected_strategies:
        conditions.append(df['MACD'] > df['MACD_signal'])
    if "Bollinger Bands" in selected_strategies:
        conditions.append(df['Close'] > df['boll_mid'])
    if "Trendline" in selected_strategies:
        conditions.append(df['Close'] > df['trend'])

    if conditions:
        df['signal'] = np.where(np.logical_and.reduce(conditions), 1, 0)
    else:
        df['signal'] = 0

    df['returns'] = df['Close'].pct_change()
    df['strategy'] = df['returns'] * df['signal'].shift(1)
    df['equity'] = START_EQUITY * (1 + df['strategy']).cumprod()

    end_equity = df['equity'].iloc[-1]
    final_return = (end_equity - START_EQUITY) / START_EQUITY
    max_drawdown = (df['equity'].cummax() - df['equity']).max() / df['equity'].cummax().max()
    annualized_return = df['strategy'].mean() * 252
    annualized_volatility = df['strategy'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    return df, {
        "總報酬率": final_return,
        "最大回撤": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }

if run_button:
    for ticker in selected_tickers:
        st.subheader(f"股票：{ticker}")
        df = load_data(ticker)
        df, stats = backtest(df,
            short_ma_range[0], long_ma_range[1],
            rsi_range[0], macd_fast_range[0], macd_slow_range[1], macd_signal_range[0],
            boll_window_range[0], boll_std_range[0]
        )

        st.metric("總報酬率", f"{stats['總報酬率']:.2%}")
        st.metric("最大回撤", f"{stats['最大回撤']:.2%}")
        st.metric("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}")

        # 對比 SPY benchmark
        spy_df = yf.download("SPY", start=start_date, end=end_date, interval=interval)
        if spy_df.empty and interval == "1h":
            spy_df = yf.download("SPY", start=start_date, end=end_date, interval="1d")
        spy_df.dropna(inplace=True)
        spy_df['spy_return'] = spy_df['Close'].pct_change()
        spy_df['spy_equity'] = START_EQUITY * (1 + spy_df['spy_return']).cumprod()

        common_index = df.index.intersection(spy_df.index)
        equity_compare = pd.DataFrame({
            '策略資產': df.loc[common_index, 'equity'],
            'SPY 持有': spy_df.loc[common_index, 'spy_equity']
        })

        st.line_chart(equity_compare, height=350, use_container_width=True)
