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
    df['short_ma'] = df['Close'].rolling(window=short_ma).mean()
    df['long_ma'] = df['Close'].rolling(window=long_ma).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['boll_mid'] = df['Close'].rolling(window=boll_window).mean()
    df['boll_std'] = df['Close'].rolling(window=boll_window).std()
    df['boll_upper'] = df['boll_mid'] + boll_std * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - boll_std * df['boll_std']

    # 趨勢線（線性回歸）
    if len(df) > 10:
        x = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        df['trend'] = model.predict(x)
    else:
        df['trend'] = df['Close']

    df['signal'] = np.where(
        (df['short_ma'] > df['long_ma']) &
        (df['RSI'] < 70) &
        (df['MACD'] > df['MACD_signal']) &
        (df['Close'] > df['boll_mid']) &
        (df['Close'] > df['trend']),
        1, 0)
    df['returns'] = df['Close'].pct_change()
    df['strategy'] = df['returns'] * df['signal'].shift(1)
    df['equity'] = START_EQUITY * (1 + df['strategy']).cumprod()

    end_equity = df['equity'].iloc[-1]
    final_return = (end_equity - START_EQUITY) / START_EQUITY
    max_drawdown = (df['equity'].cummax() - df['equity']).max() / df['equity'].cummax().max()
    annualized_return = df['strategy'].mean() * 252
    annualized_volatility = df['strategy'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    return {
        "短期MA": short_ma,
        "長期MA": long_ma,
        "RSI週期": rsi_period,
        "MACD快線": macd_fast,
        "MACD慢線": macd_slow,
        "MACD信號": macd_signal,
        "布林週期": boll_window,
        "布林倍數": boll_std,
        "總報酬率": final_return,
        "最大回撤": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }

if run_button:
    for ticker in selected_tickers:
        st.subheader(f"股票：{ticker}")
        df = load_data(ticker)
        result_list = []
        for short_ma, long_ma in product(
            range(short_ma_range[0], short_ma_range[1]+1, 10),
            range(long_ma_range[0], long_ma_range[1]+1, 20)):
            if short_ma >= long_ma:
                continue
            for rsi in range(rsi_range[0], rsi_range[1]+1, 5):
                for fast in range(macd_fast_range[0], macd_fast_range[1]+1, 2):
                    for slow in range(macd_slow_range[0], macd_slow_range[1]+1, 2):
                        if fast >= slow:
                            continue
                        for signal in range(macd_signal_range[0], macd_signal_range[1]+1, 1):
                            for boll_w in range(boll_window_range[0], boll_window_range[1]+1, 5):
                                for boll_s in np.arange(boll_std_range[0], boll_std_range[1]+0.1, 0.5):
                                    res = backtest(df, short_ma, long_ma, rsi, fast, slow, signal, boll_w, boll_s)
                                    result_list.append(res)

        results_df = pd.DataFrame(result_list)
        results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
        st.write("參數設定範圍：")
        st.code(f"短期 MA: {short_ma_range}, 長期 MA: {long_ma_range}, RSI: {rsi_range}, MACD: {macd_fast_range}/{macd_slow_range}/{macd_signal_range}, Boll: {boll_window_range}, Std: {boll_std_range}")
        st.dataframe(results_df)

        if not results_df.empty:
            best = results_df.iloc[0]
            st.success(f"最佳策略：短期 MA={best['短期MA']}, 長期 MA={best['長期MA']}, RSI={best['RSI週期']}, Sharpe={best['Sharpe Ratio']:.2f}")
