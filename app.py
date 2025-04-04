import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from itertools import product

# 標題
st.title("美股回測平台：策略參數優化")
st.markdown("支援多股票回測、自動尋找最佳移動平均參數（SMA 交叉策略）。")

START_EQUITY = 10000

# 股票池（最多成交前50）
most_active_stocks = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AMD", "NFLX", "BAC",
    "JPM", "INTC", "XOM", "PFE", "T", "CSCO", "VZ", "WFC", "CVX", "QCOM",
    "KO", "PEP", "MRNA", "C", "DIS", "F", "GM", "PYPL", "ABNB", "CRM",
    "NKE", "BA", "BABA", "UBER", "ORCL", "ADBE", "SQ", "PLTR", "SNAP", "LYFT",
    "GOOG", "SHOP", "TSM", "MU", "GE", "SOFI", "ROKU", "TWLO", "TWTR", "ZM"
]

# Sidebar 選項
st.sidebar.header("參數設定")
selected_tickers = st.sidebar.multiselect("選擇最多 5 隻股票", most_active_stocks, default=["PLTR"], max_selections=5)
interval = st.sidebar.selectbox("數據頻率", ["1d", "1h"], index=0)
start_date = st.sidebar.date_input("開始日期", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("結束日期", value=datetime.now())

# SMA 測試範圍
st.sidebar.markdown("---")
st.sidebar.subheader("測試短期 / 長期 MA 組合")
short_ma_range = st.sidebar.slider("短期 MA 範圍", 5, 100, (10, 50), step=10)
long_ma_range = st.sidebar.slider("長期 MA 範圍", 20, 300, (60, 200), step=20)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty and interval == "1h":
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.dropna(inplace=True)
    return df

def sma_backtest(df, short_ma, long_ma):
    df = df.copy()
    df['short_ma'] = df['Close'].rolling(window=short_ma).mean()
    df['long_ma'] = df['Close'].rolling(window=long_ma).mean()
    df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
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
        "總報酬率": final_return,
        "最大回撤": max_drawdown,
        "Sharpe Ratio": sharpe_ratio
    }

# 執行每隻股票測試
for ticker in selected_tickers:
    st.header(f"最佳參數測試：{ticker}")
    df = load_data(ticker)
    if df.empty:
        st.warning(f"{ticker} 無法下載資料")
        continue

    result_list = []
    for short_ma, long_ma in product(
        range(short_ma_range[0], short_ma_range[1]+1, 10),
        range(long_ma_range[0], long_ma_range[1]+1, 20)
    ):
        if short_ma >= long_ma:
            continue
        res = sma_backtest(df, short_ma, long_ma)
        result_list.append(res)

    results_df = pd.DataFrame(result_list)
    results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False)
    st.dataframe(results_df.reset_index(drop=True))
