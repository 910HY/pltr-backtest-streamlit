import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Streamlit App UI
st.title("PLTR 回測平台")
st.markdown("這是一個簡單的回測平台，支援自動下載 PLTR 歷史數據並測試移動平均策略。")

# 策略參數
st.sidebar.header("策略參數")
short_window = st.sidebar.number_input("短期移動平均線 (日)", min_value=1, max_value=100, value=20)
long_window = st.sidebar.number_input("長期移動平均線 (日)", min_value=1, max_value=300, value=50)
interval = st.sidebar.selectbox("數據頻率", ["1d", "1h", "15m"], index=0)
start_date = st.sidebar.date_input("開始日期", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("結束日期", value=datetime.now())

# 下載數據
def load_data():
    df = yf.download("PLTR", start=start_date, end=end_date, interval=interval)
    df.dropna(inplace=True)
    return df

with st.spinner("下載資料中..."):
    data = load_data()

st.subheader("PLTR 價格走勢")
st.line_chart(data['Close'])

# 回測邏輯（修正過錯誤）
def run_backtest(df, short_window, long_window):
    df = df.copy()
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
    df['signal'] = df['signal'].fillna(0)
    df['position'] = df['signal'].diff()
    df['returns'] = df['Close'].pct_change()
    df['strategy'] = df['returns'] * df['signal'].shift(1)
    df['equity'] = (1 + df['strategy']).cumprod()
    return df

bt = run_backtest(data, short_window, long_window)

# 繪圖
st.subheader("交易信號")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(bt.index, bt['Close'], label='Close')
ax.plot(bt.index, bt['short_ma'], label=f'Short MA ({short_window})')
ax.plot(bt.index, bt['long_ma'], label=f'Long MA ({long_window})')
ax.plot(bt[bt['position'] == 1].index, bt['Close'][bt['position'] == 1], '^', markersize=10, color='g', label='Buy')
ax.plot(bt[bt['position'] == -1].index, bt['Close'][bt['position'] == -1], 'v', markersize=10, color='r', label='Sell')
ax.legend()
st.pyplot(fig)

# 報告
st.subheader("策略表現")
final_return = bt['equity'].iloc[-1] - 1
st.metric("總報酬率", f"{final_return:.2%}")
max_drawdown = (bt['equity'].cummax() - bt['equity']).max()
st.metric("最大回撤", f"{max_drawdown:.2%}")
