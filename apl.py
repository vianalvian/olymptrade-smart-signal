# app.py
import streamlit as st
import pandas as pd
import requests
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="OlympTrade Smart Signal (Auto Mode)", layout="wide")

# ========== CONFIG ==========
API_KEY = st.secrets["API_KEY"]
PAIR_LIST = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "BTC/USD"]
INTERVAL = "1min"

# ========== FUNCTION ==========
def get_data(symbol):
    """Ambil data harga dari TwelveData"""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize=60&apikey={API_KEY}"
    r = requests.get(url).json()
    if "values" not in r:
        return None
    df = pd.DataFrame(r["values"])
    df = df.astype({"open":"float","close":"float","high":"float","low":"float"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df

def calc_indicators(df):
    """Hitung RSI, SMA, MACD, Bollinger"""
    df["SMA14"] = df["close"].rolling(window=14).mean()
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["UpperBB"] = df["SMA14"] + 2*df["close"].rolling(window=20).std()
    df["LowerBB"] = df["SMA14"] - 2*df["close"].rolling(window=20).std()

    return df

def generate_signal(df):
    """Logika sinyal otomatis"""
    latest = df.iloc[-1]
    signal = "WAIT"
    if latest["RSI"] < 30 and latest["close"] < latest["SMA14"]:
        signal = "BUY"
    elif latest["RSI"] > 70 and latest["close"] > latest["SMA14"]:
        signal = "SELL"
    elif latest["MACD"] > latest["Signal"] and latest["close"] > latest["SMA14"]:
        signal = "BUY"
    elif latest["MACD"] < latest["Signal"] and latest["close"] < latest["SMA14"]:
        signal = "SELL"
    return signal

# ========== APP ==========
st.title("📊 OlympTrade Smart Signal (Auto Mode)")
st.caption("Sinyal otomatis berdasarkan RSI, SMA, MACD, dan Bollinger Bands — update setiap 1 menit")

placeholder = st.empty()

while True:
    data_list = []
    for pair in PAIR_LIST:
        df = get_data(pair)
        if df is None:
            continue
        df = calc_indicators(df)
        signal = generate_signal(df)
        price = df.iloc[-1]["close"]
        data_list.append({"Pair": pair, "Price": round(price, 5), "Signal": signal})
    
    result_df = pd.DataFrame(data_list)
    placeholder.table(result_df)

    buy_pairs = result_df[result_df["Signal"] == "BUY"]["Pair"].tolist()
    sell_pairs = result_df[result_df["Signal"] == "SELL"]["Pair"].tolist()

    if buy_pairs:
        st.toast(f"🟢 BUY signal untuk: {', '.join(buy_pairs)}", icon="✅")
    if sell_pairs:
        st.toast(f"🔴 SELL signal untuk: {', '.join(sell_pairs)}", icon="⚠️")

    time.sleep(60)  # update tiap 1 menit
