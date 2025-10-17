# File: analyst_instant_tradingview.py
# Versi: 36.0 - MIGRASI KE TRADINGVIEW DATA (Anti-403)

import streamlit as st
import pandas as pd
import numpy as np
import time
from tvDatafeed import TvDatafeed, Interval

# --- INISIALISASI TRADINGVIEW ---
# Tidak perlu API key, hanya perlu koneksi internet.
tv = TvDatafeed()

# --- DAFTAR KOIN (SAMA SEPERTI BYBIT) ---
BASE_COIN_UNIVERSE = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT'
]
total_coins_scanned = len(BASE_COIN_UNIVERSE)

# --- STYLE UI ---
st.set_page_config(layout="wide", page_title="Instant AI Analyst (TradingView)", initial_sidebar_state="collapsed")
st.title("🚀 Instant AI Signal Dashboard (TradingView)")
st.caption(f"Menganalisis **{total_coins_scanned}+ koin utama** menggunakan **TradingView Datafeed**.")
st.markdown("---")

col1, col2 = st.columns([1.5, 1.5])
selected_tf = col1.selectbox("Pilih Timeframe Sinyal:", ['1d', '4h', '1h'])
if col2.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.rerun()

# --- PEMETAAN TIMEFRAME ---
TV_INTERVAL_MAP = {
    '1d': Interval.in_daily,
    '4h': Interval.in_4_hour,
    '1h': Interval.in_1_hour
}

# --- HEALTH CHECK (CEK DATA TRADINGVIEW) ---
@st.cache_data(ttl=60)
def tv_health_check():
    try:
        df = tv.get_hist(symbol='BTCUSDT', exchange='BINANCE', interval=Interval.in_1_hour, n_bars=5)
        if df is not None and not df.empty:
            return True
        return False
    except Exception as e:
        st.write("⚠️ Health Check Error:", str(e))
        return False

is_connected = tv_health_check()
if not is_connected:
    st.error("🔴 Tidak bisa menghubungi TradingView Datafeed. Pastikan koneksi internet stabil.")
    st.stop()
else:
    st.success("🟢 Koneksi ke TradingView berhasil!")

# --- FETCH DATA DARI TRADINGVIEW ---
@st.cache_data(show_spinner=True, ttl=120)
def fetch_tv_data(symbol, timeframe):
    try:
        df = tv.get_hist(symbol=symbol, exchange='BINANCE', interval=TV_INTERVAL_MAP[timeframe], n_bars=200)
        if df is not None and not df.empty:
            df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}, inplace=True)
            return df
        else:
            return None
    except:
        return None

# --- ANALISIS STRUKTUR PASAR (SAMA SEPERTI SEBELUMNYA) ---
def analyze_structure(df):
    if df is None or len(df) < 7:
        return {'structure': 'Data Tidak Cukup', 'current_price': None, 'fib_bias': 'Netral', 'change_pct': 0.0}
    structure = "Konsolidasi"
    current_close = df['Close'].iloc[-1]
    df_recent = df[['High', 'Low']].tail(14)
    if len(df_recent) >= 14:
        rh = df_recent['High'].rolling(5, center=True).max().dropna()
        rl = df_recent['Low'].rolling(5, center=True).min().dropna()
        if len(rh) >= 2 and len(rl) >= 2:
            if rh.iloc[-1] > rh.iloc[-2] and rl.iloc[-1] > rl.iloc[-2]:
                structure = "Bullish"
            elif rh.iloc[-1] < rh.iloc[-2] and rl.iloc[-1] < rl.iloc[-2]:
                structure = "Bearish"
    fib_bias = "Netral"
    max_price, min_price = df['High'].max(), df['Low'].min()
    diff = max_price - min_price
    fib_61 = max_price - (diff * 0.618)
    fib_38 = max_price - (diff * 0.382)
    if current_close > fib_61: fib_bias = "Bullish"
    elif current_close < fib_38: fib_bias = "Bearish"
    change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
    return {'structure': structure, 'current_price': current_close, 'fib_bias': fib_bias, 'change_pct': change_pct}

# --- PEMINDAIAN TRADINGVIEW ---
def run_scanner(coin_universe, timeframe):
    results = []
    status_placeholder = st.empty()
    for i, symbol in enumerate(coin_universe):
        df = fetch_tv_data(symbol, timeframe)
        analysis = analyze_structure(df)
        results.append({'symbol': symbol, **analysis})
        time.sleep(0.07)
        status_placeholder.info(f"Memindai {symbol} ({i+1}/{total_coins_scanned})")
    return results

# --- EKSEKUSI ---
st.info(f"Memulai pemindaian {total_coins_scanned} koin dari TradingView...")
start_time = time.time()
all_results = run_scanner(BASE_COIN_UNIVERSE, selected_tf)
total_time = time.time() - start_time
st.success(f"✅ Pemindaian selesai dalam {total_time:.2f} detik.")

# --- TAMPILKAN HASIL ---
df_res = pd.DataFrame(all_results)
st.subheader("📈 Hasil Struktur Pasar (TradingView Data)")
st.dataframe(df_res)
