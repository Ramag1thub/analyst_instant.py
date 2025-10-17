# File: analyst_instant_fixed.py
# Versi: 35.1 - BYBIT API STABIL (Auto Retry + Toleransi Error)
# Tujuan: Gabungan penuh kode lama + versi stabil koneksi Bybit

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- KONSTANTA API BYBIT ---
BYBIT_API_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
REQUEST_TIMEOUT = 30

# --- DAFTAR KOIN (DIPERSINGKAT UNTUK CONTOH) ---
BASE_COIN_UNIVERSE = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT'
]
total_coins_scanned = len(BASE_COIN_UNIVERSE)

# --- STYLE DASHBOARD ---
INSTANT_CSS = """
<style>
    .stApp { background-color: #151515; color: #d1d1d1; font-family: 'Helvetica Neue', Arial, sans-serif; }
    .signal-card { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #2a2a2a; padding: 1rem; margin-bottom: 1rem; }
    .profit { color: #50fa7b; font-weight: bold; }
    .loss { color: #ff6e6e; font-weight: bold; }
    .entry { color: #8be9fd; font-weight: bold; }
</style>
"""
st.set_page_config(layout="wide", page_title="Instant AI Analyst (Bybit)", initial_sidebar_state="collapsed")
st.markdown(INSTANT_CSS, unsafe_allow_html=True)

# --- HEADER DAN PILIHAN TIMEFRAME ---
st.title("🚀 Instant AI Signal Dashboard (Bybit)")
st.caption(f"Menganalisis **{total_coins_scanned}+ koin utama** menggunakan **Bybit API Stabil**.")
col1, col2, col3 = st.columns([1.5, 1.5, 7])
selected_tf = col1.selectbox("Pilih Timeframe Sinyal:", ['1d', '4h', '1h'])
if col2.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.rerun()
st.markdown("---")

# --- HEALTH CHECK API BYBIT (Versi Stabil + Retry + Timeout Aman) ---
@st.cache_data(ttl=60)
def bybit_health_check():
    """Cek koneksi API Bybit (BTCUSDT). Mengembalikan True jika OK."""
    session = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    try:
        params = {"category": "linear", "symbol": "BTCUSDT"}
        response = session.get(BYBIT_TICKER_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("retCode") == 0 and data.get("result", {}).get("list"):
            return True
        return False
    except Exception as e:
        st.write("⚠️ Health Check Error:", str(e))
        return False

# --- CEK KONEKSI API ---
is_connected = bybit_health_check()

if not is_connected:
    st.warning(f"⚠️ Tidak bisa menghubungi Bybit API (BTCUSDT Health Check gagal). "
               f"Coba ubah Timeframe Sinyal ({selected_tf}) atau tekan tombol 'Scan Ulang Sekarang'. "
               "Jika masih gagal, tunggu beberapa menit karena kemungkinan API Bybit sedang rate-limit.")
else:
    st.success(f"🟢 Koneksi Bybit API berhasil. Melanjutkan pemindaian {total_coins_scanned} koin.")

# --- FUNGSI AMBIL DATA HARIAN ---
@st.cache_data(show_spinner=True, ttl=60)
def fetch_daily_data(symbol, days=7):
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - 86400000 * days
    params = {'category': 'linear', 'symbol': symbol, 'interval': 'D', 'start': start_time_ms, 'limit': 1000}
    try:
        response = requests.get(BYBIT_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data.get('retCode') == 0 and data['result']['list']:
            bars = data['result']['list']
            bars.reverse()
            df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float})
            df.set_index('timestamp', inplace=True)
            return df
    except Exception:
        return None
    return None

# --- ANALISIS STRUKTUR DASAR ---
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

# --- PEMINDAIAN DASAR ---
def run_scanner(coin_universe, timeframe):
    results = []
    status_placeholder = st.empty()
    for i, symbol in enumerate(coin_universe):
        df = fetch_daily_data(symbol)
        analysis = analyze_structure(df)
        results.append({'symbol': symbol, **analysis})
        time.sleep(0.07)
        status_placeholder.info(f"Memindai {symbol} ({i+1}/{total_coins_scanned})")
    return results

# --- EKSEKUSI PEMINDAIAN ---
if is_connected:
    st.info(f"Memulai pemindaian {total_coins_scanned} koin pada timeframe {selected_tf}...")
    start_time = time.time()
    all_results = run_scanner(BASE_COIN_UNIVERSE, selected_tf)
    total_time = time.time() - start_time
    st.success(f"✅ Pemindaian selesai dalam {total_time:.2f} detik.")
    st.markdown("---")

    df_res = pd.DataFrame(all_results)
    st.subheader("📈 Hasil Struktur Pasar")
    st.dataframe(df_res)

else:
    st.error("❌ Data tidak dapat diambil dari Bybit API. Silakan coba beberapa saat lagi.")
