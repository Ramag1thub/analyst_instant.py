# File: analyst_instant_fixed.py
# Versi: 35.2 - BYBIT API STABIL (Auto Retry + Fallback + Anti 403)
# Tujuan: Gabungan penuh kode lama + perbaikan stabil koneksi Bybit

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- KONSTANTA API BYBIT (DENGAN FALLBACK DOMAIN) ---
BYBIT_API_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_TICKER_URL = "https://api.bybit.com/v5/market/tickers"
BYBIT_MIRROR_API_URL = "https://api.bytick.com/v5/market/kline"
BYBIT_MIRROR_TICKER_URL = "https://api.bytick.com/v5/market/tickers"
REQUEST_TIMEOUT = 30

# --- DAFTAR KOIN (DIPERSINGKAT UNTUK DEMO) ---
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

# --- HEADER ---
st.title("🚀 Instant AI Signal Dashboard (Bybit)")
st.caption(f"Menganalisis **{total_coins_scanned}+ koin utama** menggunakan **Bybit API Stabil (Auto Retry + Fallback)**.")
col1, col2, col3 = st.columns([1.5, 1.5, 7])
selected_tf = col1.selectbox("Pilih Timeframe Sinyal:", ['1d', '4h', '1h'])
if col2.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.rerun()
st.markdown("---")

# ====================================================
#  🔹 BYBIT HEALTH CHECK (AUTO RETRY + USER-AGENT + FALLBACK)
# ====================================================
@st.cache_data(ttl=60)
def bybit_health_check():
    """Cek koneksi API Bybit (BTCUSDT). Mengembalikan True jika OK."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0 Safari/537.36"
        )
    }

    def try_url(url):
        try:
            params = {"category": "linear", "symbol": "BTCUSDT"}
            r = session.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                return True
        except Exception as e:
            st.write("⚠️ Health Check Error:", str(e))
        return False

    # Coba API utama, jika gagal pakai mirror
    if try_url(BYBIT_TICKER_URL):
        return True
    elif try_url(BYBIT_MIRROR_TICKER_URL):
        st.info("ℹ️ Menggunakan mirror API: api.bytick.com (Bybit Mirror).")
        return True
    return False

# --- CEK KONEKSI API ---
is_connected = bybit_health_check()
if not is_connected:
    st.warning(f"⚠️ Tidak bisa menghubungi Bybit API (BTCUSDT Health Check gagal). "
               f"Coba ubah Timeframe Sinyal ({selected_tf}) atau tekan tombol 'Scan Ulang Sekarang'. "
               "Jika masih gagal, tunggu beberapa menit karena kemungkinan API Bybit sedang membatasi koneksi.")
else:
    st.success(f"🟢 Koneksi Bybit API berhasil. Melanjutkan pemindaian {total_coins_scanned} koin.")

# ====================================================
#  🔹 FETCH DATA HARIAN (DENGAN FALLBACK JUGA)
# ====================================================
@st.cache_data(show_spinner=True, ttl=60)
def fetch_daily_data(symbol, days=7):
    """Mengambil data harian dari Bybit (fallback ke mirror jika gagal)."""
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - 86400000 * days
    params = {'category': 'linear', 'symbol': symbol, 'interval': 'D', 'start': start_time_ms, 'limit': 1000}

    def get_data(url):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            d = r.json()
            if d.get('retCode') == 0 and d['result']['list']:
                bars = d['result']['list']
                bars.reverse()
                df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float})
                df.set_index('timestamp', inplace=True)
                return df
        except:
            return None
        return None

    df = get_data(BYBIT_API_URL)
    if df is None:
        df = get_data(BYBIT_MIRROR_API_URL)
    return df

# ====================================================
#  🔹 ANALISIS STRUKTUR PASAR DASAR
# ====================================================
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

# ====================================================
#  🔹 PROSES PEMINDAIAN DASAR
# ====================================================
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

# ====================================================
#  🔹 EKSEKUSI UTAMA
# ====================================================
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
    st.error("❌ Data tidak dapat diambil dari Bybit API maupun mirror. Silakan coba beberapa saat lagi.")
