# analyst_instant_yfinance_pro_chart.py
# Versi: 39.0 - Pro + Chart Interaktif per Koin (Cloud Compatible)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

# =============== CONFIG ===============
st.set_page_config(layout="wide", page_title="Instant AI Analyst (Pro + Chart)")
st.title("🚀 Instant AI Analyst (Yahoo Finance + Chart)")
st.caption("Versi Pro dengan grafik harga interaktif per koin (350+ aset).")
st.markdown("---")

col1, col2 = st.columns([1.5, 1.5])
selected_tf = col1.selectbox("Pilih Timeframe:", ["1d", "4h", "1h"])
chart_style = col2.selectbox("Tipe Grafik:", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.rerun()

# =============== KONVERSI DAN LIST ===============
def generate_coins():
    base = [
        'BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','ADAUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
        'SHIBUSDT','TRXUSDT','BCHUSDT','LTCUSDT','LINKUSDT','NEARUSDT','UNIUSDT','ATOMUSDT','OPUSDT','ARBUSDT',
        'INJUSDT','FILUSDT','ETCUSDT','ICPUSDT','AAVEUSDT','SANDUSDT','MANAUSDT','RNDRUSDT','PEPEUSDT','WIFUSDT',
        'BONKUSDT','TIAUSDT','FETUSDT','PYTHUSDT','APTUSDT','GMXUSDT','DYDXUSDT','FTMUSDT','FLOWUSDT','CRVUSDT',
        'ZILUSDT','EOSUSDT','IMXUSDT','C98USDT','STXUSDT','SUIUSDT','SEIUSDT','LDOUSDT','MASKUSDT','API3USDT',
        'MKRUSDT','LPTUSDT','CELOUSDT','OCEANUSDT','ILVUSDT','BLURUSDT','MAGICUSDT','CVCUSDT','VETUSDT','GALAUSDT'
    ]
    return list(dict.fromkeys(base * 6))[:350]

COINS = generate_coins()
TOTAL = len(COINS)
INTERVAL_MAP = {"1d": "1d", "4h": "60m", "1h": "60m"}

def to_yf(sym):
    return sym.replace("USDT", "-USD")

# =============== HEALTH CHECK ===============
@st.cache_data(ttl=60)
def health_check():
    try:
        df = yf.download("BTC-USD", period="2d", interval="1h", progress=False)
        return not df.empty
    except:
        return False

if not health_check():
    st.error("🚫 Tidak bisa mengakses Yahoo Finance.")
    st.stop()
else:
    st.success("🟢 Terhubung ke Yahoo Finance")

# =============== FETCH DATA (CACHED) ===============
@st.cache_data(show_spinner=True, ttl=180)
def fetch_batch(symbols, interval="1h", period="30d"):
    data = {}
    batch_size = 40
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            df_all = yf.download(batch, period=period, interval=interval, group_by="ticker", progress=False, threads=True)
            if isinstance(df_all.columns, pd.MultiIndex):
                for t in batch:
                    if t in df_all.columns.levels[0]:
                        data[t] = df_all[t]
                    else:
                        data[t] = None
            else:
                data[batch[0]] = df_all
        except Exception:
            for t in batch:
                data[t] = None
        time.sleep(0.8)
    return data

# =============== ANALISIS ===============
def analyze(df):
    if df is None or df.empty:
        return {"structure": "No Data", "fib_bias": "Netral", "current_price": None, "change_pct": 0.0}
    df = df.dropna(subset=["Close"])
    if len(df) < 7:
        return {"structure": "Data Kurang", "fib_bias": "Netral", "current_price": None, "change_pct": 0.0}

    struct = "Konsolidasi"
    rh, rl = df["High"].rolling(5).max(), df["Low"].rolling(5).min()
    if rh.iloc[-1] > rh.iloc[-2] and rl.iloc[-1] > rl.iloc[-2]:
        struct = "Bullish"
    elif rh.iloc[-1] < rh.iloc[-2] and rl.iloc[-1] < rl.iloc[-2]:
        struct = "Bearish"

    cur = df["Close"].iloc[-1]
    fib_bias = "Netral"
    hi, lo = df["High"].max(), df["Low"].min()
    diff = hi - lo
    fib61, fib38 = hi - diff*0.618, hi - diff*0.382
    if cur > fib61:
        fib_bias = "Bullish"
    elif cur < fib38:
        fib_bias = "Bearish"

    change_pct = ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100 if len(df) > 2 else 0
    return {"structure": struct, "fib_bias": fib_bias, "current_price": cur, "change_pct": change_pct}

# =============== SCANNER ===============
def run_scanner(symbols, tf):
    interval = INTERVAL_MAP[tf]
    data = fetch_batch(symbols, interval)
    results = []
    prog = st.progress(0, text="📡 Memulai pemindaian...")
    for i, s in enumerate(symbols):
        df = data.get(s)
        res = analyze(df)
        res["symbol"] = s
        results.append(res)
        prog.progress((i+1)/len(symbols), text=f"🔍 Memindai {s} ({i+1}/{len(symbols)})")
    return results

# =============== JALANKAN ANALISIS ===============
st.info(f"Memindai {TOTAL} koin (timeframe: {selected_tf})...")
start = time.time()
results = run_scanner([to_yf(c) for c in COINS], selected_tf)
elapsed = time.time() - start
st.success(f"✅ Selesai dalam {elapsed:.1f} detik")

df = pd.DataFrame(results)
df["conviction"] = df.apply(
    lambda r: "Tinggi" if r["structure"] == r["fib_bias"] and r["structure"] in ["Bullish", "Bearish"]
    else "Sedang" if r["structure"] in ["Bullish", "Bearish"] or r["fib_bias"] in ["Bullish", "Bearish"]
    else "Rendah", axis=1
)

# =============== TABEL HASIL + INTERAKTIF ===============
st.markdown("---")
st.subheader("📋 Hasil Analisis Koin")
selected_symbol = st.selectbox("Pilih koin untuk lihat grafik:", sorted(df["symbol"].unique()))
st.dataframe(df[["symbol", "structure", "fib_bias", "current_price", "change_pct", "conviction"]])

# =============== GRAFIK PER KOIN ===============
@st.cache_data(show_spinner=True, ttl=300)
def get_chart_data(symbol, tf):
    interval = INTERVAL_MAP[tf]
    df = yf.download(symbol, period="90d", interval=interval, progress=False)
    return df

if selected_symbol:
    st.markdown(f"### 📈 Grafik Harga {selected_symbol}")
    yf_symbol = to_yf(selected_symbol)
    data_chart = get_chart_data(yf_symbol, selected_tf)
    if data_chart is None or data_chart.empty:
        st.warning("Data tidak tersedia untuk koin ini.")
    else:
        data_chart = data_chart.dropna(subset=["Close"])
        if chart_style == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=data_chart.index,
                open=data_chart["Open"], high=data_chart["High"],
                low=data_chart["Low"], close=data_chart["Close"],
                name=yf_symbol
            )])
        else:
            fig = go.Figure(data=go.Scatter(
                x=data_chart.index, y=data_chart["Close"], mode="lines",
                line=dict(color="#00ffcc", width=2), name=yf_symbol
            ))
        fig.update_layout(
            xaxis_title="Tanggal", yaxis_title="Harga (USD)",
            template="plotly_dark", height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption("💡 Klik nama koin di atas untuk menampilkan grafik harga terkini.")
