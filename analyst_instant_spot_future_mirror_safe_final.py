# analyst_instant_spot_structure_fib_support_final.py
# Spot Only — menampilkan struktur, fibonacci bias, support/resistance, conviction
# 100% aman di Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide", page_title="AI Analyst Spot — Struktur & Fibo")
st.title("🚀 Instant AI Analyst — Struktur, Fibonacci, Support/Resistance (Spot Only)")
st.caption("Yahoo Mirror • EMA/RSI/Bollinger • Analisa lengkap tanpa error")
st.markdown("---")

col1, col2 = st.columns([2, 8])
tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.experimental_rerun()

COINS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","DOGEUSDT","PEPEUSDT",
         "ARBUSDT","OPUSDT","AVAXUSDT","MATICUSDT","TIAUSDT","LINKUSDT","RNDRUSDT"]
TOTAL = len(COINS)

def to_yf(sym): return sym[:-4] + "-USD" if sym.endswith("USDT") else sym
YF_INTERVAL = {"1d": "1d", "4h": "60m", "1h": "60m"}

# ===========================
# SAFE ANALYZE
# ===========================
def analyze_df(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,
                "current":None,"change":0.0,"conviction":"Rendah"}

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        else:
            return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,
                    "current":None,"change":0.0,"conviction":"Rendah"}

    df = df.dropna(subset=["Close"], errors="ignore")
    if len(df) < 5:
        return {"structure":"Data Kurang","fib_bias":"Netral","support":None,"resistance":None,
                "current":None,"change":0.0,"conviction":"Rendah"}

    for c in ["High", "Low"]:
        if c not in df.columns:
            df[c] = df["Close"]

    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else cur
    change = ((cur - prev) / prev * 100) if prev != 0 else 0.0

    hi, lo = float(df["High"].max()), float(df["Low"].min())
    diff = hi - lo if hi and lo else 0.0
    fib_bias = "Netral"
    if diff > 0:
        fib61, fib38 = hi - diff * 0.618, hi - diff * 0.382
        if cur > fib61: fib_bias = "Bullish"
        elif cur < fib38: fib_bias = "Bearish"

    # Struktur harga
    struct = "Konsolidasi"
    try:
        rec = df[["High", "Low"]].tail(20)
        if len(rec) >= 10:
            hh = rec["High"].rolling(5).max().dropna()
            ll = rec["Low"].rolling(5).min().dropna()
            if len(hh) >= 2 and len(ll) >= 2:
                if hh.iloc[-1] > hh.iloc[-2] and ll.iloc[-1] > ll.iloc[-2]:
                    struct = "Bullish"
                elif hh.iloc[-1] < hh.iloc[-2] and ll.iloc[-1] < ll.iloc[-2]:
                    struct = "Bearish"
    except Exception:
        pass

    # Support & Resistance
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]

    # Conviction
    if struct in ["Bullish","Bearish"] and fib_bias == struct:
        conviction = "Tinggi"
    elif struct in ["Bullish","Bearish"] or fib_bias in ["Bullish","Bearish"]:
        conviction = "Sedang"
    else:
        conviction = "Rendah"

    return {"structure":struct,"fib_bias":fib_bias,"support":support,
            "resistance":resistance,"current":cur,"change":change,"conviction":conviction}

# ===========================
# FETCH + SCAN
# ===========================
@st.cache_data(ttl=120, show_spinner=True)
def fetch(symbols, interval):
    data = {}
    for s in symbols:
        try:
            df = yf.download(to_yf(s), period="90d", interval=interval, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                data[s] = df
        except Exception:
            continue
    return data

st.info(f"⏳ Memindai {TOTAL} koin spot (Yahoo Mirror)...")
start = time.time()
data = fetch(COINS, YF_INTERVAL[tf])
rows=[]
for i,(s,df) in enumerate(data.items()):
    res = analyze_df(df)
    res["symbol"]=s
    rows.append(res)
    st.progress((i+1)/len(data), text=f"📊 {s}")
    time.sleep(0.01)
elapsed = time.time()-start
st.success(f"✅ Selesai dalam {elapsed:.1f} detik")

df = pd.DataFrame(rows)
st.dataframe(df[["symbol","structure","fib_bias","support","resistance","change","conviction"]]
             .sort_values("change", ascending=False).reset_index(drop=True))

# ===========================
# CHART
# ===========================
symbol = st.selectbox("Pilih koin untuk grafik:", sorted(df["symbol"].unique()))
try:
    chart = yf.download(to_yf(symbol), period="120d", interval=YF_INTERVAL[tf], progress=False)
except Exception:
    chart = None

if chart is None or chart.empty:
    st.warning("⚠️ Data grafik tidak tersedia.")
else:
    chart["EMA20"]=chart["Close"].ewm(span=20).mean()
    chart["EMA50"]=chart["Close"].ewm(span=50).mean()
    ma = chart["Close"].rolling(20).mean()
    sd = chart["Close"].rolling(20).std()
    chart["BB_up"]=ma+2*sd; chart["BB_dn"]=ma-2*sd

    last = analyze_df(chart)
    fig = go.Figure()
    if chart_style == "Candlestick":
        fig.add_trace(go.Candlestick(x=chart.index, open=chart["Open"], high=chart["High"],
                                     low=chart["Low"], close=chart["Close"], name="Price"))
    else:
        fig.add_trace(go.Scatter(x=chart.index, y=chart["Close"], mode="lines", name="Close"))
    for i in ["EMA20","EMA50"]:
        if i in chart.columns:
            fig.add_trace(go.Scatter(x=chart.index, y=chart[i], mode="lines", name=i))
    if {"BB_up","BB_dn"}.issubset(chart.columns):
        fig.add_trace(go.Scatter(x=chart.index, y=chart["BB_up"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=chart.index, y=chart["BB_dn"], fill="tonexty", line=dict(width=0), showlegend=False))

    # Tambahkan level Support & Resistance
    if last["support"]:
        fig.add_hline(y=last["support"], line=dict(color="green", width=1, dash="dot"), annotation_text="Support")
    if last["resistance"]:
        fig.add_hline(y=last["resistance"], line=dict(color="red", width=1, dash="dot"), annotation_text="Resistance")

    fig.update_layout(template="plotly_dark", height=520,
                      title=f"{symbol} | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Analisa: Struktur, Fibonacci Bias, Support/Resistance, Conviction • Data Spot (Yahoo Mirror)")
