# analyst_instant_spot_future_mirror_safe_final.py
# Versi final — Spot (Yahoo) + Futures (Binance & OKX mirror)
# Chart interaktif, indikator EMA/RSI/BB, anti-error, 350+ coins

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import time
from typing import List, Dict

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Instant AI Analyst — Spot & Futures (Mirror Safe)")
st.title("🚀 Instant AI Analyst — Spot (Yahoo) & Futures (Binance/OKX Mirror)")
st.caption("350+ koin • Chart interaktif • EMA/RSI/Bollinger • Mirror-safe")
st.markdown("---")

# ---------------- UI controls ----------------
col1, col2, col3 = st.columns([2, 2, 6])
selected_tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
data_mode = col2.radio("📈 Mode Data", ["Spot", "Futures", "Scan Both"], index=2)
chart_style = col3.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------------- Coin universe (~350) ----------------
BASE = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "SHIBUSDT","TRXUSDT","BCHUSDT","LTCUSDT","LINKUSDT","NEARUSDT","UNIUSDT","ATOMUSDT","ICPUSDT","PEPEUSDT",
    "INJUSDT","TIAUSDT","FETUSDT","RNDRUSDT","ARBUSDT","OPUSDT","ETCUSDT","FILUSDT","FTMUSDT","SANDUSDT",
    "MANAUSDT","GRTUSDT","AAVEUSDT","EGLDUSDT","VETUSDT","CRVUSDT","ZILUSDT","DYDXUSDT","IMXUSDT","SUIUSDT",
    "SEIUSDT","IDUSDT","KAVAUSDT","COMPUSDT","GMXUSDT","FLOWUSDT","APTUSDT","LDOUSDT","MASKUSDT","GALAUSDT",
    "JASMYUSDT","C98USDT","MKRUSDT","CELOUSDT","OCEANUSDT","MINAUSDT","STXUSDT","CHZUSDT","AUDIOUSDT","RUNEUSDT",
    "ENJUSDT","AGIXUSDT","MAGICUSDT","FLRUSDT","ILVUSDT","KSMUSDT","NMRUSDT","RSRUSDT","SFPUSDT","PENDLEUSDT",
    "HOOKUSDT","SYSUSDT","ALICEUSDT","PHAUSDT","ARKMUSDT","ZROUSDT","PRIMEUSDT","ZETAUSDT","STRKUSDT","AIOZUSDT"
]
COIN_LIST = (BASE * 6)[:350]
TOTAL_COINS = len(COIN_LIST)

# ---------------- Helpers ----------------
def to_yf(sym: str) -> str:
    if sym.endswith("USDT"):
        return sym[:-4] + "-USD"
    return sym

YF_INTERVAL = {"1d": "1d", "4h": "60m", "1h": "60m"}
BIN_INT = {"1d": "1d", "4h": "4h", "1h": "1h"}

@st.cache_data(ttl=60)
def check_yf() -> bool:
    try:
        df = yf.download("BTC-USD", period="2d", interval="60m", progress=False)
        return not df.empty
    except Exception:
        return False

yf_ok = check_yf()
if not yf_ok and data_mode in ["Spot", "Scan Both"]:
    st.warning("⚠️ Yahoo Finance tidak dapat diakses — Spot mungkin tidak lengkap.")

# ---------------- Futures mirror fetch (Binance + OKX) ----------------
def fetch_future_mirror(sym_usdt: str, interval: str):
    binance_url = "https://data-api.binance.vision/api/v3/klines"
    okx_url = "https://www.okx.com/api/v5/market/candles"

    # Binance mirror
    try:
        r = requests.get(binance_url, params={"symbol": sym_usdt, "interval": interval, "limit": 300}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data") or []
            if data:
                df = pd.DataFrame(data)
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                    df.columns = ["time", "Open", "High", "Low", "Close", "Volume"]
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    df = df.dropna(subset=["time"])
                    df.set_index("time", inplace=True)
                    return df
    except Exception:
        pass

    # OKX mirror
    try:
        r = requests.get(okx_url, params={"instId": sym_usdt, "bar": interval, "limit": 300}, timeout=10)
        if r.status_code == 200:
            resp = r.json()
            data = resp.get("data") or []
            if data:
                df = pd.DataFrame(data)
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                    df.columns = ["time", "Open", "High", "Low", "Close", "Volume"]
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    df = df.dropna(subset=["time"])
                    df.set_index("time", inplace=True)
                    return df
    except Exception:
        pass
    return None

# ---------------- Analysis routine (SAFE) ----------------
def analyze_df(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"structure": "No Data", "fib_bias": "Netral", "current_price": None,
                "change_pct": 0.0, "high": None, "low": None}

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        else:
            return {"structure": "No Data", "fib_bias": "Netral", "current_price": None,
                    "change_pct": 0.0, "high": None, "low": None}

    df = df.dropna(subset=["Close"], errors="ignore")
    if df.empty or len(df) < 3:
        return {"structure": "Data Kurang", "fib_bias": "Netral", "current_price": None,
                "change_pct": 0.0, "high": None, "low": None}

    for c in ["High", "Low"]:
        if c not in df.columns:
            df[c] = df["Close"]

    current = float(df["Close"].iloc[-1])
    high = float(df["High"].max())
    low = float(df["Low"].min())
    diff = high - low if high and low else 0.0

    fib_bias = "Netral"
    if diff > 0:
        fib61 = high - diff * 0.618
        fib38 = high - diff * 0.382
        if current > fib61:
            fib_bias = "Bullish"
        elif current < fib38:
            fib_bias = "Bearish"

    structure = "Konsolidasi"
    try:
        recent = df[["High", "Low"]].tail(14)
        if len(recent) >= 14:
            rh = recent["High"].rolling(5, center=True).max().dropna()
            rl = recent["Low"].rolling(5, center=True).min().dropna()
            if len(rh) >= 2 and len(rl) >= 2:
                if rh.iloc[-1] > rh.iloc[-2] and rl.iloc[-1] > rl.iloc[-2]:
                    structure = "Bullish"
                elif rh.iloc[-1] < rh.iloc[-2] and rl.iloc[-1] < rl.iloc[-2]:
                    structure = "Bearish"
    except Exception:
        pass

    change_pct = 0.0
    if len(df) >= 2:
        prev = float(df["Close"].iloc[-2])
        if prev != 0:
            change_pct = ((current - prev) / prev) * 100

    return {"structure": structure, "fib_bias": fib_bias, "current_price": current,
            "change_pct": change_pct, "high": high, "low": low}

# ---------------- Fetching spot data (yfinance, cached) ----------------
@st.cache_data(ttl=180, show_spinner=True)
def fetch_spot_batch(yf_symbols: List[str], interval: str, period_days: int = 90) -> Dict[str, pd.DataFrame]:
    out = {}
    for sym in yf_symbols:
        try:
            df = yf.download(sym, period=f"{period_days}d", interval=interval, progress=False)
            out[sym] = df if (df is not None and not df.empty) else None
        except Exception:
            out[sym] = None
    return out

# ---------------- Run scan ----------------
def run_scan(mode: str, coins_usdt: List[str], tf: str):
    results = []
    spot_data = {}
    if mode in ["Spot", "Scan Both"] and yf_ok:
        yf_syms = [to_yf(c) for c in coins_usdt]
        spot_data = fetch_spot_batch(yf_syms, YF_INTERVAL[tf], period_days=120)

    progress = st.progress(0, text="Memulai pemindaian...")
    total = len(coins_usdt)
    for i, coin in enumerate(coins_usdt):
        row = {"symbol": coin}
        if mode in ["Spot", "Scan Both"] and yf_ok:
            yf_sym = to_yf(coin)
            df_spot = spot_data.get(yf_sym)
            if df_spot is not None and tf == "4h":
                try:
                    df_spot.index = pd.to_datetime(df_spot.index)
                    if set(["Open", "High", "Low", "Close"]).issubset(df_spot.columns):
                        df_spot = df_spot.resample("4H").agg({
                            "Open": "first", "High": "max", "Low": "min",
                            "Close": "last", "Volume": "sum"
                        }).dropna()
                except Exception:
                    pass
            res_spot = analyze_df(df_spot)
            for k, v in res_spot.items():
                row[f"spot_{k}"] = v

        if mode in ["Futures", "Scan Both"]:
            df_fut = fetch_future_mirror(coin, BIN_INT[tf])
            res_fut = analyze_df(df_fut)
            for k, v in res_fut.items():
                row[f"fut_{k}"] = v

        results.append(row)
        progress.progress((i+1)/total, text=f"Memindai {coin} ({i+1}/{total})")
        time.sleep(0.02)
    return results

# ---------------- Execute scan ----------------
st.info(f"⏳ Memindai {TOTAL_COINS} koin ({data_mode}) — proses bisa memakan waktu beberapa menit.")
scan_results = run_scan(data_mode, COIN_LIST, selected_tf)
df = pd.DataFrame(scan_results)

# Pastikan kolom utama aman
for c in ["spot_structure","fut_structure","spot_fib_bias","fut_fib_bias","spot_change_pct","fut_change_pct"]:
    if c not in df.columns:
        df[c] = "No Data"

# Gabungan nilai unified
def unified(col):
    if f"spot_{col}" in df and df[f"spot_{col}"].notna().any():
        return df[f"spot_{col}"]
    if f"fut_{col}" in df:
        return df[f"fut_{col}"]
    return pd.Series(["No Data"] * len(df))

df["structure"] = unified("structure")
df["fib_bias"] = unified("fib_bias")
df["change_pct"] = pd.to_numeric(unified("change_pct"), errors="coerce").fillna(0.0)

# Conviction
def compute_conv(r):
    s, f = r.get("structure","No Data"), r.get("fib_bias","Netral")
    if s in ["Bullish","Bearish"] and s == f: return "Tinggi"
    if s in ["Bullish","Bearish"] or f in ["Bullish","Bearish"]: return "Sedang"
    return "Rendah"

df["conviction"] = df.apply(compute_conv, axis=1)

# ---------------- Top Movers ----------------
st.subheader("📈 Top 3 Bullish Movers")
bull = df.sort_values("change_pct", ascending=False).head(3)
for _, r in bull.iterrows():
    st.markdown(f"**{r['symbol']}** → {r['change_pct']:.2f}% | {r.get('structure','-')} | {r.get('conviction','-')}")

st.subheader("📉 Top 3 Bearish Movers")
bear = df.sort_values("change_pct", ascending=True).head(3)
for _, r in bear.iterrows():
    st.markdown(f"**{r['symbol']}** → {r['change_pct']:.2f}% | {r.get('structure','-')} | {r.get('conviction','-')}")

# ---------------- Chart ----------------
symbol = st.selectbox("Pilih simbol untuk grafik:", sorted(df["symbol"].unique()))
yf_sym = to_yf(symbol)

def add_indicators(d):
    if d is None or d.empty or "Close" not in d.columns: return d
    d["EMA20"] = d["Close"].ewm(span=20).mean()
    d["EMA50"] = d["Close"].ewm(span=50).mean()
    m = d["Close"].rolling(20).mean(); s = d["Close"].rolling(20).std()
    d["BB_up"] = m + 2*s; d["BB_dn"] = m - 2*s
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14).mean()
    rs = gain/loss; d["RSI"] = 100 - (100/(1+rs))
    return d

@st.cache_data(ttl=300)
def get_chart(sym, tf):
    try:
        d = yf.download(to_yf(sym), period="180d", interval=YF_INTERVAL[tf], progress=False)
        if d is not None and not d.empty: return d
    except: pass
    return fetch_future_mirror(sym, BIN_INT[tf])

chart = get_chart(symbol, selected_tf)
if chart is None or chart.empty:
    st.warning("⚠️ Data grafik tidak tersedia.")
else:
    chart = add_indicators(chart)
    fig = go.Figure()
    if chart_style == "Candlestick" and {"Open","High","Low","Close"}.issubset(chart.columns):
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
    fig.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig, use_container_width=True)

    if "RSI" in chart.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=chart.index, y=chart["RSI"], mode="lines", name="RSI14"))
        fig2.update_layout(template="plotly_dark", height=200, yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Spot via Yahoo • Futures via Binance/OKX Mirror • Indikator: EMA20/50, BB, RSI14")
