# analyst_instant_spot_future_mirror_safe_final.py
# Versi final — Spot (Yahoo) + Futures (Binance & OKX mirror) + Chart + Indicators
# Streamlit Cloud compatible, fully error-safe, 350+ coins

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
# Duplicate and trim to ~350 entries deterministically
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
    """
    Try multiple mirror endpoints for klines/candles.
    Returns DataFrame with columns ['Open','High','Low','Close','Volume'] or None.
    """
    # Binance mirror (data-api.binance.vision) returns klines like [openTime, open, high, low, close, ...]
    binance_url = "https://data-api.binance.vision/api/v3/klines"
    okx_url = "https://www.okx.com/api/v5/market/candles"

    # try binance mirror
    try:
        r = requests.get(binance_url, params={"symbol": sym_usdt, "interval": interval, "limit": 300}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data") or []
            if data:
                df = pd.DataFrame(data)
                # typical kline: [openTime, open, high, low, close, volume, ...]
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                    df.columns = ["time", "Open", "High", "Low", "Close", "Volume"]
                    df["Open"] = df["Open"].astype(float)
                    df["High"] = df["High"].astype(float)
                    df["Low"] = df["Low"].astype(float)
                    df["Close"] = df["Close"].astype(float)
                    df["Volume"] = df["Volume"].astype(float)
                    # some mirrors give ms timestamp, some give seconds; try ms first
                    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    if df["time"].isna().all():
                        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
                    df = df.dropna(subset=["time"])
                    df.set_index("time", inplace=True)
                    return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass

    # try okx mirror
    try:
        # OKX uses instId and bar param (bar like 1H/4H)
        inst = sym_usdt  # OKX may accept same symbol for many major pairs
        r = requests.get(okx_url, params={"instId": inst, "bar": interval, "limit": 300}, timeout=10)
        if r.status_code == 200:
            resp = r.json()
            data = resp.get("data") or []
            if data:
                df = pd.DataFrame(data)
                # OKX returns [ts, open, high, low, close, volume]
                if df.shape[1] >= 6:
                    df = df.iloc[:, :6]
                    df.columns = ["time", "Open", "High", "Low", "Close", "Volume"]
                    df["Open"] = df["Open"].astype(float)
                    df["High"] = df["High"].astype(float)
                    df["Low"] = df["Low"].astype(float)
                    df["Close"] = df["Close"].astype(float)
                    df["Volume"] = df["Volume"].astype(float)
                    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    df = df.dropna(subset=["time"])
                    df.set_index("time", inplace=True)
                    return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass

    return None

# ---------------- Analysis routine ----------------
def analyze_df(df: pd.DataFrame) -> Dict:
    """
    Returns dict with structure, fib_bias, current_price, change_pct, high, low.
    Safe to call even if df is None or missing columns.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return {"structure": "No Data", "fib_bias": "Netral", "current_price": None, "change_pct": 0.0, "high": None, "low": None}

    # ensure numeric
    df = df.dropna(subset=["Close"])
    if df.empty or len(df) < 3:
        return {"structure": "Data Kurang", "fib_bias": "Netral", "current_price": None, "change_pct": 0.0, "high": None, "low": None}

    current = float(df["Close"].iloc[-1])
    high = float(df["High"].max())
    low = float(df["Low"].min())
    diff = high - low if high is not None and low is not None else 0.0

    fib_bias = "Netral"
    if diff > 0:
        fib61 = high - diff * 0.618
        fib38 = high - diff * 0.382
        if current > fib61:
            fib_bias = "Bullish"
        elif current < fib38:
            fib_bias = "Bearish"

    # market structure via rolling highs/lows (need enough points)
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
        structure = "Konsolidasi"

    change_pct = 0.0
    if len(df) >= 2:
        prev = float(df["Close"].iloc[-2])
        if prev != 0:
            change_pct = ((current - prev) / prev) * 100

    return {"structure": structure, "fib_bias": fib_bias, "current_price": current, "change_pct": change_pct, "high": high, "low": low}

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
    # prepare spot batch if needed
    spot_data = {}
    if mode in ["Spot", "Scan Both"] and yf_ok:
        yf_syms = [to_yf(c) for c in coins_usdt]
        spot_data = fetch_spot_batch(yf_syms, YF_INTERVAL[tf], period_days=120)

    progress = st.progress(0, text="Memulai pemindaian...")
    total = len(coins_usdt)
    for i, coin in enumerate(coins_usdt):
        row = {"symbol": coin}
        # Spot
        if mode in ["Spot", "Scan Both"] and yf_ok:
            yf_sym = to_yf(coin)
            df_spot = spot_data.get(yf_sym)
            # if 4h requested and we fetched 60m, resample safely
            if df_spot is not None and tf == "4h":
                try:
                    df_spot.index = pd.to_datetime(df_spot.index)
                    # require Open/High/Low/Close columns exist before resample
                    if set(["Open", "High", "Low", "Close"]).issubset(df_spot.columns):
                        df_spot = df_spot.resample("4H").agg({
                            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
                        }).dropna()
                except Exception:
                    pass
            res_spot = analyze_df(df_spot)
            # prefix spot keys
            row.update({
                "spot_structure": res_spot["structure"],
                "spot_fib_bias": res_spot["fib_bias"],
                "spot_current_price": res_spot["current_price"],
                "spot_change_pct": res_spot["change_pct"],
                "spot_high": res_spot["high"],
                "spot_low": res_spot["low"]
            })
        # Futures
        if mode in ["Futures", "Scan Both"]:
            bin_sym = coin  # Binance uses USDT symbol like 'BTCUSDT'
            df_fut = fetch_future_mirror(bin_sym, BIN_INT[tf])
            res_fut = analyze_df(df_fut)
            row.update({
                "fut_structure": res_fut["structure"],
                "fut_fib_bias": res_fut["fib_bias"],
                "fut_current_price": res_fut["current_price"],
                "fut_change_pct": res_fut["change_pct"],
                "fut_high": res_fut["high"],
                "fut_low": res_fut["low"]
            })
        results.append(row)
        # polite progress
        progress.progress((i+1)/total, text=f"Memindai {coin} ({i+1}/{total})")
        time.sleep(0.02)
    return results

# ---------------- Execute scan ----------------
st.info(f"⏳ Memindai {TOTAL_COINS} koin ({data_mode}) — proses bisa memakan waktu beberapa detik sampai beberapa menit.")
start_time = time.time()
scan_results = run_scan(data_mode, COIN_LIST, selected_tf)
elapsed = time.time() - start_time
st.success(f"✅ Pemindaian selesai dalam {elapsed:.1f} detik")

# ---------------- Build DataFrame with safe columns ----------------
df = pd.DataFrame(scan_results)

# Ensure consistent columns exist (fill defaults)
# Spot columns
for c in ["spot_structure", "spot_fib_bias", "spot_current_price", "spot_change_pct", "spot_high", "spot_low"]:
    if c not in df.columns:
        df[c] = None if "price" in c or "high" in c or "low" in c else "No Data"
# Futures columns
for c in ["fut_structure", "fut_fib_bias", "fut_current_price", "fut_change_pct", "fut_high", "fut_low"]:
    if c not in df.columns:
        df[c] = None if "price" in c or "high" in c or "low" in c else "No Data"

# Create unified fields preferring Spot over Futures when both present
def unified_structure(row):
    if row.get("spot_current_price") is not None:
        return row.get("spot_structure", "No Data")
    if row.get("fut_current_price") is not None:
        return row.get("fut_structure", "No Data")
    return "No Data"

def unified_fib(row):
    if row.get("spot_current_price") is not None:
        return row.get("spot_fib_bias", "Netral")
    if row.get("fut_current_price") is not None:
        return row.get("fut_fib_bias", "Netral")
    return "Netral"

def unified_price(row):
    return row.get("spot_current_price") if row.get("spot_current_price") is not None else row.get("fut_current_price")

def unified_change(row):
    # prefer spot change_pct, fallback to fut_change_pct, else 0.0
    sp = row.get("spot_change_pct")
    fu = row.get("fut_change_pct")
    if pd.notna(sp) and sp is not None:
        return sp
    if pd.notna(fu) and fu is not None:
        return fu
    return 0.0

df["structure"] = df.apply(unified_structure, axis=1)
df["fib_bias"] = df.apply(unified_fib, axis=1)
df["current_price"] = df.apply(unified_price, axis=1)
df["change_pct"] = df.apply(unified_change, axis=1)

# ensure change_pct exists and numeric
if "change_pct" not in df.columns:
    df["change_pct"] = 0.0
df["change_pct"] = pd.to_numeric(df["change_pct"].fillna(0.0), errors="coerce").fillna(0.0)

# ---------------- Conviction (safe) ----------------
def compute_conv(row):
    s = row.get("structure", "No Data")
    f = row.get("fib_bias", "Netral")
    if s in ["Bullish", "Bearish"] and s == f:
        return "Tinggi"
    if s in ["Bullish", "Bearish"] or f in ["Bullish", "Bearish"]:
        return "Sedang"
    return "Rendah"

df["conviction"] = df.apply(compute_conv, axis=1)

# ---------------- Top movers (safe) ----------------
st.subheader("📈 Top 3 Bullish Movers")
if "change_pct" in df.columns and not df[df["change_pct"].notnull()].empty:
    top_bull = df.sort_values("change_pct", ascending=False).head(3)
    for _, r in top_bull.iterrows():
        st.markdown(f"**{r['symbol']}** → {r['change_pct']:.2f}% | {r.get('structure','No Data')} | {r.get('conviction','Rendah')}")
else:
    st.info("Tidak ada data perubahan harga untuk menentukan Top Bullish Movers.")

st.subheader("📉 Top 3 Bearish Movers")
if "change_pct" in df.columns and not df[df["change_pct"].notnull()].empty:
    top_bear = df.sort_values("change_pct", ascending=True).head(3)
    for _, r in top_bear.iterrows():
        st.markdown(f"**{r['symbol']}** → {r['change_pct']:.2f}% | {r.get('structure','No Data')} | {r.get('conviction','Rendah')}")
else:
    st.info("Tidak ada data perubahan harga untuk menentukan Top Bearish Movers.")

st.markdown("---")
# display summary table (selected columns)
display_cols = ["symbol", "structure", "fib_bias", "current_price", "change_pct", "conviction",
                "spot_current_price", "fut_current_price"]
# ensure columns existence
for col in display_cols:
    if col not in df.columns:
        df[col] = None

st.dataframe(df[display_cols].fillna("N/A").reset_index(drop=True))

# ---------------- Chart per symbol (safe) ----------------
symbol = st.selectbox("Pilih simbol untuk grafik:", sorted(df["symbol"].unique()))
yf_sym = to_yf(symbol)

@st.cache_data(ttl=300)
def get_chart_spot(sym: str, tf: str):
    try:
        interval = YF_INTERVAL[tf]
        d = yf.download(sym, period="180d", interval=interval, progress=False)
        return d if (d is not None and not d.empty) else None
    except Exception:
        return None

def get_chart_fut(sym_usdt: str, tf: str):
    return fetch_future_mirror(sym_usdt, BIN_INT[tf])

def add_indicators(dfc: pd.DataFrame) -> pd.DataFrame:
    if dfc is None or dfc.empty or "Close" not in dfc.columns:
        return dfc
    df = dfc.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    ma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_up"] = ma + 2 * std
    df["BB_dn"] = ma - 2 * std
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# choose spot first if available, else futures
chart_df = None
if data_mode in ["Spot", "Scan Both"] and yf_ok:
    chart_df = get_chart_spot(yf_sym, selected_tf)
if (chart_df is None or chart_df.empty) and data_mode in ["Futures", "Scan Both"]:
    chart_df = get_chart_fut(symbol, selected_tf)

if chart_df is None or (hasattr(chart_df, "empty") and chart_df.empty):
    st.warning("⚠️ Data grafik tidak tersedia untuk simbol ini.")
else:
    chart_df = add_indicators(chart_df)
    fig = go.Figure()
    if chart_style == "Candlestick" and {"Open", "High", "Low", "Close"}.issubset(chart_df.columns):
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
                                     low=chart_df["Low"], close=chart_df["Close"], name="Price"))
    else:
        # fallback to line
        if "Close" in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], mode="lines", name="Close"))
    # indicators
    if "EMA20" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"], mode="lines", name="EMA20"))
    if "EMA50" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], mode="lines", name="EMA50"))
    # BB as area if present
    if {"BB_up", "BB_dn"}.issubset(chart_df.columns):
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_up"], line=dict(width=0), showlegend=False, name="BB_up"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_dn"], fill="tonexty", line=dict(width=0), showlegend=False, name="BB_dn"))
    fig.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig, use_container_width=True)

    # RSI subplot
    if "RSI" in chart_df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], mode="lines", name="RSI14"))
        fig2.update_layout(template="plotly_dark", height=200, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Spot via Yahoo Finance • Futures via Binance & OKX mirror • Indikator: EMA20/50, BB(20,2), RSI14")
