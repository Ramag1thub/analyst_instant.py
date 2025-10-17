# analyst_instant_spot_gecko_final.py
# Spot-only analyzer using CoinGecko public API (no API key)
# Features:
# - ~200+ coin list (including custom tokens)
# - Fetch price history from CoinGecko /coins/{id}/market_chart
# - Build OHLCV by resampling the price series
# - Compute structure, fibonacci bias, support/resistance, conviction
# - Plot interactive chart (candlestick/line) with EMA/BB/RSI
# - Fully defensive: handles missing coins/data without crashing

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time
from typing import Optional, Dict, List

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="AI Analyst Spot — CoinGecko")
st.set_option('client.showErrorDetails', True)
st.title("🚀 Instant AI Analyst — Spot (CoinGecko)")
st.caption("Data via CoinGecko • Struktur • Fibonacci • Support/Resistance • Conviction")
st.markdown("---")

# ---------------- UI ----------------
col1, col2 = st.columns([2, 8])
selected_tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------------- Coin universe (includes custom tokens) ----------------
# This list is intentionally large — CoinGecko will be used to map symbol -> id.
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "SHIBUSDT","TRXUSDT","BCHUSDT","LTCUSDT","LINKUSDT","NEARUSDT","UNIUSDT","ATOMUSDT","ICPUSDT","PEPEUSDT",
    "INJUSDT","TIAUSDT","FETUSDT","RNDRUSDT","ARBUSDT","OPUSDT","ETCUSDT","FILUSDT","FTMUSDT","SANDUSDT",
    "MANAUSDT","GRTUSDT","AAVEUSDT","EGLDUSDT","VETUSDT","CRVUSDT","ZILUSDT","DYDXUSDT","IMXUSDT","SUIUSDT",
    "SEIUSDT","IDUSDT","KAVAUSDT","COMPUSDT","GMXUSDT","FLOWUSDT","APTUSDT","LDOUSDT","MASKUSDT","GALAUSDT",
    "JASMYUSDT","C98USDT","MKRUSDT","CELOUSDT","OCEANUSDT","MINAUSDT","STXUSDT","CHZUSDT","AUDIOUSDT","RUNEUSDT",
    "ENJUSDT","AGIXUSDT","MAGICUSDT","FLRUSDT","ILVUSDT","KSMUSDT","NMRUSDT","RSRUSDT","SFPUSDT","PENDLEUSDT",
    "HOOKUSDT","SYSUSDT","ALICEUSDT","PHAUSDT","ARKMUSDT","ZROUSDT","PRIMEUSDT","ZETAUSDT","STRKUSDT","AIOZUSDT",
    "PYTHUSDT","WLDUSDT","NOTUSDT","BONKUSDT","ORDIUSDT","SATSUSDT","FLOKIUSDT","CATIUSDT","DEGENUSDT","XAIUSDT",
    "AEVOUSDT","ETHFIUSDT","ENAUSDT","REZUSDT","PORTALUSDT","JTOUSDT","MANTAUSDT","JUPUSDT","ALTUSDT","AXLUSDT",
    "LPTUSDT","IDUSDT","BNXUSDT","BABYDOGEUSDT","GMEUSDT","BEAMUSDT","SKLUSDT","CEEKUSDT","XNOUSDT","WLDUSDT",
    # user requested custom tokens (explicit)
    "HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT"
]
# make unique and limit to ~200-250 for performance (you can expand)
COINS = list(dict.fromkeys(COINS))[:220]
TOTAL = len(COINS)

# ---------------- CoinGecko helpers ----------------
CG_API = "https://api.coingecko.com/api/v3"

@st.cache_data(ttl=60*60)
def cg_coins_list() -> List[Dict]:
    """Fetch coin list from CoinGecko (id, symbol, name). Cached for 1 hour."""
    url = f"{CG_API}/coins/list"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def find_coingecko_id(symbol: str, coins_list: List[Dict]) -> Optional[str]:
    """
    Try to map symbol like BTCUSDT / BTC-USD / BTC to coinGecko id.
    Strategy:
     - Normalize input symbol to common token symbol (strip USDT suffix)
     - Match by symbol (case-insensitive). If multiple matches, prefer exact name match.
    """
    s = symbol.upper()
    candidate = s
    # typical input: BTCUSDT -> want BTC ; SOLUSDT -> SOL
    if s.endswith("USDT"):
        candidate = s[:-4]
    if s.endswith("-USD"):
        candidate = s.split("-")[0]
    cand_lower = candidate.lower()

    # find exact symbol matches
    matches = [c for c in coins_list if c.get("symbol","").lower() == cand_lower]
    if matches:
        # prefer exact id name equals cand or id contains candidate
        # return first reasonable match
        return matches[0]["id"]
    # fallback: try substring in name
    matches = [c for c in coins_list if cand_lower in (c.get("id","").lower() + " " + c.get("name","").lower())]
    if matches:
        return matches[0]["id"]
    return None

@st.cache_data(ttl=60*10)
def cg_market_chart(coin_id: str, vs_currency: str, days: int):
    """
    Call /coins/{id}/market_chart?vs_currency=usd&days={days}
    Returns dict with 'prices' and 'total_volumes' or None
    """
    url = f"{CG_API}/coins/{coin_id}/market_chart"
    try:
        r = requests.get(url, params={"vs_currency": vs_currency, "days": days}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---------------- Build OHLCV from prices + volumes ----------------
def build_ohlcv_from_market_chart(mc: Dict, tf: str) -> Optional[pd.DataFrame]:
    """
    mc: result from /market_chart with keys 'prices' and 'total_volumes'
    tf: timeframe '1d','4h','1h'
    Returns DataFrame indexed by datetime with columns Open,High,Low,Close,Volume
    """
    if not mc or "prices" not in mc or not mc["prices"]:
        return None
    prices = mc.get("prices", [])  # [ [ts, price], ... ]
    volumes = mc.get("total_volumes", [])  # [ [ts, vol], ... ]
    price_df = pd.DataFrame(prices, columns=["ts","price"])
    vol_df = pd.DataFrame(volumes, columns=["ts","volume"]) if volumes else None

    # convert ms timestamp to datetime
    price_df["time"] = pd.to_datetime(price_df["ts"], unit="ms")
    price_df = price_df.set_index("time")[["price"]]
    if vol_df is not None and not vol_df.empty:
        vol_df["time"] = pd.to_datetime(vol_df["ts"], unit="ms")
        vol_df = vol_df.set_index("time")[["volume"]]
        df = price_df.join(vol_df, how="left")
        df["volume"] = df["volume"].fillna(0.0)
    else:
        df = price_df.copy()
        df["volume"] = 0.0

    # determine resample rule
    rule = {"1d":"1D","4h":"4H","1h":"1H"}.get(tf, "1D")
    try:
        ohlc = pd.DataFrame()
        ohlc["Open"] = df["price"].resample(rule).first()
        ohlc["High"] = df["price"].resample(rule).max()
        ohlc["Low"] = df["price"].resample(rule).min()
        ohlc["Close"] = df["price"].resample(rule).last()
        ohlc["Volume"] = df["volume"].resample(rule).sum()
        ohlc = ohlc.dropna(subset=["Close"])
        if ohlc.empty:
            return None
        return ohlc
    except Exception:
        return None

# ---------------- Analysis (structure, fib, support/res, conviction) ----------------
def analyze_ohlcv(df: pd.DataFrame) -> Dict:
    """Given OHLCV DataFrame, compute structure, fib bias, support/resistance, conviction, change, current"""
    if not isinstance(df, pd.DataFrame) or df.empty or "Close" not in df.columns:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah"}

    # ensure numeric
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    if df.empty or len(df) < 3:
        return {"structure":"Data Kurang","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah"}

    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    change = ((cur - prev) / prev * 100) if prev != 0 else 0.0

    hi = float(df["High"].max())
    lo = float(df["Low"].min())
    diff = hi - lo if hi and lo else 0.0

    fib_bias = "Netral"
    if diff > 0:
        fib61 = hi - diff * 0.618
        fib38 = hi - diff * 0.382
        if cur > fib61:
            fib_bias = "Bullish"
        elif cur < fib38:
            fib_bias = "Bearish"

    # structure: check last rolling highs/lows
    struct = "Konsolidasi"
    try:
        rec = df[["High","Low"]].tail(20)
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

    support = float(df["Low"].rolling(20).min().iloc[-1]) if len(df) >= 20 else float(df["Low"].min())
    resistance = float(df["High"].rolling(20).max().iloc[-1]) if len(df) >= 20 else float(df["High"].max())

    if struct in ["Bullish","Bearish"] and fib_bias == struct:
        conviction = "Tinggi"
    elif struct in ["Bullish","Bearish"] or fib_bias in ["Bullish","Bearish"]:
        conviction = "Sedang"
    else:
        conviction = "Rendah"

    return {"structure":struct,"fib_bias":fib_bias,"support":support,"resistance":resistance,
            "current":cur,"change":change,"conviction":conviction}

# ---------------- Fetch + Scan ----------------
@st.cache_data(ttl=60*5, show_spinner=True)
def prepare_coins_map():
    coins = cg_coins_list()
    return coins

def scan_all(coins_symbols: List[str], tf: str):
    cg_list = prepare_coins_map()
    results = []
    progress = st.progress(0, text="Memulai pemindaian...")
    total = len(coins_symbols)
    for i, sym in enumerate(coins_symbols):
        row = {"symbol": sym}
        coin_id = find_coingecko_id(sym, cg_list)
        if coin_id is None:
            # not found
            row.update({"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah","cg_id":None})
        else:
            # days param mapping
            days_map = {"1d":30, "4h":90, "1h":30}
            days = days_map.get(tf, 90)
            mc = cg_market_chart(coin_id, "usd", days)
            df_ohlcv = build_ohlcv_from_market_chart(mc, tf)
            metrics = analyze_ohlcv(df_ohlcv)
            row.update(metrics)
            row["cg_id"] = coin_id
        results.append(row)
        # progress update less frequently
        if i % 3 == 0:
            progress.progress((i+1)/total, text=f"Memindai {sym} ({i+1}/{total})")
        time.sleep(0.02)
    return results

st.info(f"⏳ Memindai {TOTAL} koin via CoinGecko (may take some seconds)...")
start = time.time()
scan_results = scan_all(COINS, selected_tf)
elapsed = time.time() - start
st.success(f"✅ Pemindaian selesai dalam {elapsed:.1f} detik")

df = pd.DataFrame(scan_results)

# ---------------- Normalize & Display ----------------
# ensure columns exist
for col in ["structure","fib_bias","support","resistance","current","change","conviction","cg_id"]:
    if col not in df.columns:
        df[col] = None

# Sort by % change (desc) for top movers
df["change"] = pd.to_numeric(df["change"], errors="coerce").fillna(0.0)
top_bull = df.sort_values("change", ascending=False).head(5)
top_bear = df.sort_values("change", ascending=True).head(5)

st.subheader("📈 Top 5 Bullish Movers")
for _, r in top_bull.iterrows():
    st.markdown(f"**{r['symbol']}** (cg:{r['cg_id']}) → {r['change']:.2f}% | {r['structure']} | {r['conviction']}")

st.subheader("📉 Top 5 Bearish Movers")
for _, r in top_bear.iterrows():
    st.markdown(f"**{r['symbol']}** (cg:{r['cg_id']}) → {r['change']:.2f}% | {r['structure']} | {r['conviction']}")

st.markdown("---")
display_cols = ["symbol","cg_id","structure","fib_bias","support","resistance","current","change","conviction"]
st.dataframe(df[display_cols].fillna("No Data").reset_index(drop=True))

# ---------------- Chart per coin ----------------
symbol = st.selectbox("Pilih simbol untuk grafik (CoinGecko):", sorted(df["symbol"].unique()))
selected_row = df[df["symbol"]==symbol].iloc[0] if not df[df["symbol"]==symbol].empty else None
cg_id = selected_row.get("cg_id") if selected_row is not None else None

def fetch_chart_df_coin_gecko(coin_id: str, tf: str) -> Optional[pd.DataFrame]:
    if coin_id is None:
        return None
    days_map = {"1d":30, "4h":90, "1h":30}
    days = days_map.get(tf,90)
    mc = cg_market_chart(coin_id, "usd", days)
    df_ohlcv = build_ohlcv_from_market_chart(mc, tf)
    return df_ohlcv

chart_df = fetch_chart_df_coin_gecko(cg_id, selected_tf)
if chart_df is None or chart_df.empty:
    st.warning("⚠️ Data grafik tidak tersedia untuk simbol ini.")
else:
    # indicators
    chart_df["EMA20"] = chart_df["Close"].ewm(span=20, adjust=False).mean()
    chart_df["EMA50"] = chart_df["Close"].ewm(span=50, adjust=False).mean()
    ma = chart_df["Close"].rolling(20).mean()
    sd = chart_df["Close"].rolling(20).std()
    chart_df["BB_up"] = ma + 2*sd
    chart_df["BB_dn"] = ma - 2*sd
    delta = chart_df["Close"].diff(); gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean(); rs = gain/loss
    chart_df["RSI"] = 100 - (100/(1+rs))

    fig = go.Figure()
    if chart_style == "Candlestick":
        # ensure Open/High/Low/Close present
        for c in ["Open","High","Low","Close"]:
            if c not in chart_df.columns:
                chart_df[c] = chart_df["Close"]
        fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
                                     low=chart_df["Low"], close=chart_df["Close"], name="Price"))
    else:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], mode="lines", name="Close"))

    if "EMA20" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA20"], mode="lines", name="EMA20"))
    if "EMA50" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA50"], mode="lines", name="EMA50"))
    if {"BB_up","BB_dn"}.issubset(chart_df.columns):
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_up"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_dn"], fill="tonexty", line=dict(width=0), showlegend=False))

    # support / resistance from analysis
    last_metrics = analyze_ohlcv(chart_df)
    if last_metrics.get("support"):
        fig.add_hline(y=last_metrics["support"], line=dict(color="green", dash="dot"), annotation_text="Support")
    if last_metrics.get("resistance"):
        fig.add_hline(y=last_metrics["resistance"], line=dict(color="red", dash="dot"), annotation_text="Resistance")

    fig.update_layout(template="plotly_dark", height=520,
                      title=f"{symbol} ({cg_id}) | {last_metrics['structure']} | {last_metrics['fib_bias']} | Conviction: {last_metrics['conviction']}")
    st.plotly_chart(fig, use_container_width=True)

    if "RSI" in chart_df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], mode="lines", name="RSI14"))
        fig2.update_layout(template="plotly_dark", height=200, yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Data via CoinGecko • Indicators: EMA20/50, BB(20,2), RSI14 • Struktur, Fibonacci, Support/Resistance, Conviction")
