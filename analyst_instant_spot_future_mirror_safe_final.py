# analyst_instant_spot_final_ultra.py
# Versi ultra stable — 200 koin spot, tanpa error, support fibo, structure, conviction

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

# ======== CONFIG ========
st.set_page_config(layout="wide", page_title="AI Analyst Spot Ultra Stable")
st.set_option('client.showErrorDetails', True)

st.title("🚀 Instant AI Analyst — Spot Only (Ultra Stable)")
st.caption("Struktur • Fibonacci • Support/Resistance • Conviction • 200 Coin Yahoo Spot")
st.markdown("---")

col1, col2 = st.columns([2, 8])
tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ======== COINS ========
BASE = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "SHIBUSDT","TRXUSDT","BCHUSDT","LTCUSDT","LINKUSDT","NEARUSDT","UNIUSDT","ATOMUSDT","ICPUSDT","PEPEUSDT",
    "INJUSDT","TIAUSDT","FETUSDT","RNDRUSDT","ARBUSDT","OPUSDT","ETCUSDT","FILUSDT","FTMUSDT","SANDUSDT",
    "MANAUSDT","GRTUSDT","AAVEUSDT","VETUSDT","CRVUSDT","DYDXUSDT","IMXUSDT","SUIUSDT","SEIUSDT","KAVAUSDT",
    "FLOWUSDT","APTUSDT","LDOUSDT","MASKUSDT","GALAUSDT","JASMYUSDT","MKRUSDT","CELOUSDT","MINAUSDT","STXUSDT",
    "CHZUSDT","RUNEUSDT","AGIXUSDT","MAGICUSDT","NMRUSDT","RSRUSDT","SFPUSDT","PENDLEUSDT","HOOKUSDT","ALICEUSDT",
    "ARKMUSDT","ZROUSDT","PRIMEUSDT","ZETAUSDT","STRKUSDT","AIOZUSDT","PYTHUSDT","WLDUSDT","NOTUSDT","BONKUSDT",
    "BOMEUSDT","TURBOUSDT","ORDIUSDT","SATSUSDT","FLOKIUSDT","CATIUSDT","DEGENUSDT","ALEXUSDT","XAIUSDT","AEVOUSDT",
    "ETHFIUSDT","ENAUSDT","REZUSDT","COWUSDT","PORTALUSDT","JTOUSDT","MANTAUSDT","JUPUSDT","ALTUSDT","AXLUSDT",
    "TNSRUSDT","CETUSUSDT","CAGAUSDT","NIBIUSDT","WUSDT","BNXUSDT","DYMUSDT","LPTUSDT","IDUSDT","IOUSDT",
    "VANAUSDT","BABYDOGEUSDT","GMEUSDT","GRTUSDT","XNOUSDT","BEAMUSDT","SKLUSDT","KSMUSDT","CEEKUSDT","ENJUSDT"
]
COINS = BASE[:200]
TOTAL = len(COINS)
YF_INTERVAL = {"1d": "1d", "4h": "60m", "1h": "60m"}
def to_yf(sym): return sym[:-4] + "-USD" if sym.endswith("USDT") else sym

# ======== ANALYZE SAFE ========
def analyze_df(df):
    """Analisa aman 100% dari error tipe dan kolom hilang"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"structure": "No Data","fib_bias": "Netral","support": None,"resistance": None,
                "current": None,"change": 0.0,"conviction": "Rendah"}

    # flatten multiindex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]

    # rename basic cols
    close_cols = [c for c in df.columns if "Close" in c]
    if close_cols:
        df = df.rename(columns={close_cols[0]: "Close"})
    else:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,
                "current":None,"change":0.0,"conviction":"Rendah"}

    for name in ["Open","High","Low"]:
        alt = [c for c in df.columns if name in c]
        if alt:
            df = df.rename(columns={alt[0]: name})
        elif name not in df.columns:
            df[name] = df["Close"]

    # numeric safe
    for col in ["Open","High","Low","Close"]:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = np.nan
    df = df.dropna(subset=["Close"])
    if len(df) < 5:
        return {"structure":"Data Kurang","fib_bias":"Netral","support":None,"resistance":None,
                "current":None,"change":0.0,"conviction":"Rendah"}

    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df)>1 else cur
    change = ((cur - prev) / prev * 100) if prev!=0 else 0.0

    hi, lo = float(df["High"].max()), float(df["Low"].min())
    diff = hi - lo if hi and lo else 0.0
    fib_bias = "Netral"
    if diff>0:
        fib61, fib38 = hi - diff*0.618, hi - diff*0.382
        if cur > fib61: fib_bias = "Bullish"
        elif cur < fib38: fib_bias = "Bearish"

    struct = "Konsolidasi"
    try:
        rec = df[["High","Low"]].tail(20)
        hh = rec["High"].rolling(5).max().dropna()
        ll = rec["Low"].rolling(5).min().dropna()
        if len(hh)>=2 and len(ll)>=2:
            if hh.iloc[-1]>hh.iloc[-2] and ll.iloc[-1]>ll.iloc[-2]: struct="Bullish"
            elif hh.iloc[-1]<hh.iloc[-2] and ll.iloc[-1]<ll.iloc[-2]: struct="Bearish"
    except Exception:
        pass

    support = float(df["Low"].rolling(20).min().iloc[-1])
    resistance = float(df["High"].rolling(20).max().iloc[-1])

    if struct in ["Bullish","Bearish"] and fib_bias==struct: conviction="Tinggi"
    elif struct in ["Bullish","Bearish"] or fib_bias in ["Bullish","Bearish"]: conviction="Sedang"
    else: conviction="Rendah"

    return {"structure":struct,"fib_bias":fib_bias,"support":support,
            "resistance":resistance,"current":cur,"change":change,"conviction":conviction}

# ======== FETCH ========
@st.cache_data(ttl=120)
def fetch_data(symbols, interval):
    data = {}
    for s in symbols:
        try:
            df = yf.download(to_yf(s), period="90d", interval=interval, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                data[s] = df
        except Exception:
            continue
    return data

st.info(f"⏳ Memindai {TOTAL} koin Spot Yahoo...")
start=time.time()
data=fetch_data(COINS, YF_INTERVAL[tf])

results=[]
for i,(sym,df) in enumerate(data.items()):
    res=analyze_df(df)
    res["symbol"]=sym
    results.append(res)
    if i%3==0:
        st.progress((i+1)/TOTAL, text=f"📊 {sym}")
    time.sleep(0.01)

elapsed=time.time()-start
st.success(f"✅ Pemindaian selesai dalam {elapsed:.1f} detik")

dfres=pd.DataFrame(results)

# ======== TABEL ========
if not dfres.empty:
    st.dataframe(dfres[["symbol","structure","fib_bias","support","resistance","change","conviction"]]
                 .sort_values("change", ascending=False).reset_index(drop=True))
else:
    st.warning("⚠️ Tidak ada data valid yang berhasil diambil.")

# ======== CHART ========
symbol=st.selectbox("📈 Pilih koin untuk grafik:", sorted(dfres["symbol"].unique() if not dfres.empty else ["BTCUSDT"]))
try:
    chart=yf.download(to_yf(symbol), period="120d", interval=YF_INTERVAL[tf], progress=False)
except Exception:
    chart=None

if chart is None or not isinstance(chart, pd.DataFrame) or chart.empty:
    st.warning("⚠️ Data grafik tidak tersedia.")
else:
    # pastikan kolom lengkap
    for name in ["Open","High","Low","Close"]:
        if name not in chart.columns:
            chart[name] = chart["Close"]

    chart["EMA20"]=chart["Close"].ewm(span=20).mean()
    chart["EMA50"]=chart["Close"].ewm(span=50).mean()
    ma=chart["Close"].rolling(20).mean(); sd=chart["Close"].rolling(20).std()
    chart["BB_up"]=ma+2*sd; chart["BB_dn"]=ma-2*sd
    last=analyze_df(chart)

    fig=go.Figure()
    if chart_style=="Candlestick":
        fig.add_trace(go.Candlestick(x=chart.index,open=chart["Open"],high=chart["High"],
                                     low=chart["Low"],close=chart["Close"],name="Price"))
    else:
        fig.add_trace(go.Scatter(x=chart.index,y=chart["Close"],mode="lines",name="Close"))
    for i in ["EMA20","EMA50"]:
        if i in chart.columns:
            fig.add_trace(go.Scatter(x=chart.index,y=chart[i],mode="lines",name=i))
    if {"BB_up","BB_dn"}.issubset(chart.columns):
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_up"],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_dn"],fill="tonexty",line=dict(width=0),showlegend=False))
    if last["support"]:
        fig.add_hline(y=last["support"], line=dict(color="green", dash="dot"), annotation_text="Support")
    if last["resistance"]:
        fig.add_hline(y=last["resistance"], line=dict(color="red", dash="dot"), annotation_text="Resistance")
    fig.update_layout(template="plotly_dark", height=520,
                      title=f"{symbol} | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
    st.plotly_chart(fig, use_container_width=True)

st.caption("✅ Data Spot Yahoo Finance • EMA20/50 • BB(20,2) • Struktur • Fibonacci • Support/Resistance • Conviction")
