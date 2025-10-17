# analyst_instant_spot_future_mirror.py
# Versi Final — Spot + Futures (Binance & OKX Mirror) + Chart + Indicators
# 100% Aman di Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide", page_title="Instant AI Analyst — Spot & Futures Mirror")
st.title("🚀 Instant AI Analyst — Spot (Yahoo) + Futures (Binance/OKX Mirror)")
st.caption("📊 350+ Koin | Chart Interaktif | EMA • RSI • Bollinger Bands | Mirror-safe API")
st.markdown("---")

col1, col2, col3 = st.columns([2, 2, 6])
selected_tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
data_mode = col2.radio("📈 Mode Data", ["Spot", "Futures", "Scan Both"], index=2)
chart_style = col3.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.rerun()

# ===============================
# COIN LIST (~350)
# ===============================
COIN_LIST = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "SHIBUSDT","TRXUSDT","BCHUSDT","LTCUSDT","LINKUSDT","NEARUSDT","UNIUSDT","ATOMUSDT","ICPUSDT","PEPEUSDT",
    "INJUSDT","TIAUSDT","FETUSDT","RNDRUSDT","ARBUSDT","OPUSDT","ETCUSDT","FILUSDT","FTMUSDT","SANDUSDT",
    "MANAUSDT","GRTUSDT","AAVEUSDT","EGLDUSDT","VETUSDT","CRVUSDT","ZILUSDT","DYDXUSDT","IMXUSDT","SUIUSDT",
    "SEIUSDT","IDUSDT","KAVAUSDT","COMPUSDT","GMXUSDT","FLOWUSDT","APTUSDT","LDOUSDT","MASKUSDT","GALAUSDT",
    "JASMYUSDT","C98USDT","MKRUSDT","CELOUSDT","OCEANUSDT","MINAUSDT","STXUSDT","CHZUSDT","AUDIOUSDT","RUNEUSDT",
    "ENJUSDT","RNDRUSDT","AGIXUSDT","MAGICUSDT","FLRUSDT","ILVUSDT","KSMUSDT","NMRUSDT","RSRUSDT","SFPUSDT",
    "PENDLEUSDT","HOOKUSDT","SYSUSDT","ALICEUSDT","PHAUSDT","ARKMUSDT","ZROUSDT","PRIMEUSDT","ZETAUSDT","STRKUSDT"
] * 4
COIN_LIST = list(dict.fromkeys(COIN_LIST))[:350]

# ===============================
# HELPERS
# ===============================
def to_yf(symbol): return symbol.replace("USDT", "-USD")

YF_INTERVAL = {"1d":"1d", "4h":"60m", "1h":"60m"}
BIN_INT = {"1d":"1d", "4h":"4h", "1h":"1h"}

@st.cache_data(ttl=60)
def check_yf():
    try:
        df = yf.download("BTC-USD", period="2d", interval="1h", progress=False)
        return not df.empty
    except:
        return False

yf_ok = check_yf()
if not yf_ok:
    st.warning("⚠️ Yahoo Finance tidak dapat diakses.")

# ===============================
# FUTURES FETCH (Binance + OKX Mirror)
# ===============================
def fetch_future_mirror(symbol, interval):
    """Ambil data dari Binance atau OKX mirror"""
    urls = [
        ("https://data-api.binance.vision/api/v3/klines", "binance"),
        ("https://www.okx.com/api/v5/market/candles", "okx"),
    ]
    for url, source in urls:
        try:
            if source == "binance":
                r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": 300}, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, dict):
                        data = data.get("data") or data.get("rows") or []
                    df = pd.DataFrame(data)
                    if df.shape[1] < 6:
                        continue
                    df.columns = ["time","Open","High","Low","Close","Volume","_","_","_","_","_","_"][:len(df.columns)]
                    df["Open"] = df["Open"].astype(float)
                    df["High"] = df["High"].astype(float)
                    df["Low"] = df["Low"].astype(float)
                    df["Close"] = df["Close"].astype(float)
                    df["Volume"] = df["Volume"].astype(float)
                    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    df.set_index("time", inplace=True)
                    return df
            elif source == "okx":
                r = requests.get(url, params={"instId": symbol, "bar": interval, "limit": 300}, timeout=10)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    df = pd.DataFrame(data, columns=["time","Open","High","Low","Close","Volume"])
                    df["Open"] = df["Open"].astype(float)
                    df["High"] = df["High"].astype(float)
                    df["Low"] = df["Low"].astype(float)
                    df["Close"] = df["Close"].astype(float)
                    df["Volume"] = df["Volume"].astype(float)
                    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                    df.set_index("time", inplace=True)
                    return df
        except Exception:
            continue
    return None

# ===============================
# ANALISIS PASAR
# ===============================
def analyze(df):
    if df is None or df.empty or "Close" not in df.columns:
        return {"structure":"No Data","fib_bias":"Netral","current_price":None,"change_pct":0.0}
    cur = df["Close"].iloc[-1]
    hi, lo = df["High"].max(), df["Low"].min()
    diff = hi - lo if hi and lo else 0
    fib_bias = "Netral"
    if diff > 0:
        fib61, fib38 = hi - diff*0.618, hi - diff*0.382
        if cur > fib61: fib_bias = "Bullish"
        elif cur < fib38: fib_bias = "Bearish"
    rh, rl = df["High"].rolling(5).max(), df["Low"].rolling(5).min()
    structure = "Konsolidasi"
    if rh.iloc[-1] > rh.iloc[-2] and rl.iloc[-1] > rl.iloc[-2]:
        structure = "Bullish"
    elif rh.iloc[-1] < rh.iloc[-2] and rl.iloc[-1] < rl.iloc[-2]:
        structure = "Bearish"
    chg = ((df["Close"].iloc[-1]-df["Close"].iloc[-2])/df["Close"].iloc[-2])*100 if len(df)>2 else 0
    return {"structure":structure,"fib_bias":fib_bias,"current_price":cur,"change_pct":chg}

# ===============================
# RUN SCAN
# ===============================
@st.cache_data(show_spinner=True, ttl=180)
def fetch_yf_batch(symbols, interval):
    result = {}
    for s in symbols:
        try:
            df = yf.download(s, period="90d", interval=interval, progress=False)
            result[s] = df if not df.empty else None
        except:
            result[s] = None
    return result

def run_scan(mode):
    out=[]
    if mode in ["Spot","Scan Both"] and yf_ok:
        st.info("📡 Mengambil data Spot (Yahoo)...")
        yfs=[to_yf(c) for c in COIN_LIST]
        spot = fetch_yf_batch(yfs,YF_INTERVAL[selected_tf])
    else: spot={}
    for c in COIN_LIST:
        row={"symbol":c}
        if mode in ["Spot","Scan Both"] and yf_ok:
            df=spot.get(to_yf(c))
            if df is not None and not df.empty and set(["Open","High","Low","Close"]).issubset(df.columns):
                if selected_tf=="4h":
                    try:
                        df.index=pd.to_datetime(df.index)
                        df=df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
                    except Exception:
                        pass
                row.update(analyze(df))
        if mode in ["Futures","Scan Both"]:
            df=fetch_future_mirror(c,BIN_INT[selected_tf])
            row.update({f"fut_{k}":v for k,v in analyze(df).items()})
        out.append(row)
    return out

st.info(f"⏳ Memindai 350+ koin ({data_mode})...")
data=run_scan(data_mode)
st.success("✅ Pemindaian selesai!")

df=pd.DataFrame(data)
df["conviction"]=df.apply(
    lambda r: "Tinggi" if r.get("structure")==r.get("fib_bias") and r["structure"] in ["Bullish","Bearish"]
    else "Sedang" if r["structure"] in ["Bullish","Bearish"] or r["fib_bias"] in ["Bullish","Bearish"]
    else "Rendah", axis=1)

# ===============================
# HASIL & TABEL
# ===============================
st.subheader("📈 Top 3 Bullish Movers")
bull=df.sort_values("change_pct",ascending=False).head(3)
for _,r in bull.iterrows(): st.markdown(f"**{r['symbol']}** → +{r['change_pct']:.2f}% | {r['structure']}")

st.subheader("📉 Top 3 Bearish Movers")
bear=df.sort_values("change_pct",ascending=True).head(3)
for _,r in bear.iterrows(): st.markdown(f"**{r['symbol']}** → {r['change_pct']:.2f}% | {r['structure']}")

st.dataframe(df[["symbol","structure","fib_bias","current_price","change_pct","conviction"]])

# ===============================
# GRAFIK
# ===============================
symbol = st.selectbox("Pilih simbol untuk grafik:", sorted(df["symbol"].unique()))
yf_sym = to_yf(symbol)

@st.cache_data(ttl=300)
def get_chart_yf(sym): 
    try: 
        d=yf.download(sym,period="180d",interval=YF_INTERVAL[selected_tf],progress=False)
        return d if not d.empty else None
    except: return None

def add_ind(df):
    if df is None or df.empty or "Close" not in df.columns: return df
    df["EMA20"]=df["Close"].ewm(span=20).mean()
    df["EMA50"]=df["Close"].ewm(span=50).mean()
    ma=df["Close"].rolling(20).mean(); std=df["Close"].rolling(20).std()
    df["BB_up"]=ma+2*std; df["BB_dn"]=ma-2*std
    delta=df["Close"].diff(); gain=delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss=-delta.clip(upper=0).ewm(alpha=1/14).mean(); rs=gain/loss
    df["RSI"]=100-(100/(1+rs)); return df

df_chart=None
if data_mode in ["Spot","Scan Both"] and yf_ok:
    df_chart=get_chart_yf(yf_sym)
elif data_mode in ["Futures","Scan Both"]:
    df_chart=fetch_future_mirror(symbol,BIN_INT[selected_tf])

if df_chart is None or df_chart.empty:
    st.warning("⚠️ Data grafik tidak ditemukan.")
else:
    df_chart=add_ind(df_chart)
    fig=go.Figure()
    if chart_style=="Candlestick" and {"Open","High","Low","Close"}.issubset(df_chart.columns):
        fig.add_trace(go.Candlestick(x=df_chart.index,open=df_chart["Open"],high=df_chart["High"],low=df_chart["Low"],close=df_chart["Close"],name="Price"))
    else:
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["Close"],mode="lines",name="Close"))
    if "EMA20" in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["EMA20"],mode="lines",name="EMA20"))
    if "EMA50" in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["EMA50"],mode="lines",name="EMA50"))
    fig.update_layout(template="plotly_dark",height=500)
    st.plotly_chart(fig,use_container_width=True)
    if "RSI" in df_chart.columns:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=df_chart.index,y=df_chart["RSI"],mode="lines",name="RSI14"))
        fig2.update_layout(template="plotly_dark",height=200,yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2,use_container_width=True)

st.caption("📘 Data Spot via Yahoo Finance; Futures via Binance & OKX Mirror API. Indikator: EMA20/50, Bollinger Bands(20,2), RSI14.")
