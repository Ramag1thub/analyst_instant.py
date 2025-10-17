import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time
import difflib

st.set_page_config(layout="wide", page_title="AI Analyst Dual Source (Binance + CoinGecko)")
st.title("🚀 Instant AI Analyst — Dual API Mode (Binance + CoinGecko)")
st.caption("Akurat untuk Top Coins • Fallback CoinGecko untuk Token Langka • Struktur • Fibonacci • EMA/RSI/BB • Support/Resistance")
st.markdown("---")

col1, col2 = st.columns([2, 8])
tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan ulang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ============ CONFIG ============
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","DOTUSDT","MATICUSDT","PEPEUSDT","ARBUSDT","OPUSDT","APTUSDT",
    "SUIUSDT","TIAUSDT","INJUSDT","WLDUSDT","RNDRUSDT","ATOMUSDT","UNIUSDT",
    "FTMUSDT","LDOUSDT","FLOWUSDT","AAVEUSDT","GALAUSDT","MASKUSDT",
    "BONKUSDT","BOMEUSDT","ORDIUSDT","HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT"
]

BINANCE_API = "https://api.binance.com/api/v3/klines"
COINGECKO_API = "https://api.coingecko.com/api/v3"

# ============ HELPERS ============
@st.cache_data(ttl=3600)
def cg_coin_list():
    try:
        r = requests.get(f"{COINGECKO_API}/coins/list", timeout=20)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame(columns=["id","symbol","name"])

def fuzzy_find(symbol, df):
    s = symbol.lower().replace("usdt","")
    matches = df[df["symbol"] == s]
    if not matches.empty:
        return matches.iloc[0]["id"]
    close = difflib.get_close_matches(s, df["name"].tolist(), n=1, cutoff=0.6)
    if close:
        r = df[df["name"] == close[0]]
        if not r.empty:
            return r.iloc[0]["id"]
    return None

def fetch_binance(symbol, tf):
    tf_map = {"1d":"1d","4h":"4h","1h":"1h"}
    try:
        r = requests.get(BINANCE_API, params={"symbol":symbol,"interval":tf_map[tf],"limit":500}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        df = pd.DataFrame(data, columns=["time","Open","High","Low","Close","Volume","c1","c2","c3","c4","c5","c6"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        return df
    except Exception:
        return None

def fetch_coingecko(coin_id, tf):
    days = {"1d":30,"4h":90,"1h":30}[tf]
    try:
        r = requests.get(f"{COINGECKO_API}/coins/{coin_id}/market_chart",
                         params={"vs_currency":"usd","days":days}, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        prices = js.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["time","Close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df["Open"] = df["Close"].shift(1)
        df["High"] = df["Close"].rolling(3).max()
        df["Low"] = df["Close"].rolling(3).min()
        df["Volume"] = 0
        return df.dropna()
    except Exception:
        return None

def analyze_df(df):
    if df is None or df.empty:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,
                "current":None,"change":0.0,"conviction":"Rendah"}

    cur = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df)>1 else cur
    change = ((cur-prev)/prev*100) if prev!=0 else 0.0
    hi, lo = df["High"].max(), df["Low"].min()
    diff = hi-lo if hi and lo else 0
    fib_bias="Netral"
    if diff>0:
        fib61, fib38 = hi-diff*0.618, hi-diff*0.382
        if cur>fib61: fib_bias="Bullish"
        elif cur<fib38: fib_bias="Bearish"

    struct="Konsolidasi"
    if len(df)>10:
        hh=df["High"].rolling(5).max().dropna()
        ll=df["Low"].rolling(5).min().dropna()
        if len(hh)>2 and len(ll)>2:
            if hh.iloc[-1]>hh.iloc[-2] and ll.iloc[-1]>ll.iloc[-2]:
                struct="Bullish"
            elif hh.iloc[-1]<hh.iloc[-2] and ll.iloc[-1]<ll.iloc[-2]:
                struct="Bearish"

    sup=df["Low"].tail(20).min()
    res=df["High"].tail(20).max()
    conviction="Tinggi" if struct==fib_bias and struct!="Konsolidasi" else ("Sedang" if struct!="Konsolidasi" else "Rendah")
    return {"structure":struct,"fib_bias":fib_bias,"support":sup,"resistance":res,
            "current":cur,"change":change,"conviction":conviction}

# ============ MAIN SCAN ============
cglist=cg_coin_list()
results=[]
progress=st.progress(0, text="Memulai pemindaian...")

for i,sym in enumerate(COINS):
    # Priority: Binance data (accurate)
    df=fetch_binance(sym, tf)
    if df is None or df.empty:
        cid=fuzzy_find(sym, cglist)
        df=fetch_coingecko(cid, tf)
    res=analyze_df(df)
    res["symbol"]=sym
    results.append(res)
    if i%3==0:
        progress.progress((i+1)/len(COINS), text=f"{sym} ({i+1}/{len(COINS)})")
    time.sleep(0.01)

df=pd.DataFrame(results)
st.success(f"✅ Pemindaian selesai ({len(df)} koin)")

st.dataframe(df[["symbol","structure","fib_bias","support","resistance","current","change","conviction"]])

# ============ CHART ============
symbol=st.selectbox("📊 Pilih koin untuk grafik:", sorted(df["symbol"]))
df_chart=fetch_binance(symbol, tf)
if df_chart is None or df_chart.empty:
    cid=fuzzy_find(symbol, cglist)
    df_chart=fetch_coingecko(cid, tf)

if df_chart is None or df_chart.empty:
    st.warning("⚠️ Data tidak tersedia untuk simbol ini.")
else:
    df_chart["EMA20"]=df_chart["Close"].ewm(span=20).mean()
    df_chart["EMA50"]=df_chart["Close"].ewm(span=50).mean()
    ma=df_chart["Close"].rolling(20).mean()
    sd=df_chart["Close"].rolling(20).std()
    df_chart["BB_up"]=ma+2*sd
    df_chart["BB_dn"]=ma-2*sd
    delta=df_chart["Close"].diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss=-delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs=gain/loss
    df_chart["RSI"]=100-(100/(1+rs))

    fig=go.Figure()
    if chart_style=="Candlestick":
        fig.add_trace(go.Candlestick(x=df_chart.index,open=df_chart["Open"],high=df_chart["High"],
                                     low=df_chart["Low"],close=df_chart["Close"],name="Price"))
    else:
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["Close"],mode="lines",name="Close"))
    for i in ["EMA20","EMA50"]:
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart[i],mode="lines",name=i))
    if {"BB_up","BB_dn"}.issubset(df_chart.columns):
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["BB_up"],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["BB_dn"],fill="tonexty",line=dict(width=0),showlegend=False))
    last=analyze_df(df_chart)
    fig.add_hline(y=last["support"], line=dict(color="green",dash="dot"), annotation_text="Support")
    fig.add_hline(y=last["resistance"], line=dict(color="red",dash="dot"), annotation_text="Resistance")
    fig.update_layout(template="plotly_dark",height=520,
                      title=f"{symbol} | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
    st.plotly_chart(fig, use_container_width=True)

    # RSI subplot
    if "RSI" in df_chart.columns:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=df_chart.index,y=df_chart["RSI"],mode="lines",name="RSI14"))
        fig2.update_layout(template="plotly_dark",height=200,yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("✅ Data Spot via Binance (utama) + CoinGecko (fallback) • EMA/RSI/BB • Struktur • Fibonacci • Support/Resistance • Conviction")
