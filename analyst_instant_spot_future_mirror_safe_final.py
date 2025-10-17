import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time
import difflib

# ============== CONFIG ==============
st.set_page_config(layout="wide", page_title="AI Analyst Spot (Gecko Ultimate)")
st.title("🚀 Instant AI Analyst — Spot via CoinGecko (Ultimate Search + Indicators)")
st.caption("CoinGecko + Smart Finder • Struktur • Fibonacci • EMA/RSI/BB • Support/Resistance • Conviction")
st.markdown("---")

col1, col2 = st.columns([2, 8])
tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan ulang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ============== COINS ==============
COINS = [
    "BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","DOT","MATIC","SHIB","TRX","LTC","PEPE",
    "ARB","OP","APT","SUI","TIA","INJ","SEI","WLD","RNDR","IMX","ATOM","UNI","FIL","FTM","FLOW",
    "LDO","AAVE","GALA","STX","DYDX","VET","MASK","KAVA","CRV","MINA","RUNE","XLM","EOS","AGIX",
    "CHZ","CRO","XMR","ETC","MKR","CELO","PYTH","JASMY","TURBO","BONK","BOME","ORDI",
    # custom requested tokens
    "HYPE","ASTER","LAUNCHCOIN","USELESS"
]

API = "https://api.coingecko.com/api/v3"

# ============== HELPERS ==============
@st.cache_data(ttl=3600)
def get_coin_list():
    try:
        r = requests.get(f"{API}/coins/list", timeout=20)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame(columns=["id","symbol","name"])

@st.cache_data(ttl=3600)
def find_coin_id(symbol: str, coins_df: pd.DataFrame):
    s = symbol.lower().replace("usdt","")
    # exact match by symbol
    matches = coins_df[coins_df["symbol"] == s]
    if not matches.empty:
        return matches.iloc[0]["id"]
    # fuzzy match by similarity
    names = coins_df["name"].tolist()
    close = difflib.get_close_matches(s, names, n=1, cutoff=0.6)
    if close:
        row = coins_df[coins_df["name"] == close[0]]
        if not row.empty:
            return row.iloc[0]["id"]
    # last fallback
    return s

@st.cache_data(ttl=600)
def fetch_chart(coin_id: str, tf: str):
    days = {"1d": 30, "4h": 90, "1h": 30}[tf]
    url = f"{API}/coins/{coin_id}/market_chart"
    try:
        r = requests.get(url, params={"vs_currency":"usd","days":days}, timeout=20)
        if r.status_code != 200:
            return None
        js = r.json()
        prices = js.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["time","Close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        # Approximate OHLC
        df["Open"] = df["Close"].shift(1)
        df["High"] = df["Close"].rolling(3).max()
        df["Low"] = df["Close"].rolling(3).min()
        df["Volume"] = 0
        return df.dropna()
    except Exception:
        return None

def analyze_df(df):
    if df is None or df.empty:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah"}

    cur = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df)>1 else cur
    change = ((cur-prev)/prev*100) if prev!=0 else 0.0
    hi, lo = df["High"].max(), df["Low"].min()
    diff = hi-lo if hi and lo else 0
    fib_bias = "Netral"
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
    return {"structure":struct,"fib_bias":fib_bias,"support":sup,"resistance":res,"current":cur,"change":change,"conviction":conviction}

# ============== SCAN ==============
st.info("📡 Mengambil daftar koin dari CoinGecko...")
coin_list = get_coin_list()
st.success(f"✅ {len(coin_list)} token ditemukan di CoinGecko")

results=[]
progress=st.progress(0, text="Memulai pemindaian...")
for i,sym in enumerate(COINS):
    cid=find_coin_id(sym, coin_list)
    df=fetch_chart(cid, tf)
    res=analyze_df(df)
    res["symbol"]=sym
    res["coin_id"]=cid
    results.append(res)
    if i%3==0:
        progress.progress((i+1)/len(COINS), text=f"{sym} ({i+1}/{len(COINS)})")
    time.sleep(0.01)

df=pd.DataFrame(results)
st.success(f"✅ Pemindaian selesai ({len(df)} coin)")

st.dataframe(df[["symbol","structure","fib_bias","support","resistance","current","change","conviction"]])

# ============== CHART ==============
symbol=st.selectbox("📊 Pilih koin untuk grafik:", sorted(df["symbol"]))
row=df[df["symbol"]==symbol].iloc[0]
cid=row["coin_id"]
chart=fetch_chart(cid, tf)

if chart is None or chart.empty:
    st.warning("⚠️ Data tidak tersedia untuk simbol ini.")
else:
    # Indicators
    chart["EMA20"]=chart["Close"].ewm(span=20).mean()
    chart["EMA50"]=chart["Close"].ewm(span=50).mean()
    ma=chart["Close"].rolling(20).mean()
    sd=chart["Close"].rolling(20).std()
    chart["BB_up"]=ma+2*sd
    chart["BB_dn"]=ma-2*sd
    delta=chart["Close"].diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss=-delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs=gain/loss
    chart["RSI"]=100-(100/(1+rs))

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
    last=analyze_df(chart)
    fig.add_hline(y=last["support"], line=dict(color="green",dash="dot"), annotation_text="Support")
    fig.add_hline(y=last["resistance"], line=dict(color="red",dash="dot"), annotation_text="Resistance")
    fig.update_layout(template="plotly_dark",height=520,
                      title=f"{symbol} ({cid}) | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    if "RSI" in chart.columns:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=chart.index,y=chart["RSI"],mode="lines",name="RSI14"))
        fig2.update_layout(template="plotly_dark",height=200,yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("✅ Data via CoinGecko • Smart Fuzzy Finder • EMA20/50 • BB(20,2) • RSI14 • Struktur • Fibonacci • Support/Resistance • Conviction")
