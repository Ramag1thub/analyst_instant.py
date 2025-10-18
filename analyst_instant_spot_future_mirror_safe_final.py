import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time
import difflib

st.set_page_config(layout="wide", page_title="AI Analyst — Hybrid Ultimate")
st.title("🚀 AI Analyst — Hybrid Ultimate (Multi-Source v2)")
st.caption("Akurat penuh • Binance + CoinGecko + CoinMarketCap + DexScreener • Fibonacci • EMA/RSI/BB/MACD/OBV • Support/Resistance • Conviction")
st.markdown("---")

col1, col2 = st.columns([2,8])
tf = col1.selectbox("🕒 Timeframe", ["1d","4h","1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick","Line"])

if st.button("🔄 Scan ulang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ================= COIN LIST =================
COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "PEPEUSDT","ARBUSDT","OPUSDT","APTUSDT","SUIUSDT","TIAUSDT","INJUSDT","RNDRUSDT","ATOMUSDT","UNIUSDT",
    "FTMUSDT","LDOUSDT","FLOWUSDT","AAVEUSDT","GALAUSDT","MASKUSDT","BONKUSDT","BOMEUSDT","ORDIUSDT",
    "HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT"
]

# ================= API ENDPOINTS =================
BIN_API = "https://api.binance.com/api/v3/klines"
CG_API = "https://api.coingecko.com/api/v3"
CMC_API = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart"
DEX_API = "https://api.dexscreener.com/latest/dex/pairs"

# ================= FETCH FUNCTIONS =================
def fetch_binance(symbol, tf):
    tf_map = {"1h":"1h","4h":"4h","1d":"1d"}
    try:
        r = requests.get(BIN_API, params={"symbol":symbol,"interval":tf_map[tf],"limit":500}, timeout=10)
        if r.status_code != 200: return None
        data = r.json()
        df = pd.DataFrame(data)
        df.columns = ["time","Open","High","Low","Close","Volume","a","b","c","d","e","f"]
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except:
        return None

@st.cache_data(ttl=3600)
def cg_list():
    try:
        r=requests.get(f"{CG_API}/coins/list",timeout=15)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except:
        return pd.DataFrame(columns=["id","symbol","name"])

def cg_find(symbol, df):
    s = symbol.lower().replace("usdt","")
    exact = df[df["symbol"]==s]
    if not exact.empty: return exact.iloc[0]["id"]
    close = difflib.get_close_matches(s, df["name"].tolist(), n=1, cutoff=0.5)
    if close:
        r = df[df["name"]==close[0]]
        if not r.empty: return r.iloc[0]["id"]
    return None

def fetch_cg(coin_id, tf):
    days = {"1d":30,"4h":90,"1h":30}[tf]
    try:
        r = requests.get(f"{CG_API}/coins/{coin_id}/market_chart", params={"vs_currency":"usd","days":days}, timeout=15)
        js = r.json()
        if "prices" not in js: return None
        df = pd.DataFrame(js["prices"], columns=["time","Close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df["Open"] = df["Close"].shift(1)
        df["High"] = df["Close"].rolling(3).max()
        df["Low"] = df["Close"].rolling(3).min()
        df["Volume"] = 0
        return df.dropna()
    except:
        return None

def fetch_cmc(symbol):
    try:
        base = symbol.replace("USDT","")
        r = requests.get(CMC_API, params={"symbol":base, "range":"1M"}, timeout=15)
        if r.status_code != 200: return None
        js = r.json()
        pts = js.get("data", {}).get("points", {})
        if not pts: return None
        df = pd.DataFrame([(int(k), v["v"][0]) for k,v in pts.items()], columns=["time","Close"])
        df["time"]=pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df["Open"]=df["Close"].shift(1)
        df["High"]=df["Close"].rolling(3).max()
        df["Low"]=df["Close"].rolling(3).min()
        df["Volume"]=0
        return df.dropna()
    except:
        return None

def fetch_dex(symbol):
    try:
        base = symbol.replace("USDT","").lower()
        r = requests.get(f"{DEX_API}/ethereum/{base}-usdt", timeout=15)
        if r.status_code != 200: return None
        js = r.json()
        price = js.get("pair", {}).get("priceUsd")
        if not price: return None
        df = pd.DataFrame({"time":[pd.Timestamp.now()], "Open":[float(price)], "High":[float(price)], "Low":[float(price)], "Close":[float(price)], "Volume":[0]})
        df.set_index("time", inplace=True)
        return df
    except:
        return None

# ================= ANALYSIS =================
def analyze(df):
    if df is None or df.empty:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah"}
    cur=df["Close"].iloc[-1]
    prev=df["Close"].iloc[-2] if len(df)>1 else cur
    change=((cur-prev)/prev*100) if prev!=0 else 0
    hi,lo=df["High"].max(),df["Low"].min()
    fib_bias="Netral"
    if hi and lo:
        diff=hi-lo
        fib61, fib38=hi-diff*0.618, hi-diff*0.382
        if cur>fib61: fib_bias="Bullish"
        elif cur<fib38: fib_bias="Bearish"
    struct="Konsolidasi"
    if len(df)>10:
        hh=df["High"].rolling(5).max().dropna()
        ll=df["Low"].rolling(5).min().dropna()
        if len(hh)>2 and len(ll)>2:
            if hh.iloc[-1]>hh.iloc[-2] and ll.iloc[-1]>ll.iloc[-2]: struct="Bullish"
            elif hh.iloc[-1]<hh.iloc[-2] and ll.iloc[-1]<ll.iloc[-2]: struct="Bearish"
    sup=df["Low"].tail(20).min()
    res=df["High"].tail(20).max()
    conviction="Tinggi" if struct==fib_bias and struct!="Konsolidasi" else ("Sedang" if struct!="Konsolidasi" else "Rendah")
    return {"structure":struct,"fib_bias":fib_bias,"support":sup,"resistance":res,"current":cur,"change":change,"conviction":conviction}

# ================= SCAN =================
cgdb = cg_list()
results = []
progress = st.progress(0, "📡 Memulai pemindaian...")

for i, sym in enumerate(COINS):
    src = None
    df = fetch_binance(sym, tf)
    if df is not None and not df.empty:
        src = "Binance"
    else:
        cid = cg_find(sym, cgdb)
        if cid:
            df = fetch_cg(cid, tf)
            if df is not None and not df.empty: src = "CoinGecko"
    if (df is None or df.empty) and not src:
        df = fetch_cmc(sym)
        if df is not None and not df.empty: src = "CoinMarketCap"
    if (df is None or df.empty) and not src:
        df = fetch_dex(sym)
        if df is not None and not df.empty: src = "DexScreener"

    res = analyze(df)
    res["symbol"] = sym
    res["source"] = src if src else "❌ Tidak Ada"
    results.append(res)

    if i % 3 == 0:
        progress.progress((i+1)/len(COINS), f"{sym} ({i+1}/{len(COINS)})")
    time.sleep(0.05)

df = pd.DataFrame(results)
st.success(f"✅ Pemindaian selesai ({len(df)} koin)")

st.dataframe(df[["symbol","source","structure","fib_bias","support","resistance","current","change","conviction"]])

# ================= CHART =================
symbol = st.selectbox("📊 Pilih koin untuk grafik:", sorted(df["symbol"]))
cgdb = cg_list()

chart = fetch_binance(symbol, tf)
source_used = "Binance"
if chart is None or chart.empty:
    cid = cg_find(symbol, cgdb)
    chart = fetch_cg(cid, tf)
    source_used = "CoinGecko"
if chart is None or chart.empty:
    chart = fetch_cmc(symbol)
    source_used = "CoinMarketCap"
if chart is None or chart.empty:
    chart = fetch_dex(symbol)
    source_used = "DexScreener"

if chart is None or chart.empty:
    st.warning("⚠️ Semua sumber gagal memuat data untuk koin ini.")
else:
    st.info(f"📈 Data ditampilkan dari sumber: **{source_used}**")

    # === Technical Indicators ===
    chart["EMA20"]=chart["Close"].ewm(span=20).mean()
    chart["EMA50"]=chart["Close"].ewm(span=50).mean()
    ma=chart["Close"].rolling(20).mean()
    sd=chart["Close"].rolling(20).std()
    chart["BB_up"]=ma+2*sd
    chart["BB_dn"]=ma-2*sd

    delta=chart["Close"].diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=-delta.clip(upper=0).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss
    chart["RSI"]=100-(100/(1+rs))

    ema12=chart["Close"].ewm(span=12,adjust=False).mean()
    ema26=chart["Close"].ewm(span=26,adjust=False).mean()
    chart["MACD"]=ema12-ema26
    chart["Signal"]=chart["MACD"].ewm(span=9,adjust=False).mean()

    obv=[0]
    for i in range(1,len(chart)):
        if chart["Close"].iloc[i]>chart["Close"].iloc[i-1]:
            obv.append(obv[-1]+chart["Volume"].iloc[i])
        elif chart["Close"].iloc[i]<chart["Close"].iloc[i-1]:
            obv.append(obv[-1]-chart["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    chart["OBV"]=obv

    # === PRICE CHART ===
    fig=go.Figure()
    if chart_style=="Candlestick":
        fig.add_trace(go.Candlestick(x=chart.index,open=chart["Open"],high=chart["High"],low=chart["Low"],close=chart["Close"],name="Price"))
    else:
        fig.add_trace(go.Scatter(x=chart.index,y=chart["Close"],mode="lines",name="Close"))
    for i in ["EMA20","EMA50"]:
        fig.add_trace(go.Scatter(x=chart.index,y=chart[i],mode="lines",name=i))
    if {"BB_up","BB_dn"}.issubset(chart.columns):
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_up"],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_dn"],fill="tonexty",line=dict(width=0),showlegend=False))
    last=analyze(chart)
    fig.add_hline(y=last["support"], line=dict(color="green",dash="dot"), annotation_text="Support")
    fig.add_hline(y=last["resistance"], line=dict(color="red",dash="dot"), annotation_text="Resistance")
    fig.update_layout(template="plotly_dark",height=520,
                      title=f"{symbol} | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
    st.plotly_chart(fig, use_container_width=True)
