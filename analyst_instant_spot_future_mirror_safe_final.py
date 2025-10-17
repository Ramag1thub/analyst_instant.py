import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide", page_title="AI Analyst — TradingView Mirror Edition")
st.title("🚀 AI Analyst — TradingView Mirror Data (Spot Accurate)")
st.caption("Akurat seperti TradingView • Struktur • Fibonacci • EMA/RSI/BB • Support/Resistance • Conviction")
st.markdown("---")

col1, col2 = st.columns([2,8])
tf = col1.selectbox("🕒 Timeframe", ["1d","4h","1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick","Line"])

if st.button("🔄 Scan ulang"):
    st.cache_data.clear()
    st.experimental_rerun()

# =====================
# COIN LIST
# =====================
COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "PEPEUSDT","ARBUSDT","OPUSDT","APTUSDT","SUIUSDT","TIAUSDT","INJUSDT","RNDRUSDT","ATOMUSDT","UNIUSDT",
    "FTMUSDT","LDOUSDT","FLOWUSDT","AAVEUSDT","GALAUSDT","MASKUSDT","BONKUSDT","BOMEUSDT","ORDIUSDT",
    "HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT"
]

# =====================
# TRADINGVIEW MIRROR API
# =====================
def fetch_tv_data(symbol, tf):
    """Ambil data candle dari TradingView mirror"""
    tf_map = {"1h":"60","4h":"240","1d":"1D"}
    url = f"https://api.tradingview.com/markets/binance:{symbol}/charts"
    try:
        r = requests.get(url, params={"symbol":symbol, "interval":tf_map[tf], "range":"90"}, timeout=10)
        if r.status_code != 200:
            return None
        js = r.json()
        if "candles" not in js:
            return None
        data = js["candles"]
        if not data:
            return None
        df = pd.DataFrame(data, columns=["time","Open","High","Low","Close","Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df
    except Exception:
        return None

# =====================
# ANALISIS
# =====================
def analyze_df(df):
    if df is None or df.empty:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah"}
    cur = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df)>1 else cur
    change = ((cur-prev)/prev*100) if prev!=0 else 0
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
    return {"structure":struct,"fib_bias":fib_bias,"support":sup,"resistance":res,"current":cur,"change":change,"conviction":conviction}

# =====================
# SCAN
# =====================
st.info("📡 Mengambil data TradingView Mirror ...")
results=[]
progress=st.progress(0, text="Memulai pemindaian...")
for i,sym in enumerate(COINS):
    df = fetch_tv_data(sym, tf)
    res = analyze_df(df)
    res["symbol"] = sym
    results.append(res)
    if i%3==0:
        progress.progress((i+1)/len(COINS), text=f"{sym} ({i+1}/{len(COINS)})")
    time.sleep(0.01)

df = pd.DataFrame(results)
st.success(f"✅ Pemindaian selesai ({len(df)} koin)")

st.dataframe(df[["symbol","structure","fib_bias","support","resistance","current","change","conviction"]])

# =====================
# CHART
# =====================
symbol = st.selectbox("📊 Pilih koin untuk grafik:", sorted(df["symbol"]))
chart = fetch_tv_data(symbol, tf)

if chart is None or chart.empty:
    st.warning("⚠️ Data tidak tersedia untuk simbol ini.")
else:
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
        fig.add_trace(go.Scatter(x=chart.index,y=chart[i],mode="lines",name=i))
    if {"BB_up","BB_dn"}.issubset(chart.columns):
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_up"],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_dn"],fill="tonexty",line=dict(width=0),showlegend=False))
    last=analyze_df(chart)
    fig.add_hline(y=last["support"], line=dict(color="green",dash="dot"), annotation_text="Support")
    fig.add_hline(y=last["resistance"], line=dict(color="red",dash="dot"), annotation_text="Resistance")
    fig.update_layout(template="plotly_dark",height=520,
                      title=f"{symbol} | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
    st.plotly_chart(fig, use_container_width=True)

    if "RSI" in chart.columns:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=chart.index,y=chart["RSI"],mode="lines",name="RSI14"))
        fig2.update_layout(template="plotly_dark",height=200,yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2, use_container_width=True)

st.caption("✅ Data real-time via TradingView Mirror API • EMA20/50 • BB(20,2) • RSI14 • Struktur • Fibonacci • Support/Resistance • Conviction")
