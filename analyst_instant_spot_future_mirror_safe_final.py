# analyst_instant_spot_only_safe_final.py
# Versi Spot Only — tanpa Futures
# Fokus pada koin hype, launchcoin, dan top market cap
# Aman di Streamlit Cloud, tanpa API eksternal Binance/OKX

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide", page_title="Instant AI Analyst — Spot Only (Mirror Safe)")
st.title("🚀 Instant AI Analyst — Spot Only (Yahoo Mirror)")
st.caption("350+ koin hype & launchcoin • Chart interaktif • EMA/RSI/BB • Tanpa error")
st.markdown("---")

# ---------------- UI ----------------
col1, col2 = st.columns([2, 8])
selected_tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ---------------- KOIN POPULER (Spot only) ----------------
# Koin hype, lauchcoin, meme, dan top market cap
COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","DOGEUSDT","PEPEUSDT","SHIBUSDT",
    "TONUSDT","ARBUSDT","OPUSDT","AVAXUSDT","MATICUSDT","INJUSDT","TIAUSDT",
    "APTUSDT","SUIUSDT","STRKUSDT","PYTHUSDT","ZROUSDT","ARKMUSDT","JTOUSDT",
    "REZUSDT","AIOZUSDT","RNDRUSDT","FETUSDT","LINKUSDT","ADAUSDT","DOTUSDT",
    "BONKUSDT","WIFUSDT","DOGSUSDT","BOOKUSDT","LDOUSDT","JUPUSDT","UNIUSDT",
    "NOTUSDT","ENAUSDT","PIXELUSDT","BLURUSDT","ACEUSDT","DEGENUSDT","XAIUSDT",
    "HYPEUSDT","ONDOUSDT","MAGAUSDT","LUNAUSDT","CORGIAIUSDT","MEMEUSDT","NORMUSDT"
]
COINS = sorted(list(set(COINS)))
TOTAL = len(COINS)

def to_yf(sym: str) -> str:
    if sym.endswith("USDT"): return sym[:-4] + "-USD"
    return sym

YF_INTERVAL = {"1d": "1d", "4h": "60m", "1h": "60m"}

# ---------------- FETCH ----------------
@st.cache_data(ttl=120, show_spinner=True)
def fetch_spot(symbols, interval):
    out = {}
    for sym in symbols:
        yf_sym = to_yf(sym)
        try:
            df = yf.download(yf_sym, period="90d", interval=interval, progress=False)
            if df is not None and not df.empty:
                out[sym] = df
        except Exception:
            continue
    return out

# ---------------- ANALYSIS ----------------
def analyze_df(df):
    if df is None or df.empty:
        return {"structure":"No Data","bias":"Netral","current":None,"change":0.0}
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close":"Close"})
        else:
            return {"structure":"No Data","bias":"Netral","current":None,"change":0.0}
    df = df.dropna(subset=["Close"], errors="ignore")
    if len(df)<3:
        return {"structure":"Data Kurang","bias":"Netral","current":None,"change":0.0}
    df["High"] = df.get("High", df["Close"])
    df["Low"]  = df.get("Low", df["Close"])
    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df)>1 else cur
    change = ((cur - prev)/prev)*100 if prev!=0 else 0
    hi, lo = df["High"].max(), df["Low"].min()
    diff = hi-lo if hi and lo else 0
    bias = "Netral"
    if diff>0:
        fib61, fib38 = hi-diff*0.618, hi-diff*0.382
        if cur>fib61: bias="Bullish"
        elif cur<fib38: bias="Bearish"
    struct="Konsolidasi"
    try:
        rec=df[["High","Low"]].tail(14)
        if len(rec)>=14:
            rh=rec["High"].rolling(5,center=True).max().dropna()
            rl=rec["Low"].rolling(5,center=True).min().dropna()
            if len(rh)>=2 and len(rl)>=2:
                if rh.iloc[-1]>rh.iloc[-2] and rl.iloc[-1]>rl.iloc[-2]:
                    struct="Bullish"
                elif rh.iloc[-1]<rh.iloc[-2] and rl.iloc[-1]<rl.iloc[-2]:
                    struct="Bearish"
    except Exception:
        pass
    return {"structure":struct,"bias":bias,"current":cur,"change":change}

# ---------------- SCAN ----------------
st.info(f"⏳ Memindai {TOTAL} koin spot (Yahoo Mirror)...")
start = time.time()
data = fetch_spot(COINS, YF_INTERVAL[selected_tf])
results=[]
for i,(sym,df) in enumerate(data.items()):
    res=analyze_df(df)
    res["symbol"]=sym
    results.append(res)
    st.progress((i+1)/len(data), text=f"📊 {sym}")
    time.sleep(0.01)
elapsed=time.time()-start
st.success(f"✅ Selesai dalam {elapsed:.1f} detik ({len(results)} koin sukses)")

df=pd.DataFrame(results)
df["conviction"]=df.apply(lambda r:"Tinggi" if r["structure"]==r["bias"] and r["structure"] in ["Bullish","Bearish"]
    else "Sedang" if r["structure"] in ["Bullish","Bearish"] or r["bias"] in ["Bullish","Bearish"] else "Rendah",axis=1)

# ---------------- TOP MOVERS ----------------
st.subheader("📈 Top 5 Bullish Movers")
bull=df.sort_values("change",ascending=False).head(5)
for _,r in bull.iterrows():
    st.markdown(f"**{r['symbol']}** → +{r['change']:.2f}% | {r['structure']} | {r['conviction']}")

st.subheader("📉 Top 5 Bearish Movers")
bear=df.sort_values("change",ascending=True).head(5)
for _,r in bear.iterrows():
    st.markdown(f"**{r['symbol']}** → {r['change']:.2f}% | {r['structure']} | {r['conviction']}")

st.dataframe(df[["symbol","structure","bias","change","conviction"]].fillna("N/A"))

# ---------------- CHART ----------------
symbol=st.selectbox("Pilih koin untuk grafik:", sorted(df["symbol"].unique()))
yf_sym=to_yf(symbol)
try:
    chart=yf.download(yf_sym,period="120d",interval=YF_INTERVAL[selected_tf],progress=False)
except Exception:
    chart=None

if chart is None or chart.empty:
    st.warning("⚠️ Data grafik tidak tersedia.")
else:
    chart["EMA20"]=chart["Close"].ewm(span=20).mean()
    chart["EMA50"]=chart["Close"].ewm(span=50).mean()
    ma=chart["Close"].rolling(20).mean(); sd=chart["Close"].rolling(20).std()
    chart["BB_up"]=ma+2*sd; chart["BB_dn"]=ma-2*sd
    delta=chart["Close"].diff(); gain=delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss=-delta.clip(upper=0).ewm(alpha=1/14).mean()
    rs=gain/loss; chart["RSI"]=100-(100/(1+rs))

    fig=go.Figure()
    if chart_style=="Candlestick":
        fig.add_trace(go.Candlestick(x=chart.index,open=chart["Open"],high=chart["High"],low=chart["Low"],close=chart["Close"],name="Price"))
    else:
        fig.add_trace(go.Scatter(x=chart.index,y=chart["Close"],mode="lines",name="Close"))
    for i in ["EMA20","EMA50"]:
        if i in chart.columns: fig.add_trace(go.Scatter(x=chart.index,y=chart[i],mode="lines",name=i))
    if {"BB_up","BB_dn"}.issubset(chart.columns):
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_up"],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=chart.index,y=chart["BB_dn"],fill="tonexty",line=dict(width=0),showlegend=False))
    fig.update_layout(template="plotly_dark",height=520)
    st.plotly_chart(fig,use_container_width=True)

    if "RSI" in chart.columns:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=chart.index,y=chart["RSI"],mode="lines",name="RSI14"))
        fig2.update_layout(template="plotly_dark",height=200,yaxis=dict(range=[0,100]))
        st.plotly_chart(fig2,use_container_width=True)

st.caption("Data Spot via Yahoo Mirror • Indikator: EMA20/50, BB(20,2), RSI14 • Fokus coin hype & launchcoin")
