import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time

# ============== CONFIG ==============
st.set_page_config(layout="wide", page_title="AI Analyst Spot (Gecko Fix)")
st.title("🚀 Instant AI Analyst — Spot via CoinGecko (Fixed ID Mapping)")
st.caption("Data Spot • Struktur • Fibonacci • Support/Resistance • Conviction")
st.markdown("---")

col1, col2 = st.columns([2, 8])
tf = col1.selectbox("🕒 Timeframe", ["1d", "4h", "1h"])
chart_style = col2.selectbox("💹 Jenis Grafik", ["Candlestick", "Line"])

if st.button("🔄 Scan ulang"):
    st.cache_data.clear()
    st.experimental_rerun()

# ============== COINS ==============
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT",
    "AVAXUSDT","DOTUSDT","MATICUSDT","SHIBUSDT","TRXUSDT","LTCUSDT","PEPEUSDT",
    "INJUSDT","OPUSDT","ARBUSDT","LINKUSDT","ATOMUSDT","UNIUSDT","APTUSDT",
    "SUIUSDT","TIAUSDT","SEIUSDT","RNDRUSDT","FETUSDT","AAVEUSDT","JASMYUSDT",
    "GALAUSDT","BCHUSDT","FTMUSDT","LDOUSDT","MINAUSDT","FLOWUSDT","CRVUSDT",
    "DYDXUSDT","KAVAUSDT","IMXUSDT","IDUSDT","MASKUSDT","AGIXUSDT","RUNEUSDT",
    "WLDUSDT","ORDIUSDT","BONKUSDT","BOMEUSDT",
    "HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT"
]

# manual mapping symbol -> CoinGecko ID
COINGECKO_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "SOL": "solana",
    "XRP": "ripple", "DOGE": "dogecoin", "ADA": "cardano", "AVAX": "avalanche-2",
    "DOT": "polkadot", "MATIC": "matic-network", "SHIB": "shiba-inu",
    "TRX": "tron", "LTC": "litecoin", "PEPE": "pepe", "INJ": "injective-protocol",
    "OP": "optimism", "ARB": "arbitrum", "LINK": "chainlink", "ATOM": "cosmos",
    "UNI": "uniswap", "APT": "aptos", "SUI": "sui", "TIA": "celestia",
    "SEI": "sei-network", "RNDR": "render-token", "FET": "fetch-ai",
    "AAVE": "aave", "JASMY": "jasmycoin", "GALA": "gala", "BCH": "bitcoin-cash",
    "FTM": "fantom", "LDO": "lido-dao", "MINA": "mina-protocol", "FLOW": "flow",
    "CRV": "curve-dao-token", "DYDX": "dydx", "KAVA": "kava", "IMX": "immutable-x",
    "ID": "space-id", "MASK": "mask-network", "AGIX": "singularitynet",
    "RUNE": "thorchain", "WLD": "worldcoin-wld", "ORDI": "ordi", "BONK": "bonk",
    "BOME": "book-of-meme",
}

API = "https://api.coingecko.com/api/v3"

# ============== HELPERS ==============
@st.cache_data(ttl=600)
def get_coin_id(symbol: str):
    base = symbol.replace("USDT","").upper()
    if base in COINGECKO_MAP:
        return COINGECKO_MAP[base]
    return base.lower()  # fallback

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

# ============== MAIN SCAN ==============
results=[]
progress=st.progress(0, text="Memulai scan CoinGecko...")
for i,sym in enumerate(COINS):
    cid=get_coin_id(sym)
    df=fetch_chart(cid, tf)
    res=analyze_df(df)
    res["symbol"]=sym
    res["coin_id"]=cid
    results.append(res)
    if i%3==0:
        progress.progress((i+1)/len(COINS), text=f"{sym} ({i+1}/{len(COINS)})")
    time.sleep(0.02)

df=pd.DataFrame(results)
st.success(f"✅ Pemindaian selesai ({len(df)} coin)")

st.dataframe(df[["symbol","structure","fib_bias","support","resistance","current","change","conviction"]])

# ============== CHART ==============
symbol=st.selectbox("Pilih koin untuk grafik:", sorted(df["symbol"]))
cid=get_coin_id(symbol)
chart=fetch_chart(cid, tf)

if chart is None or chart.empty:
    st.warning("⚠️ Data tidak tersedia.")
else:
    fig=go.Figure()
    if chart_style=="Candlestick":
        fig.add_trace(go.Candlestick(x=chart.index,open=chart["Open"],high=chart["High"],
                                     low=chart["Low"],close=chart["Close"],name="Price"))
    else:
        fig.add_trace(go.Scatter(x=chart.index,y=chart["Close"],mode="lines",name="Close"))
    last=analyze_df(chart)
    fig.add_hline(y=last["support"], line=dict(color="green",dash="dot"))
    fig.add_hline(y=last["resistance"], line=dict(color="red",dash="dot"))
    fig.update_layout(template="plotly_dark",height=500,title=f"{symbol} | {last['structure']} | {last['fib_bias']}")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Data via CoinGecko • Struktur • Fibonacci • Support/Resistance • Conviction")
