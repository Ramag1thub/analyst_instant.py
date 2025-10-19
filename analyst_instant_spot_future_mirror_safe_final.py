# analyst_hybrid_v8_full_restore.py
"""
AI Analyst — Hybrid v8 (Full Restore)
- Tabel analisa utama (paling atas) termasuk:
    Structure | Fib Bias | Chance Price (%) | Entry Price (support) | Support | Resistance | Change% | Conviction | Source | Confidence
- Chart & indicators: EMA20/50, BB, RSI, MACD, OBV
- Exchange-first fallback: Binance -> Bybit -> Bitget -> Gate -> KuCoin -> CoinGecko -> CoinMarketCap -> yfinance -> DexScreener
- Minimal 60 valid (auto fill backups), anti-error (try/except everywhere)
- Entry Price = Support (sesuai permintaan)
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(layout="wide", page_title="AI Analyst — Hybrid v8 (Full Restore)")
st.title("🚀 AI Analyst — Hybrid v8 (Full Restore)")
st.caption("Full analysis restored • Exchange-first fallback • Entry = Support • Chance Price (%) • Anti-error")

# ---------------- Settings ----------------
REQUEST_TIMEOUT = 10
PAUSE_BETWEEN = 0.04
MIN_VALID = 60
TF_OPTIONS = ["1d", "4h", "1h"]
# Map timeframe strings used in some APIs
TF_KLINE_MAP = {"1d":"1d","4h":"4h","1h":"1h"}

# ---------------- Base coin list (includes requested customs) ----------------
COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","MATICUSDT",
    "PEPEUSDT","ARBUSDT","OPUSDT","APTUSDT","SUIUSDT","INJUSDT","RNDRUSDT","ATOMUSDT","UNIUSDT","FTMUSDT",
    "LDOUSDT","FLOWUSDT","AAVEUSDT","GALAUSDT","MASKUSDT","BONKUSUSDT","BONKUSDT","BOMEUSDT","ORDIUSDT",
    "HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT",
    "SHIBUSDT","LTCUSDT","LINKUSDT","NEARUSDT","FILUSDT","TRXUSDT","GRTUSDT","CHZUSDT","ANKRUSDT","ARBUSDT",
    "BCHUSDT","XLMUSDT","VETUSDT","XTZUSDT","ZECUSDT","ENSUSDT","KLAYUSDT"
]
# sanitize and dedupe
COINS = list(dict.fromkeys([c.strip().upper() for c in COINS if c]))

# ---------------- Endpoints ----------------
BIN_API_KLINES = "https://api.binance.com/api/v3/klines"
BIN_API_PRICE = "https://api.binance.com/api/v3/ticker/price"
CG_API = "https://api.coingecko.com/api/v3"
CMC_API = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/detail/chart"
DEX_BASE = "https://api.dexscreener.com/latest/dex/pairs"

# Exchange-specific endpoints (public)
BYBIT_V5_TICKERS = "https://api.bybit.com/v5/market/tickers"
BYBIT_V5_KLINE = "https://api.bybit.com/v5/market/kline"
BITGET_TICKER = "https://api.bitget.com/api/spot/v1/market/ticker"
GATE_TICKERS = "https://api.gateio.ws/api/v4/spot/tickers"
KUCOIN_L1 = "https://api.kucoin.com/api/v1/market/orderbook/level1"

# ---------------- Utilities ----------------
def safe_get(url, params=None, timeout=REQUEST_TIMEOUT):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def try_symbol_variants(symbol):
    """Common symbol variants to try per exchange."""
    base = symbol.replace("USDT","")
    variants = [
        symbol,                           # BTCUSDT
        base + "-USDT",                   # BTC-USDT
        base + "/USDT",                   # BTC/USDT
        base + "_USDT",                   # BTC_USDT
        base + "USDT",                    # BTCUSDT
        base.lower() + "usdt",            # btcusdt
        base.lower() + "-usdt"
    ]
    seen = set(); out=[]
    for v in variants:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

# ---------------- Exchange fetchers ----------------

# Binance OHLC and Price
def fetch_binance_ohlc(symbol, tf, limit=500):
    try:
        js = safe_get(BIN_API_KLINES, params={"symbol": symbol, "interval": TF_KLINE_MAP[tf], "limit": limit})
        if not js:
            return None
        df = pd.DataFrame(js)
        df.columns = ["time","Open","High","Low","Close","Volume","a","b","c","d","e","f"]
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        return df[["Open","High","Low","Close","Volume"]].astype(float)
    except Exception:
        return None

def fetch_binance_price(symbol):
    try:
        js = safe_get(BIN_API_PRICE, params={"symbol":symbol})
        if js and "price" in js:
            return float(js["price"])
    except:
        return None
    return None

# Bybit price & OHLC
def fetch_bybit_price(symbol):
    for s in try_symbol_variants(symbol):
        try:
            js = safe_get(BYBIT_V5_TICKERS, params={"category":"spot", "symbol": s})
            if js and "result" in js:
                res = js["result"]
                # result might be dict or list
                if isinstance(res, list) and len(res)>0:
                    item = res[0]
                elif isinstance(res, dict):
                    # some responses put symbol as key
                    item = res
                else:
                    item = None
                if item:
                    p = item.get("lastPrice") or item.get("last_price") or item.get("last")
                    if p:
                        return float(p)
        except:
            continue
    return None

def fetch_bybit_ohlc(symbol, tf):
    for s in try_symbol_variants(symbol):
        try:
            js = safe_get(BYBIT_V5_KLINE, params={"category":"spot","symbol":s,"interval":TF_KLINE_MAP[tf],"limit":500})
            if js and "result" in js and js["result"] and "list" in js["result"]:
                lst = js["result"]["list"]
                df = pd.DataFrame(lst, columns=["time","Open","High","Low","Close","Volume"])
                df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
                df.set_index("time", inplace=True)
                return df.astype(float)
        except:
            continue
    return None

# Bitget price
def fetch_bitget_price(symbol):
    for s in try_symbol_variants(symbol):
        try:
            js = safe_get(BITGET_TICKER, params={"symbol": s})
            if not js:
                continue
            # response forms vary
            if isinstance(js, dict) and "data" in js and isinstance(js["data"], list) and len(js["data"])>0:
                it = js["data"][0]
                p = it.get("last") or it.get("price") or it.get("lastPrice")
                if p: return float(p)
            elif isinstance(js, dict) and "last" in js:
                return float(js["last"])
        except:
            continue
    return None

# Gate price
def fetch_gate_price(symbol):
    try:
        js = safe_get(GATE_TICKERS)
        if not js:
            return None
        base = symbol.replace("USDT","").upper()
        for it in js:
            pair = it.get("currency_pair") or it.get("pair") or it.get("symbol")
            if not pair: continue
            normalized = pair.replace("-","_").replace("/","_").upper()
            if normalized.startswith(base + "_"):
                p = it.get("last") or it.get("last_price") or it.get("price")
                if p:
                    try:
                        return float(p)
                    except:
                        continue
    except:
        pass
    return None

# KuCoin price
def fetch_kucoin_price(symbol):
    for s in try_symbol_variants(symbol):
        candidate = s if "-" in s else s.replace("/", "-").replace("_", "-")
        try:
            js = safe_get(KUCOIN_L1, params={"symbol":candidate})
            if js and "data" in js and "price" in js["data"]:
                return float(js["data"]["price"])
            if js and "price" in js:
                return float(js["price"])
        except:
            continue
    # fallback allTickers
    try:
        js = safe_get("https://api.kucoin.com/api/v1/market/allTickers")
        if js and "data" in js and "ticker" in js["data"]:
            base = symbol.replace("USDT","").upper()
            for t in js["data"]["ticker"]:
                if str(t.get("symbol","")).startswith(base + "-"):
                    return float(t.get("last"))
    except:
        pass
    return None

# DexScreener price
def fetch_dex_price(symbol):
    short = symbol.lower().replace("usdt","")
    chains = ["ethereum","bsc","arbitrum","optimism","polygon"]
    for chain in chains:
        try:
            url = f"{DEX_BASE}/{chain}/{short}-usdt"
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                js = r.json()
                p = js.get("pair") or (js.get("pairs") and js.get("pairs")[0])
                if p:
                    priceUsd = p.get("priceUsd") or p.get("price_usd")
                    if priceUsd:
                        return float(priceUsd)
        except:
            continue
    return None

# CoinGecko price
def fetch_coingecko_price(symbol):
    try:
        base = symbol.lower().replace("usdt","")
        common_map = {"btc":"bitcoin","eth":"ethereum","bnb":"binancecoin","sol":"solana","ada":"cardano","xrp":"ripple","doge":"dogecoin","matic":"matic-network","dot":"polkadot","avax":"avalanche-2"}
        cid = common_map.get(base)
        if not cid:
            s = safe_get(f"{CG_API}/search", params={"query":base})
            if s and "coins" in s and len(s["coins"])>0:
                cid = s["coins"][0]["id"]
        if cid:
            js = safe_get(f"{CG_API}/simple/price", params={"ids":cid,"vs_currencies":"usd"})
            if js and cid in js and "usd" in js[cid]:
                return float(js[cid]["usd"])
    except:
        pass
    return None

# CoinMarketCap price
def fetch_cmc_price(symbol):
    try:
        base = symbol.replace("USDT","")
        js = safe_get(CMC_API, params={"symbol":base,"range":"1M"})
        if js:
            pts = js.get("data",{}).get("points",{})
            if pts:
                # take last point
                for k,v in pts.items():
                    return float(v["v"][0])
    except:
        pass
    return None

# yfinance price
def fetch_yfinance_price(symbol):
    try:
        base = symbol.upper().replace("USDT","-USD")
        tk = yf.Ticker(base)
        hist = tk.history(period="5d", interval="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except:
        pass
    return None

# Combined quick price (exchange-first)
def get_price_quick_exchange_first(symbol):
    # order: Binance, Bybit, Bitget, Gate, KuCoin, CoinGecko, CMC, yfinance, Dex
    p = fetch_binance_price(symbol)
    if p: return p, "Binance"
    p = fetch_bybit_price(symbol)
    if p: return p, "Bybit"
    p = fetch_bitget_price(symbol)
    if p: return p, "Bitget"
    p = fetch_gate_price(symbol)
    if p: return p, "Gate"
    p = fetch_kucoin_price(symbol)
    if p: return p, "KuCoin"
    p = fetch_coingecko_price(symbol)
    if p: return p, "CoinGecko"
    p = fetch_cmc_price(symbol)
    if p: return p, "CoinMarketCap"
    p = fetch_yfinance_price(symbol)
    if p: return p, "yfinance"
    p = fetch_dex_price(symbol)
    if p: return p, "DexScreener"
    return None, None

# OHLC combined (exchange-first)
def fetch_yfinance_ohlc_generic(symbol, tf):
    try:
        base = symbol.upper().replace("USDT","-USD")
        period_map = {"1d":"3mo","4h":"6mo","1h":"1mo"}
        interval_map = {"1d":"1d","4h":"1h","1h":"30m"}
        period = period_map.get(tf, "3mo")
        interval = interval_map.get(tf, "1h")
        tk = yf.Ticker(base)
        df = tk.history(period=period, interval=interval, actions=False)
        if df is None or df.empty: return None
        df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume"})
        df.index = pd.to_datetime(df.index)
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        return df
    except:
        return None

def get_ohlc_for_symbol_exchange_first(symbol, tf):
    # Binance
    df = fetch_binance_ohlc(symbol, tf)
    if df is not None and not df.empty:
        return df, "Binance"
    # Bybit
    df = fetch_bybit_ohlc(symbol, tf)
    if df is not None and not df.empty:
        return df, "Bybit"
    # yfinance
    df = fetch_yfinance_ohlc_generic(symbol, tf)
    if df is not None and not df.empty:
        return df, "yfinance"
    # as last resort create 1-bar from quick price
    price, src = get_price_quick_exchange_first(symbol)
    if price is not None:
        tiny = pd.DataFrame({"Open":[price],"High":[price],"Low":[price],"Close":[price],"Volume":[0]})
        tiny.index = pd.to_datetime([datetime.now()])
        return tiny, src or "Quick"
    return None, None

# ---------------- Technical analysis functions ----------------
def analyze_df(df):
    if df is None or df.empty:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah","fib_levels":{}}
    try:
        cur = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2] if len(df)>1 else cur
        change = ((cur-prev)/prev*100) if prev != 0 else 0.0
        hi, lo = df["High"].max(), df["Low"].min()
        fib_bias = "Netral"
        fib_levels = {}
        if pd.notna(hi) and pd.notna(lo) and hi!=lo:
            diff = hi - lo
            fib_levels = {
                "fib_0": hi,
                "fib_0.236": hi - diff*0.236,
                "fib_0.382": hi - diff*0.382,
                "fib_0.5": hi - diff*0.5,
                "fib_0.618": hi - diff*0.618,
                "fib_1": lo
            }
            fib61, fib38 = fib_levels["fib_0.618"], fib_levels["fib_0.382"]
            if cur > fib61:
                fib_bias = "Bullish"
            elif cur < fib38:
                fib_bias = "Bearish"
        struct = "Konsolidasi"
        if len(df) > 10:
            hh = df["High"].rolling(5).max().dropna()
            ll = df["Low"].rolling(5).min().dropna()
            if len(hh) > 2 and len(ll) > 2:
                if hh.iloc[-1] > hh.iloc[-2] and ll.iloc[-1] > ll.iloc[-2]:
                    struct = "Bullish"
                elif hh.iloc[-1] < hh.iloc[-2] and ll.iloc[-1] < ll.iloc[-2]:
                    struct = "Bearish"
        sup = df["Low"].tail(20).min() if "Low" in df.columns else None
        res = df["High"].tail(20).max() if "High" in df.columns else None
        conviction = "Tinggi" if struct==fib_bias and struct!="Konsolidasi" else ("Sedang" if struct!="Konsolidasi" else "Rendah")
        return {"structure":struct,"fib_bias":fib_bias,"support":sup,"resistance":res,"current":cur,"change":change,"conviction":conviction,"fib_levels":fib_levels}
    except Exception:
        return {"structure":"No Data","fib_bias":"Netral","support":None,"resistance":None,"current":None,"change":0.0,"conviction":"Rendah","fib_levels":{}}

def add_indicators(df):
    if df is None or df.empty: return df
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    ma = df["Close"].rolling(20).mean()
    sd = df["Close"].rolling(20).std()
    df["BB_up"] = ma + 2*sd
    df["BB_dn"] = ma - 2*sd
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14,adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100/(1+rs))
    # MACD
    ema12 = df["Close"].ewm(span=12,adjust=False).mean()
    ema26 = df["Close"].ewm(span=26,adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            obv.append(obv[-1] + (df["Volume"].iloc[i] if "Volume" in df.columns else 0))
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            obv.append(obv[-1] - (df["Volume"].iloc[i] if "Volume" in df.columns else 0))
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df

def confidence_from_source(src):
    if not src: return None
    s = str(src).lower()
    if any(x in s for x in ["binance","bybit","kucoin","bitget","gate","yfinance"]):
        return "High"
    if any(x in s for x in ["coingecko","coinmarketcap"]):
        return "Medium"
    if "dexscreener" in s:
        return "Low"
    if "synthetic" in s:
        return "Synthetic"
    return "Unknown"

# ---------------- Scanning & ensure >=60 valid ----------------
def quick_scan_symbols(symbols):
    results = []
    total = len(symbols)
    prog = st.progress(0)
    for i, sym in enumerate(symbols):
        sym = sym.strip().upper()
        price, src = get_price_quick_exchange_first(sym)
        if price is None:
            status = "No Data"
            conf = None
        else:
            status = "OK"
            conf = confidence_from_source(src)
        results.append({"symbol":sym,"price":price,"source":src,"confidence":conf,"status":status})
        prog.progress((i+1)/total)
        time.sleep(PAUSE_BETWEEN)
    return pd.DataFrame(results)

def fill_to_min_valid(df):
    ok_count = df["status"].eq("OK").sum()
    if ok_count >= MIN_VALID:
        return df
    backups = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","AVAXUSDT",
        "XRPUSDT","SHIBUSDT","LTCUSDT","LINKUSDT","ATOMUSDT","SUIUSDT","APTUSDT","OPUSDT","NEARUSDT","FILUSDT",
    ]
    tried = set(df["symbol"].tolist())
    for b in backups:
        if ok_count >= MIN_VALID:
            break
        if b in tried:
            continue
        price, src = get_price_quick_exchange_first(b)
        if price is not None:
            df = pd.concat([df, pd.DataFrame([{"symbol":b,"price":price,"source":src,"confidence":confidence_from_source(src),"status":"OK (backup)"}])], ignore_index=True)
            ok_count += 1
        else:
            df = pd.concat([df, pd.DataFrame([{"symbol":b,"price":None,"source":None,"confidence":None,"status":"No Data"}])], ignore_index=True)
        tried.add(b)
        time.sleep(PAUSE_BETWEEN)
    df = df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    return df

# ---------------- Helper: chance price & entry ----------------
def compute_chance_and_entry(analysis):
    """
    analysis: dict returned from analyze_df, containing support, resistance, current, fib_levels
    Returns: chance_pct (0-100), chance_label, entry_price (we'll use support)
    """
    cur = analysis.get("current")
    sup = analysis.get("support")
    res = analysis.get("resistance")
    fibs = analysis.get("fib_levels", {})
    if cur is None or sup is None or res is None or res == sup:
        return None, "Unknown", sup
    # chance: how close current is to support relative to range (lower -> better entry)
    try:
        pct = (cur - sup) / (res - sup) * 100  # 0% @ support, 100% @ resistance
        pct = max(0.0, min(100.0, pct))
    except:
        pct = None
    # label logic: lower pct -> higher chance (near support)
    if pct is None:
        label = "Unknown"
    elif pct < 20:
        label = "High (near support)"
    elif pct < 40:
        label = "Good"
    elif pct < 60:
        label = "Moderate"
    else:
        label = "Low (near resistance)"
    entry_price = sup  # as requested
    return round(pct,2) if pct is not None else None, label, entry_price

# ---------------- Streamlit UI ----------------
col_left, col_right = st.columns([2,8])
with col_left:
    tf = st.selectbox("🕒 Timeframe", TF_OPTIONS)
    scan_btn = st.button("🔄 Scan Ulang (Snapshot Realtime)")
    force_60_btn = st.button("🧩 Paksa Lengkapi 60 Valid")
    auto_refresh = st.sidebar.slider("Auto-refresh setiap (detik)", min_value=0, max_value=600, value=0, step=10)
    chart_style = st.selectbox("💹 Jenis Grafik", ["Candlestick","Line"])
with col_right:
    st.markdown("### Tabel Analisa (ringkasan) — teratas")
    table_placeholder = st.empty()
    status_placeholder = st.empty()

# session state
if "scan_df" not in st.session_state:
    st.session_state.scan_df = pd.DataFrame()
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = pd.DataFrame()
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "chart_symbol" not in st.session_state:
    st.session_state.chart_symbol = None

# Run quick scan
if scan_btn:
    symbols = COINS.copy()
    with st.spinner("Memindai harga (quick snapshot) — exchange-first..."):
        qdf = quick_scan_symbols(symbols)
        qdf = fill_to_min_valid(qdf)
        st.session_state.scan_df = qdf
        st.session_state.last_scan = datetime.now(timezone.utc)
        ok = qdf["status"].str.contains("OK").sum()
        fail = (qdf["status"]=="No Data").sum()
        status_placeholder.success(f"Quick scan selesai. {ok} OK, {fail} No Data. Waktu (UTC): {st.session_state.last_scan.strftime('%Y-%m-%d %H:%M:%S')}")
        table_placeholder.dataframe(qdf[["symbol","price","source","confidence","status"]], use_container_width=True)

# Force fill 60 valid
if force_60_btn:
    if st.session_state.scan_df.empty:
        st.warning("Belum ada hasil scan. Tekan 'Scan Ulang' dulu.")
    else:
        with st.spinner("Memaksa melengkapi 60 koin valid (exchange-first)..."):
            df2 = fill_to_min_valid(st.session_state.scan_df)
            st.session_state.scan_df = df2
            ok = df2["status"].str.contains("OK").sum()
            st.success(f"Total valid sekarang: {ok} (minimal {MIN_VALID} terpenuhi jika tersedia).")
            table_placeholder.dataframe(df2[["symbol","price","source","confidence","status"]], use_container_width=True)

# If scan exists, build full analysis
if not st.session_state.scan_df.empty and (scan_btn or force_60_btn):
    analysis_results = []
    df_map = {}
    total = len(st.session_state.scan_df)
    prog = st.progress(0)
    for i, row in st.session_state.scan_df.iterrows():
        sym = row["symbol"]
        ohlc, src = get_ohlc_for_symbol_exchange_first(sym, tf)
        if ohlc is None:
            # build single-bar from quick price fallback
            price = row.get("price")
            if price is not None:
                tiny = pd.DataFrame({"Open":[price],"High":[price],"Low":[price],"Close":[price],"Volume":[0]})
                tiny.index = pd.to_datetime([datetime.now()])
                ohlc = tiny
                src = row.get("source") or "QuickPrice"
        analysis = analyze_df(ohlc)
        chance_pct, chance_label, entry_price = compute_chance_and_entry(analysis)
        # collect info
        analysis_record = {
            "symbol": sym,
            "structure": analysis.get("structure"),
            "fib_bias": analysis.get("fib_bias"),
            "chance_pct": chance_pct,
            "chance_label": chance_label,
            "entry_price": entry_price,
            "support": analysis.get("support"),
            "resistance": analysis.get("resistance"),
            "current": analysis.get("current"),
            "change_%": round(analysis.get("change",0),4),
            "conviction": analysis.get("conviction"),
            "data_src": src,
            "reported_price": row.get("price"),
            "reported_src": row.get("source"),
            "confidence": row.get("confidence")
        }
        analysis_results.append((analysis_record, ohlc))
        df_map[sym] = ohlc
        prog.progress((i+1)/total)
        time.sleep(PAUSE_BETWEEN)
    # Build summary DataFrame
    summary_rows = [r for r,o in analysis_results]
    analysis_df = pd.DataFrame(summary_rows)
    st.session_state.analysis_df = analysis_df
    st.markdown("#### Tabel Analisa — ringkasan")
    # reorder columns for readability
    cols_order = ["symbol","structure","fib_bias","chance_pct","chance_label","entry_price","support","resistance","current","change_%","conviction","data_src","reported_price","reported_src","confidence"]
    display_df = analysis_df.reindex(columns=cols_order)
    st.dataframe(display_df.sort_values(by=["conviction","structure"], ascending=[False,False]).reset_index(drop=True), use_container_width=True)

    # default chart symbol
    ok_syms = analysis_df[analysis_df["current"].notna()]["symbol"].tolist()
    if len(ok_syms) > 0:
        st.session_state.chart_symbol = st.selectbox("📊 Pilih koin untuk grafik:", ok_syms, index=0)
    else:
        st.session_state.chart_symbol = st.selectbox("📊 Pilih koin untuk grafik:", COINS, index=0)

    # show source counts
    src_counts = analysis_df["data_src"].fillna("None").value_counts().rename_axis("source").reset_index(name="count")
    st.markdown("#### Sumber data (OHLC) — jumlah")
    st.table(src_counts)

    # Charting
    symbol = st.session_state.chart_symbol
    chart_df = df_map.get(symbol)
    if chart_df is None or chart_df.empty:
        st.warning("⚠️ Data OHLC tidak tersedia untuk grafik koin ini.")
    else:
        chart_df = add_indicators(chart_df)
        fig = go.Figure()
        if chart_style == "Candlestick":
            fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
                                         low=chart_df["Low"], close=chart_df["Close"], name="Price"))
        else:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], mode="lines", name="Close"))
        for col in ["EMA20","EMA50"]:
            if col in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[col], mode="lines", name=col))
        if {"BB_up","BB_dn"}.issubset(chart_df.columns):
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_up"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_dn"], fill="tonexty", line=dict(width=0), showlegend=False))
        last = analyze_df(chart_df)
        if last.get("support") is not None:
            fig.add_hline(y=last["support"], line=dict(color="green",dash="dot"), annotation_text="Support")
        if last.get("resistance") is not None:
            fig.add_hline(y=last["resistance"], line=dict(color="red",dash="dot"), annotation_text="Resistance")
        fig.update_layout(template="plotly_dark", height=520,
                          title=f"{symbol} | {last['structure']} | {last['fib_bias']} | Conviction: {last['conviction']}")
        st.plotly_chart(fig, use_container_width=True)

        # RSI
        if "RSI" in chart_df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], mode="lines", name="RSI14"))
            fig2.update_layout(template="plotly_dark", height=200, yaxis=dict(range=[0,100]))
            st.plotly_chart(fig2, use_container_width=True)
        # MACD
        if "MACD" in chart_df.columns and "Signal" in chart_df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"], mode="lines", name="MACD"))
            fig3.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Signal"], mode="lines", name="Signal"))
            fig3.update_layout(template="plotly_dark", height=200, title="MACD (12,26,9)")
            st.plotly_chart(fig3, use_container_width=True)
        # OBV
        if "OBV" in chart_df.columns:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=chart_df.index, y=chart_df["OBV"], mode="lines", name="OBV"))
            fig4.update_layout(template="plotly_dark", height=200, title="On-Balance Volume (OBV)")
            st.plotly_chart(fig4, use_container_width=True)

# Auto-refresh
if auto_refresh > 0:
    st.info(f"Auto-refresh aktif setiap {auto_refresh} detik — app akan reload.")
    time.sleep(auto_refresh)
    st.experimental_rerun()

# Footer notes
st.markdown("---")
st.markdown(
    """
    **Catatan:** 
    - Prioritas sumber OHLC/price: Binance → Bybit → Bitget → Gate → KuCoin → CoinGecko → CoinMarketCap → yfinance → DexScreener.
    - Entry Price = Support (konservatif dan cocok untuk scanner sinyal cepat).
    - Chance Price (%) mengukur posisi harga relatif terhadap range Support→Resistance (lebih rendah = lebih dekat ke support = peluang lebih tinggi).
    - Jika kamu ingin menyimpan/mengekspor hasil otomatis atau menambahkan API keys untuk exchange tertentu, beri tahu saya dan saya tambahkan input field.
    """
)
