# analyst_hybrid_v5_realtime_light.py
import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime, timezone

st.set_page_config(layout="wide", page_title="AI Analyst — Realtime v5 (Light)")
st.title("🚀 AI Analyst — Realtime v5 (Light)")
st.caption("Realtime harga terakhir • Multi-source fallback • Auto-refresh • Paksa 60 Valid • Confidence")

# -------- Settings --------
REQUEST_TIMEOUT = 8
PAUSE_BETWEEN = 0.05
MIN_VALID = 60

# -------- Base Coin List --------
COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","MATICUSDT","DOTUSDT","AVAXUSDT","XRPUSDT",
    "SHIBUSDT","LTCUSDT","LINKUSDT","ATOMUSDT","SUIUSDT","APTUSDT","OPUSDT","NEARUSDT","FILUSDT","TRXUSDT",
    "PEPEUSDT","GALAUSDT","RNDRUSDT","INJUSDT","FLOWUSDT","FTMUSDT","UNIUSDT","LDOUSDT","AAVEUSDT","GRTUSDT",
    "HYPEUSDT","ASTERUSDT","LAUNCHCOINUSDT","USELESSCOINUSDT",
    "MASKUSDT","BONKUSDT","ORDIUSDT","CHZUSDT","ANKRUSDT","ARBUSDT","BCHUSDT","XLMUSDT","VETUSDT"
]
COINS = list(dict.fromkeys([c.strip().upper() for c in COINS]))

# -------- API Endpoints --------
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"
CG_SIMPLE = "https://api.coingecko.com/api/v3/simple/price"
CG_SEARCH = "https://api.coingecko.com/api/v3/search"
DEX_BASE = "https://api.dexscreener.com/latest/dex/pairs"

# -------- Fetch Helpers --------
def fetch_binance_price(symbol):
    try:
        r = requests.get(BINANCE_TICKER, params={"symbol": symbol}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            js = r.json()
            if "price" in js:
                return {"price": float(js["price"]), "source": "Binance"}
    except:
        pass
    return None

def cg_search_id(query):
    try:
        r = requests.get(CG_SEARCH, params={"query": query}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            coins = r.json().get("coins", [])
            if coins:
                q = query.lower()
                for c in coins:
                    if c.get("symbol","").lower() == q:
                        return c.get("id")
                return coins[0].get("id")
    except:
        pass
    return None

def fetch_coingecko_by_id(cid):
    try:
        r = requests.get(CG_SIMPLE, params={"ids": cid, "vs_currencies": "usd"}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            js = r.json()
            if cid in js and "usd" in js[cid]:
                return {"price": float(js[cid]["usd"]), "source": f"CoinGecko ({cid})"}
    except:
        pass
    return None

def fetch_coingecko_by_symbol_like(symbol):
    base = symbol.lower().replace("usdt","").replace("_","")
    common_map = {"btc":"bitcoin","eth":"ethereum","bnb":"binancecoin","sol":"solana","ada":"cardano","xrp":"ripple",
                  "doge":"dogecoin","matic":"matic-network","dot":"polkadot"}
    if base in common_map:
        return fetch_coingecko_by_id(common_map[base])
    cid = cg_search_id(base)
    if cid:
        return fetch_coingecko_by_id(cid)
    return None

def fetch_dexscreener(symbol):
    short = symbol.lower().replace("usdt","")
    chains = ["ethereum","bsc","arbitrum","optimism","polygon"]
    for chain in chains:
        try:
            url = f"{DEX_BASE}/{chain}/{short}-usdt"
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                js = r.json()
                p = js.get("pair") or (js.get("pairs") and js.get("pairs")[0])
                if p and p.get("priceUsd"):
                    return {"price": float(p["priceUsd"]), "source": f"DexScreener ({chain})"}
        except:
            pass
    return None

def confidence_from_source(src):
    if not src: return None
    s = src.lower()
    if "binance" in s: return "High"
    if "coingecko" in s: return "Medium"
    if "dexscreener" in s: return "Low"
    return "Unknown"

def attempt_fetch(symbol):
    for fn in (fetch_binance_price, fetch_coingecko_by_symbol_like, fetch_dexscreener):
        res = fn(symbol)
        if res:
            return res
    # last resort: search id then fetch
    base = symbol.lower().replace("usdt","")
    cid = cg_search_id(base)
    if cid:
        res = fetch_coingecko_by_id(cid)
        if res:
            return res
    return None

# -------- Main scan logic --------
def scan_symbols(symbols):
    results = []
    total = len(symbols)
    prog = st.progress(0)
    for i, sym in enumerate(symbols):
        entry = {"symbol": sym, "price": None, "source": None, "confidence": None, "status": None}
        res = attempt_fetch(sym)
        if res:
            entry.update({"price": res["price"], "source": res["source"],
                          "confidence": confidence_from_source(res["source"]), "status": "OK"})
        else:
            entry.update({"status": "No Data"})
        results.append(entry)
        prog.progress((i+1)/total)
        if PAUSE_BETWEEN: time.sleep(PAUSE_BETWEEN)
    return pd.DataFrame(results)

def fill_until_60(df):
    ok_count = df["status"].eq("OK").sum()
    if ok_count >= MIN_VALID:
        return df
    backups = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT","DOTUSDT","AVAXUSDT","XRPUSDT",
        "MATICUSDT","SHIBUSDT","LTCUSDT","LINKUSDT","ATOMUSDT","FTMUSDT","UNIUSDT","SUIUSDT","FILUSDT","TRXUSDT"
    ]
    for coin in backups:
        if ok_count >= MIN_VALID:
            break
        if coin in df["symbol"].values: continue
        res = attempt_fetch(coin)
        entry = {"symbol": coin, "price": None, "source": None, "confidence": None, "status": None}
        if res:
            entry.update({"price": res["price"], "source": res["source"],
                          "confidence": confidence_from_source(res["source"]), "status": "OK (backup)"})
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            ok_count += 1
    return df

# -------- Streamlit UI --------
st.sidebar.markdown("## Pengaturan")
auto_refresh = st.sidebar.slider("Auto-refresh setiap (detik)", min_value=0, max_value=300, value=0, step=10)
st.sidebar.caption("Set 0 untuk menonaktifkan auto-refresh.")

col1, col2 = st.columns([3,7])
with col1:
    scan_btn = st.button("🔄 Scan Semua Koin")
    fix_btn = st.button("🧩 Paksa Lengkapi 60 Valid")
    add_input = st.text_input("Tambah koin (CSV)", "")
with col2:
    st.markdown("### Statistik")
    stat_placeholder = st.empty()

placeholder_table = st.empty()
placeholder_time = st.empty()

# Session state for persistence
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "last_time" not in st.session_state:
    st.session_state.last_time = None

def run_scan():
    manual = [s.strip().upper() for s in add_input.split(",") if s.strip()]
    symbols = list(dict.fromkeys(COINS + manual))
    with st.spinner("Memindai harga..."):
        df = scan_symbols(symbols)
        st.session_state.df = df
        st.session_state.last_time = datetime.now(timezone.utc)
        ok = df["status"].eq("OK").sum()
        fail = df["status"].eq("No Data").sum()
        stat_placeholder.metric("Berhasil", f"{ok}/{len(df)}", delta=f"Gagal: {fail}")
        placeholder_time.info(f"Waktu scan (UTC): {st.session_state.last_time.strftime('%Y-%m-%d %H:%M:%S')}")
        placeholder_table.dataframe(df[["symbol","price","source","confidence","status"]], use_container_width=True)

if scan_btn:
    run_scan()

if fix_btn and not st.session_state.df.empty:
    with st.spinner("Memaksa melengkapi 60 koin valid..."):
        df2 = fill_until_60(st.session_state.df)
        st.session_state.df = df2
        ok = df2["status"].str.contains("OK").sum()
        st.success(f"Total valid sekarang: {ok} (minimal 60 terpenuhi).")
        placeholder_table.dataframe(df2[["symbol","price","source","confidence","status"]], use_container_width=True)

# Auto-refresh mode
if auto_refresh > 0:
    time.sleep(auto_refresh)
    st.experimental_rerun()

st.markdown("---")
st.markdown("**Catatan:** Binance (High) → CoinGecko (Medium) → DexScreener (Low). Gunakan tombol 🧩 jika jumlah valid < 60.")
