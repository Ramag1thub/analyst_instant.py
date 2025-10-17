# File: analyst_instant.py
# Versi: 15.4 - Resource Koins Kuat (200+ Koin, Termasuk Meme)
# Tujuan: Menambah resource koin secara signifikan untuk meningkatkan hasil analisis.

import streamlit as st
import pandas as pd
import ccxt.pro as ccxt  # ccxt sekarang adalah alias untuk ccxt.pro
import asyncio
import time
import numpy as np

# --- KONSTANTA OPTIMASI GLOBAL ---
ASYNC_BATCH_SIZE = 20 # Membatasi koneksi serentak untuk menghindari 'too many file descriptors'

# --- DAFTAR KOIN DASAR (DIPERKUAT MENJADI 200+ SIMBOL) ---
BASE_COIN_UNIVERSE = [
    # --- BLOCKCHAIN UTAMA & DEFI CORE --- (Sekitar 200+ total)
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'BNB/USDT', 'ADA/USDT',
    'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT', 'SHIB/USDT', 'TRX/USDT', 'BCH/USDT',
    'LTC/USDT', 'NEAR/USDT', 'UNI/USDT', 'ICP/USDT', 'PEPE/USDT', 'TON/USDT', 'KAS/USDT',
    'INJ/USDT', 'RNDR/USDT', 'TIA/USDT', 'FET/USDT', 'WIF/USDT', 'ARB/USDT', 'OP/USDT',
    'ETC/USDT', 'XLM/USDT', 'FIL/USDT', 'IMX/USDT', 'APT/USDT', 'FTM/USDT', 'SAND/USDT', 
    'MANA/USDT', 'GRT/USDT', 'AAVE/USDT', 'ATOM/USDT', 'ZIL/USDT', 'ALGO/USDT', 'EGLD/USDT', 
    'SUI/USDT', 'SEI/USDT', 'PYTH/USDT', 'GMT/USDT', 'ID/USDT', 'KNC/USDT', 'WLD/USDT', 
    'MINA/USDT', 'DYDX/USDT', 'GALA/USDT', 'LDO/USDT', 'BTT/USDT', 'VET/USDT', 'OCEAN/USDT', 
    'ROSE/USDT', 'EOS/USDT', 'FLOW/USDT', 'THETA/USDT', 'AXS/USDT', 'ENJ/USDT', 'CRV/USDT', 
    'GMX/USDT', 'COMP/USDT', 'YFI/USDT', 'SNX/USDT', 'MKR/USDT', 'FXS/USDT', 'RUNE/USDT', 
    'ZEC/USDT', 'BAT/USDT', '1INCH/USDT', 'CELO/USDT', 'ZRX/USDT', 'ONT/USDT', 'DASH/USDT', 
    'CVC/USDT', 'NEO/USDT', 'QTUM/USDT', 'ICX/USDT', 'WAVES/USDT', 'DCR/USDT', 'OMG/USDT', 
    'BAND/USDT', 'KSM/USDT', 'SRM/USDT', 'CHZ/USDT', 'HBAR/USDT', 'IOTX/USDT', 'ZEN/USDT',
    'PERP/USDT', 'RLC/USDT', 'CTXC/USDT', 'BAKE/USDT', 'KAVA/USDT', 'CELR/USDT', 'RVN/USDT', 
    'TLM/USDT', 'TFUEL/USDT', 'STX/USDT', 'JASMY/USDT', 'GLMR/USDT', 'MASK/USDT', 'DODO/USDT', 
    'ASTR/USDT', 'ACH/USDT', 'AGIX/USDT', 'SPELL/USDT', 'WOO/USDT', 'VELO/USDT',
    
    # --- TAMBAHAN MEME COINS (High Volatility) ---
    'FLOKI/USDT', 'WIF/USDT', 'BONK/USDT', '1000PEPE/USDT', 'MEME/USDT', 
    'PENGU/USDT', 'TRUMP/USDT', 'FART/USDT', 'TOSHI/USDT', 'BOME/USDT',
    
    # --- TAMBAHAN AI & DEPIN ---
    'ARKM/USDT', 'TAO/USDT', 'AKT/USDT', 'NMT/USDT', 'OLAS/USDT', 'IOTX/USDT', 'CQT/USDT', 
    'PHB/USDT', 'OPSEC/USDT', 'ROSE/USDT', 'GLM/USDT', 'FET/USDT', 'AGIX/USDT', 'OCEAN/USDT',

    # --- TAMBAHAN L1/L2 & EKOSISTEM BARU ---
    'SEILOR/USDT', 'METIS/USDT', 'STRK/USDT', 'ZETA/USDT', 'ALT/USDT', 'LSK/USDT', 'BEAM/USDT',
    'AEVO/USDT', 'CKB/USDT', 'FXS/USDT', 'SSV/USDT', 'MAVIA/USDT', 'JTO/USDT', 'BLUR/USDT',
    'SC/USDT', 'CFX/USDT', 'FLR/USDT', 'MOVR/USDT', 'GNS/USDT', 'HIGH/USDT', 'KAS/USDT',
    'MINA/USDT', 'CELR/USDT', 'MAGIC/USDT', 'RDNT/USDT', 'LEVER/USDT', 'CTSI/USDT', 'VRA/USDT',
    'POLS/USDT', 'XEC/USDT', 'KLAY/USDT', 'WEMIX/USDT', 'API3/USDT', 'BEL/USDT', 'CHESS/USDT',
    'SKL/USDT', 'STMX/USDT', 'ONG/USDT', 'ARPA/USDT', 'BAKE/USDT', 'HFT/USDT', 'PENDLE/USDT', 
    'GTC/USDT', 'EDU/USDT', 'TLM/USDT', 'YGG/USDT', 'LINA/USDT', 'SSV/USDT', 'MC/USDT', 
    'C98/USDT', 'ZRO/USDT', 'PYTH/USDT', 'AITECH/USDT', 'PRIME/USDT', 'MANTLE/USDT',
    
    # --- KOIN KAPITALISASI KECIL LAINNYA ---
    'GAL/USDT', 'NMR/USDT', 'SFP/USDT', 'TOMO/USDT', 'SYS/USDT', 'TLM/USDT', 'VRA/USDT',
    'WAXP/USDT', 'PHA/USDT', 'ALICE/USDT', 'DUSK/USDT', 'ILV/USDT', 'RSR/USDT', 'T/USDT',
    'RIF/USDT', 'BADGER/USDT', 'KP3R/USDT', 'DAR/USDT', 'CPOOL/USDT', 'AUCTION/USDT',
    'ZEN/USDT', 'ZKS/USDT', 'XVS/USDT', 'NKN/USDT', 'MDT/USDT', 'PROS/USDT', 'TRU/USDT',
    'REI/USDT', 'DATA/USDT', 'KEY/USDT', 'LOOM/USDT', 'HIFI/USDT', 'LRC/USDT', 'ZBC/USDT',
    'HYPE/USDT', 'ASTER/USDT', 'USELESS/USDT', 'LAUNCHCOIN/USDT', 'PERP/USDT', 
]


# --- PENGATURAN HALAMAN & KONFIGURASI AWAL ---
st.set_page_config(layout="wide", page_title="Instant AI Analyst", initial_sidebar_state="collapsed")

# --- UI/UX CSS ---
INSTANT_CSS = """
<style>
    .stApp { background-color: #151515; color: #d1d1d1; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 14px; }
    .block-container { padding: 2rem 1.5rem 1rem 1.5rem !important; }
    .st-emotion-cache-16txtl3 { padding-top: 0rem; }
    .signal-card { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #2a2a2a; padding: 1rem; margin-bottom: 1rem; }
    footer { visibility: hidden; }
    .profit { color: #50fa7b; font-weight: bold; } 
    .loss { color: #ff6e6e; font-weight: bold; }
    .entry { color: #8be9fd; font-weight: bold; }
    .new-coin { border: 2px solid #ffaa00; background-color: #2a2215; } 
    .highlight-pct { font-size: 1.1em; color: #ffaa00; font-weight: bold; }
</style>
"""
st.markdown(INSTANT_CSS, unsafe_allow_html=True)

# --- FUNGSI UTILITY: DISCOVERY KOIN BARU ---

async def get_new_perpetual_symbols(current_coins):
    """Membandingkan daftar koin yang sudah ada dengan pasar perpetual terbaru dari 5 exchange teratas."""
    
    EXCHANGES = ['okx', 'bybit', 'gateio', 'binanceusdm', 'mexc'] 
    all_new_symbols = set()
    current_set = set(current_coins)

    async def fetch_markets(exchange_id):
        try:
            exchange = getattr(ccxt, exchange_id)({
                'options': {'defaultType': 'future'},
                'timeout': 10000 
            })
            markets = await exchange.load_markets()
            await exchange.close()
            perpetual_symbols = {
                symbol for symbol, market in markets.items() 
                if 'USDT' in symbol and market['active'] and market['type'] in ('swap', 'future')
            }
            return perpetual_symbols
        except Exception:
            return set()

    tasks = [fetch_markets(eid) for eid in EXCHANGES]
    results = await asyncio.gather(*tasks)
    
    for perpetual_symbols in results:
        all_new_symbols.update(perpetual_symbols)

    newly_launched = list(all_new_symbols - current_set)
    newly_launched = sorted([s for s in newly_launched if s.endswith('/USDT')])
    
    return newly_launched

# --- FUNGSI-FUNGSI BACKEND (MURNI ASINKRON) ---

async def fetch_daily_data(symbol, days=365):
    """Mengambil data harian asinkron dari Bybit (Sumber Utama)."""
    
    exchange = ccxt.bybit({
        'options': {'defaultType': 'future'}
    }) 
    
    df = None
    try:
        since = exchange.milliseconds() - 86400000 * days
        bars = await exchange.fetch_ohlcv(symbol, '1d', since=since)
        if bars:
            df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
    except Exception as e:
        pass 
    finally:
        await exchange.close()
    return df

@st.cache_data(show_spinner=False)
def analyze_structure(df):
    """Menggunakan @st.cache_data untuk meng-cache hasil analisis SR/Fib/Struktur."""
    if df is None or len(df) < 20: 
        return {'structure': 'Data Tidak Cukup', 'current_price': None, 'fib_bias': 'Netral'}

    structure = "Konsolidasi"
    current_close = df['Close'].iloc[-1]
    df_recent = df[['High', 'Low']].tail(14)

    # Logika Market Structure (Cepat)
    if len(df_recent) >= 14:
        recent_highs = df_recent['High'].rolling(window=5, center=True).max().dropna()
        recent_lows = df_recent['Low'].rolling(window=5, center=True).min().dropna()
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            high_h = recent_highs.iloc[-1] > recent_highs.iloc[-2]
            low_h = recent_lows.iloc[-1] > recent_lows.iloc[-2]
            if high_h and low_h: structure = "Bullish"
            elif recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]: structure = "Bearish"

    # Logika Fibonacci (Cepat)
    max_price = df['High'].max(); min_price = df['Low'].min(); diff = max_price - min_price
    fib_level = {}
    for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
        fib_level[f'FIB_{int(level*100)}'] = max_price - (diff * level)
        
    fib_bias = "Netral"
    if 'FIB_61' in fib_level and current_close > fib_level['FIB_61']: fib_bias = "Bullish"
    elif 'FIB_38' in fib_level and current_close < fib_level['FIB_38']: fib_bias = "Bearish"
        
    return {'structure': structure, 'current_price': current_close, 'fib_bias': fib_bias}

async def find_signal_resampled(symbol, user_timeframe):
    """Mengambil data dan menganalisis secara lokal."""
    
    df_daily = await fetch_daily_data(symbol) 
    
    if df_daily is None: return None

    resample_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    if len(df_daily) < 7: return None

    try:
        analysis_d = analyze_structure(df_daily)
        df_w = df_daily.resample('W').agg(resample_rules).dropna()
        analysis_w = analyze_structure(df_w)
        
        if user_timeframe == '1d':
            analysis_user = analysis_d
        else:
            df_user = df_daily.resample(user_timeframe).agg(resample_rules).dropna()
            analysis_user = analyze_structure(df_user)
            
    except Exception:
        return None

    # Logika Konviksi Ditingkatkan
    structure_w = analysis_w['structure']; structure_d = analysis_d['structure']; structure_user = analysis_user['structure']
    fib_bias_d = analysis_d['fib_bias']; current_price = analysis_d['current_price']
    
    conviction = "Rendah"; bias = "Netral"
    
    # Inisialisasi Trading Parameter
    entry_price = current_price
    sl_pct = 0; tp1_pct = 0; tp2_pct = 0
    
    # --- Kriteria Kuat (Conviction TINGGI) - RR 2:1 (Ketat) ---
    if structure_w == 'Bullish' and structure_d == 'Bullish' and structure_user == 'Bullish' and fib_bias_d == 'Bullish': 
        conviction = "Tinggi"; bias = "Bullish Kuat"
        tp1_pct = 5; tp2_pct = 10; sl_pct = 2.5 
    elif structure_w == 'Bearish' and structure_d == 'Bearish' and structure_user == 'Bearish' and fib_bias_d == 'Bearish': 
        conviction = "Tinggi"; bias = "Bearish Kuat"
        tp1_pct = -5; tp2_pct = -10; sl_pct = -2.5 
    
    # --- Kriteria Sedang (Conviction SEDANG) - RR Fleksibel (3:1) ---
    elif structure_d == 'Bullish' and structure_user == 'Bullish': 
        conviction = "Sedang"; bias = "Cenderung Bullish"
        tp1_pct = 3; tp2_pct = 7; sl_pct = 1.0 
    elif structure_d == 'Bearish' and structure_user == 'Bearish': 
        conviction = "Sedang"; bias = "Cenderung Bearish"
        tp1_pct = -3; tp2_pct = -7; sl_pct = -1.0 

    # Jika sinyal ditemukan
    if conviction in ['Tinggi', 'Sedang']:
        # Hitung Nilai SL dan TP berdasarkan Persentase
        sl_multiplier = 1 + (sl_pct / 100) if sl_pct >= 0 else 1 - abs(sl_pct / 100)
        sl_target = current_price * sl_multiplier
        tp1_target = current_price * (1 + (tp1_pct / 100))
        tp2_target = current_price * (1 + (tp2_pct / 100))
        
        # Hitung RR Ratio (absolut)
        risk_amount = abs(current_price * (abs(sl_pct) / 100))
        reward_amount = abs(current_price * (abs(tp1_pct) / 100))
        
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'symbol': symbol, 'bias': bias, 'timeframe': user_timeframe, 'conviction': conviction,
            'entry': entry_price, 
            'sl_target': sl_target,
            'tp1_target': tp1_target, 'tp2_target': tp2_target,
            't1_pct': tp1_pct, 't2_pct': tp2_pct, 'sl_pct': sl_pct,
            'rr_ratio': rr_ratio
        }
    return None

async def run_scanner_streamed(coin_universe, timeframe, placeholder):
    """Menjalankan pemindaian dengan BATCHING asinkron."""
    
    found_trades = []
    
    for i in range(0, len(coin_universe), ASYNC_BATCH_SIZE):
        batch = coin_universe[i:i + ASYNC_BATCH_SIZE]
        tasks = [find_signal_resampled(symbol, timeframe) for symbol in batch]
        
        batch_results = await asyncio.gather(*tasks)
        
        for result in batch_results:
            if result:
                found_trades.append(result)
                placeholder.success(f"✅ Ditemukan {len(found_trades)} sinyal... Memproses batch {i//ASYNC_BATCH_SIZE + 1} dari {len(coin_universe)//ASYNC_BATCH_SIZE + 1}.")

    placeholder.success(f"✅ Pemindaian data selesai. Total {len(found_trades)} sinyal ditemukan.")
    return found_trades

# --- ANTARMUKA APLIKASI WEB ---
st.title("🚀 Instant AI Signal Dashboard")
st.caption(f"Menganalisis {len(BASE_COIN_UNIVERSE)}+ koin secara paralel dengan **Batching ({ASYNC_BATCH_SIZE} koin/batch)**.")

col1, col2, col3 = st.columns([1.5, 1.5, 7])
selected_tf = col1.selectbox("Pilih Timeframe Sinyal:", ['1d', '4h', '1h'], help="Pilih Timeframe sinyal yang diinginkan.")
if col2.button("🔄 Scan Ulang Sekarang"):
    if 'new_symbols' in st.session_state:
        del st.session_state['new_symbols']
    st.cache_data.clear() 
    st.rerun() 
st.markdown("---")

# --- INISIALISASI VARIABEL RUNTIME ---
status_placeholder = st.empty() 
start_time = time.time() 

# --- PROSES DISCOVERY DAN MERGE ---
start_discovery = time.time()
status_placeholder.info(f"Memuat pasar perpetual terbaru dari 5 exchange teratas...") 

if 'new_symbols' not in st.session_state:
    new_symbols = asyncio.run(get_new_perpetual_symbols(BASE_COIN_UNIVERSE))
    st.session_state['new_symbols'] = new_symbols
else:
    new_symbols = st.session_state['new_symbols']

COIN_UNIVERSE = list(set(BASE_COIN_UNIVERSE + new_symbols))
COIN_UNIVERSE.sort() 

discovery_time = time.time() - start_discovery
st.sidebar.markdown(f"**⏰ Discovery Time:** {discovery_time:.2f} detik")
st.sidebar.markdown(f"**✨ Koin Baru Ditemukan:** {len(new_symbols)}")

# --- PROSES PEMINDAIAN UTAMA ---

start_scan_time = time.time()
status_placeholder.info(f"Memulai pemindaian instan untuk **{len(COIN_UNIVERSE)} koin**...")

# Panggilan utama ASINKRON
found_trades = asyncio.run(run_scanner_streamed(COIN_UNIVERSE, selected_tf, status_placeholder))
total_time = time.time() - start_scan_time

# --- TAMPILKAN HASIL AKHIR ---
status_placeholder.empty()

# Hitung Persentase Sinyal
total_coins_scanned = len(COIN_UNIVERSE)
total_signals = len(found_trades)
signal_percentage = (total_signals / total_coins_scanned) * 100 if total_coins_scanned > 0 else 0

if not found_trades:
    st.success(f"✅ Pemindaian selesai dalam **{total_time:.2f} detik**. Tidak ditemukan sinyal kuat saat ini. (0.00% pasar aktif)")
else:
    st.success(f"""
        ✅ Pemindaian selesai dalam **{total_time:.2f} detik**. 
        Ditemukan {total_signals} sinyal potensial dari {total_coins_scanned} koin.
        **Status Pasar (Sinyal Sedang/Tinggi):** <span class='highlight-pct'>{signal_percentage:.2f}%</span> dari koin yang dipindai.
    """, unsafe_allow_html=True)
    
    # 1. Pisahkan dan Urutkan sinyal berdasarkan Conviction (Tinggi > Sedang)
    high_conviction = [t for t in found_trades if t['conviction'] == 'Tinggi']
    medium_conviction = [t for t in found_trades if t['conviction'] == 'Sedang']
    
    # Gabungkan dan Urutkan (Tinggi di atas Sedang, lalu abjad)
    all_signals = sorted(high_conviction, key=lambda x: x['symbol']) + sorted(medium_conviction, key=lambda x: x['symbol'])

    # 2. Ambil 3 Koin Teratas
    top_3_signals = all_signals[:3]
    
    # Fungsi Pembantu untuk Format Harga
    def format_price(price):
        if price is None: return "N/A"
        if price < 0.001: return f"{price:,.8f}"
        if price < 0.1: return f"{price:,.5f}"
        if price < 10: return f"{price:,.4f}"
        return f"{price:,.2f}"
    
    # Fungsi Pembantu untuk Format RR Ratio
    def format_rr(ratio):
        return f"{ratio:.1f}:1" if ratio > 0 else "0:1"

    # --- TAMPILAN TOP 3 KOIN ---
    if top_3_signals:
        st.header("🏆 Top 3 Sinyal Kuat (Rekomendasi Trading)")
        cols_top = st.columns(3)
        for i, trade in enumerate(top_3_signals):
            with cols_top[i]:
                with st.container(border=True):
                    st.markdown(f"**#{i+1}: {trade['symbol']}**", unsafe_allow_html=True) 
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    is_bullish = "Bullish" in trade['bias']

                    st.markdown(f"**Konviksi:** <strong style='color:{color};'>{trade['conviction']} ({trade['bias']})</strong>", unsafe_allow_html=True)
                    st.caption(f"Timeframe: {trade['timeframe']}")
                    
                    # REKOMENDASI ENTRI
                    st.markdown(f"**Entri:** <span class='entry'>{format_price(trade['entry'])}</span>", unsafe_allow_html=True)

                    # REKOMENDASI SL & TP
                    sl_class = "loss"
                    t1_class = "profit"
                    t2_class = "profit"
                    
                    st.markdown(f"**SL ({abs(trade['sl_pct'])}%):** <span class='{sl_class}'>{format_price(trade['sl_target'])}</span>", unsafe_allow_html=True)
                    st.markdown(f"**TP 1 ({abs(trade['t1_pct'])}%):** <span class='{t1_class}'>{format_price(trade['tp1_target'])}</span>", unsafe_allow_html=True)
                    st.markdown(f"**TP 2 ({abs(trade['t2_pct'])}%):** <span class='{t2_class}'>{format_price(trade['tp2_target'])}</span>", unsafe_allow_html=True)
                    
                    # REKOMENDASI RR
                    st.markdown(f"**R/R Ratio (TP1):** **{format_rr(trade['rr_ratio'])}**", unsafe_allow_html=True)


    st.markdown("---") # Garis pemisah antara Top 3 dan Daftar Lengkap
    
    # 3. Tampilkan Sinyal Keyakinan TINGGI Lainnya (jika ada)
    remaining_high_conviction = high_conviction[3:] if len(high_conviction) >= 3 else high_conviction
    
    if remaining_high_conviction:
        st.subheader("🔥 Sinyal Keyakinan TINGGI Lainnya"); num_cols = 4; cols = st.columns(num_cols)
        for i, trade in enumerate(remaining_high_conviction):
            with cols[i % num_cols]:
                with st.container(border=True): 
                    st.subheader(f"{trade['symbol']} {'(BARU)' if trade['symbol'] in new_symbols else ''}")
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    is_bullish = "Bullish" in trade['bias']
                    
                    st.markdown(f"**Sinyal:** <strong style='color:{color};'>{trade['bias']}</strong>", unsafe_allow_html=True)
                    st.caption(f"Harga Masuk: {format_price(trade['entry'])}")
                    
                    st.markdown(f"SL ({abs(trade['sl_pct'])}%): {format_price(trade['sl_target'])}", unsafe_allow_html=True)
                    st.markdown(f"TP 1 ({abs(trade['t1_pct'])}%): {format_price(trade['tp1_target'])}", unsafe_allow_html=True)
                    st.markdown(f"RR: {format_rr(trade['rr_ratio'])}", unsafe_allow_html=True)
                    
    # 4. Tampilkan Sinyal Keyakinan SEDANG
    if medium_conviction:
        st.subheader("👍 Sinyal Keyakinan SEDANG (RR 3:1 Fleksibel)"); num_cols = 4; cols = st.columns(num_cols)
        for i, trade in enumerate(medium_conviction):
            with cols[i % num_cols]:
                with st.container(border=True):
                    st.subheader(f"{trade['symbol']} {'(BARU)' if trade['symbol'] in new_symbols else ''}")
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    is_bullish = "Bullish" in trade['bias']
                    
                    st.markdown(f"**Sinyal:** <strong style='color:{color};'>{trade['bias']}</strong>", unsafe_allow_html=True)
                    st.caption(f"Harga Masuk: {format_price(trade['entry'])}")
                    
                    st.markdown(f"SL ({abs(trade['sl_pct'])}%): {format_price(trade['sl_target'])}", unsafe_allow_html=True)
                    st.markdown(f"TP 1 ({abs(trade['t1_pct'])}%): {format_price(trade['tp1_target'])}", unsafe_allow_html=True)
                    st.markdown(f"RR: {format_rr(trade['rr_ratio'])}", unsafe_allow_html=True)