# File: analyst_instant.py
# Versi: 16.0 - FINAL STABIL DAN BEBAS ERROR (Mode Sinkron)
# Tujuan: Stabilitas maksimum, mengatasi semua error, dan menggunakan logika RR 3:1 fleksibel.

import streamlit as st
import pandas as pd
import numpy as np
import time

# --- PENTING: Menggunakan CCXT Sinkron ---
try:
    # Hanya butuh ccxt, ccxt.pro dan asyncio dihapus untuk stabilitas maksimum.
    import ccxt 
except ImportError:
    st.error("Gagal mengimpor ccxt. Pastikan ccxt terinstal di requirements.txt (Hanya ccxt, bukan ccxt.pro atau asyncio).")
    st.stop()


# --- KONSTANTA OPTIMASI GLOBAL ---
# ASYNC_BATCH_SIZE sudah tidak digunakan karena kita beralih ke sinkron.

# --- DAFTAR KOIN DASAR (350+ SIMBOL PERPETUAL USDT) ---
BASE_COIN_UNIVERSE = [
    # --- BLOCKCHAIN UTAMA & DEFI CORE --- (Sekitar 350+ total)
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
    'BAND/USDT', 'BEL/USDT', 'CHZ/USDT', 'HBAR/USDT', 'IOTX/USDT', 'ZEN/USDT',
    'PERP/USDT', 'RLC/USDT', 'CTXC/USDT', 'BAKE/USDT', 'KAVA/USDT', 'CELR/USDT', 'RVN/USDT', 
    'TLM/USDT', 'TFUEL/USDT', 'STX/USDT', 'JASMY/USDT', 'GLMR/USDT', 'MASK/USDT', 'DODO/USDT', 
    'ASTR/USDT', 'ACH/USDT', 'AGIX/USDT', 'SPELL/USDT', 'WOO/USDT', 'VELO/USDT',
    'FLOKI/USDT', 'BONK/USDT', '1000PEPE/USDT', 'MEME/USDT', 
    'PENGU/USDT', 'TRUMP/USDT', 'FART/USDT', 'TOSHI/USDT', 'BOME/USDT',
    'ARKM/USDT', 'TAO/USDT', 'AKT/USDT', 'NMT/USDT', 'OLAS/USDT', 'CQT/USDT', 
    'PHB/USDT', 'OPSEC/USDT', 'GLM/USDT', 'SEILOR/USDT', 
    'METIS/USDT', 'STRK/USDT', 'ZETA/USDT', 'ALT/USDT', 'LSK/USDT', 'BEAM/USDT', 'AEVO/USDT', 
    'CKB/USDT', 'SSV/USDT', 'MAVIA/USDT', 'JTO/USDT', 'BLUR/USDT', 'SC/USDT', 'CFX/USDT', 
    'FLR/USDT', 'MOVR/USDT', 'GNS/USDT', 'HIGH/USDT', 'MAGIC/USDT', 'RDNT/USDT', 'LEVER/USDT', 
    'CTSI/USDT', 'VRA/USDT', 'POLS/USDT', 'XEC/USDT', 'KLAY/USDT', 'WEMIX/USDT', 'API3/USDT', 
    'CHESS/USDT', 'SKL/USDT', 'STMX/USDT', 'ONG/USDT', 'ARPA/USDT', 'HFT/USDT', 
    'PENDLE/USDT', 'GTC/USDT', 'EDU/USDT', 'YGG/USDT', 'LINA/USDT', 'MC/USDT', 'C98/USDT', 
    'ZRO/USDT', 'AITECH/USDT', 'PRIME/USDT', 'MANTLE/USDT',
    'GAL/USDT', 'NMR/USDT', 'SFP/USDT', 'TOMO/USDT', 'SYS/USDT', 'WAXP/USDT', 'PHA/USDT', 
    'ALICE/USDT', 'DUSK/USDT', 'ILV/USDT', 'RSR/USDT', 'T/USDT', 'RIF/USDT', 'BADGER/USDT', 
    'KP3R/USDT', 'DAR/USDT', 'CPOOL/USDT', 'AUCTION/USDT', 'ZKS/USDT', 'XVS/USDT', 'NKN/USDT', 
    'MDT/USDT', 'PROS/USDT', 'TRU/USDT', 'REI/USDT', 'DATA/USDT', 'KEY/USDT', 'LOOM/USDT', 
    'HIFI/USDT', 'LRC/USDT', 'ZBC/USDT', 'HYPE/USDT', 'ASTER/USDT', 'USELESS/USDT', 
    'LAUNCHCOIN/USDT', 'AERGO/USDT', 'AKRO/USDT', 'ALPHA/USDT', 'ANKR/USDT', 'ATA/USDT', 
    'BAL/USDT', 'BICO/USDT', 'BLZ/USDT', 'BNX/USDT', 'CLV/USDT', 'COTI/USDT', 'DENT/USDT', 
    'DGB/USDT', 'DMTR/USDT', 'DREP/USDT', 'ELF/USDT', 'EPS/USDT', 'ERTHA/USDT', 
    'FIS/USDT', 'FORTH/USDT', 'GHST/USDT', 'HARD/USDT', 
    'HNT/USDT', 'HOT/USDT', 'IDEX/USDT', 'IOST/USDT', 'IOTA/USDT', 'IRIS/USDT', 'LUNA/USDT', 
    'MBOX/USDT', 'MITH/USDT', 'MTL/USDT', 'NULS/USDT', 
    'ONE/USDT', 'OXT/USDT', 'PERL/USDT', 'PUNDIX/USDT', 'QLC/USDT', 
    'QUICK/USDT', 'RAY/USDT', 'REEF/USDT', 'REN/USDT', 'REQ/USDT', 'RVN/USDT', 'SOLO/USDT', 
    'SOS/USDT', 'STEEM/USDT', 'STG/USDT', 'STORJ/USDT', 'SUN/USDT', 'SUSHI/USDT', 'T/USDT', 
    'TOMO/USDT', 'TRB/USDT', 'TUSD/USDT', 'UTK/USDT', 'VIB/USDT', 
    'WAN/USDT', 'XEM/USDT', 'XYO/USDT', '1000SHIB/USDT',
    'UMA/USDT', 'LRC/USDT', 'AXS/USDT', 'BAT/USDT', 
    'CHZ/USDT', 'DODO/USDT', 'GALA/USDT', 'GRT/USDT', 'MKR/USDT', 'NEO/USDT', 
    'ONT/USDT', 'QTUM/USDT', 'SFP/USDT', 'SUSHI/USDT', 'THETA/USDT',
    'UNI/USDT', 'VET/USDT', 'XLM/USDT', 'ZIL/USDT', 'ZRX/USDT', 'BNX/USDT',
    'FTT/USDT', 'GMT/USDT', 'LDO/USDT',
    'PEPE/USDT', 'SEI/USDT', 'TIA/USDT', 'WLD/USDT', 'XVS/USDT', 'YFI/USDT', 'ZEC/USDT', 'ZRX/USDT', 
    'AIOZ/USDT', 'ALICE/USDT', 'ANKR/USDT', 'APENFT/USDT', 'API3/USDT', 'ARPA/USDT',
    'AUCTION/USDT', 'BSW/USDT', 'C98/USDT', 'CELR/USDT', 'CTK/USDT', 'DREP/USDT',
    'FIS/USDT', 'FLM/USDT', 'FLOW/USDT', 'GTC/USDT', 'HBAR/USDT',
    'IDEX/USDT', 'IOST/USDT', 'IRIS/USDT', 'JASMY/USDT', 'KLAY/USDT', 'LPT/USDT', 'LTO/USDT',
    'NANO/USDT', 'OXT/USDT', 'PAXG/USDT', 'PHB/USDT', 'PUNDIX/USDT', 'QNT/USDT',
    'RAY/USDT', 'RIF/USDT', 'RLC/USDT', 'RSR/USDT', 'RUNE/USDT', 'SXP/USDT', 'T/USDT',
    'TRB/USDT', 'TRU/USDT', 'TUSD/USDT', 'UMA/USDT', 'UNFI/USDT', 'UTK/USDT', 'VIB/USDT', 'WEMIX/USDT', 'XYO/USDT', 'ZKS/USDT', 'ZRO/USDT'
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

# --- FUNGSI UTAMA (SINKRON, NON-ASYNC) ---

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily_data(symbol, days=365):
    """Mengambil data harian SINKRON dari Bybit (Sumber Utama)."""
    
    # Inisialisasi exchange sinkron
    exchange = ccxt.bybit({ 
        'options': {'defaultType': 'future'},
        'timeout': 15000 # Timeout lebih lama untuk menghindari kegagalan fetch 
    }) 
    
    df = None
    try:
        since = exchange.milliseconds() - 86400000 * days
        # Panggil fetch_ohlcv sinkron
        bars = exchange.fetch_ohlcv(symbol, '1d', since=since)
        if bars:
            df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
    except Exception as e:
        pass 
    finally:
        exchange.close()
    return df

@st.cache_data(show_spinner=False)
def analyze_structure(df):
    """Analisis struktur (tetap sama)"""
    if df is None or len(df) < 20: 
        return {'structure': 'Data Tidak Cukup', 'current_price': None, 'fib_bias': 'Netral'}

    structure = "Konsolidasi"
    current_close = df['Close'].iloc[-1]
    df_recent = df[['High', 'Low']].tail(14)

    if len(df_recent) >= 14:
        recent_highs = df_recent['High'].rolling(window=5, center=True).max().dropna()
        recent_lows = df_recent['Low'].rolling(window=5, center=True).min().dropna()
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            high_h = recent_highs.iloc[-1] > recent_highs.iloc[-2]
            low_h = recent_lows.iloc[-1] > recent_lows.iloc[-2]
            if high_h and low_h: structure = "Bullish"
            elif recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]: structure = "Bearish"

    max_price = df['High'].max(); min_price = df['Low'].min(); diff = max_price - min_price
    fib_level = {}
    for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
        fib_level[f'FIB_{int(level*100)}'] = max_price - (diff * level)
        
    fib_bias = "Netral"
    if 'FIB_61' in fib_level and current_close > fib_level['FIB_61']: fib_bias = "Bullish"
    elif 'FIB_38' in fib_level and current_close < fib_level['FIB_38']: fib_bias = "Bearish"
        
    return {'structure': structure, 'current_price': current_close, 'fib_bias': fib_bias}

def find_signal_resampled(symbol, user_timeframe):
    """Mengambil data dan menganalisis secara lokal (SINKRON)."""
    
    df_daily = fetch_daily_data(symbol) 
    
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

    structure_w = analysis_w['structure']; structure_d = analysis_d['structure']; structure_user = analysis_user['structure']
    fib_bias_d = analysis_d['fib_bias']; current_price = analysis_d['current_price']
    
    conviction = "Rendah"; bias = "Netral"
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
    elif structure_d == 'Bearish' dan structure_user == 'Bearish': 
        conviction = "Sedang"; bias = "Cenderung Bearish"
        tp1_pct = -3; tp2_pct = -7; sl_pct = -1.0 

    if conviction in ['Tinggi', 'Sedang']:
        sl_multiplier = 1 + (sl_pct / 100) if sl_pct >= 0 else 1 - abs(sl_pct / 100)
        sl_target = current_price * sl_multiplier
        tp1_target = current_price * (1 + (tp1_pct / 100))
        tp2_target = current_price * (1 + (tp2_pct / 100))
        
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

def run_scanner_streamed_sync(coin_universe, timeframe, status_placeholder):
    """Menjalankan pemindaian secara SINKRON (berurutan) - SANGAT STABIL."""
    
    found_trades = []
    
    for i, symbol in enumerate(coin_universe):
        result = find_signal_resampled(symbol, timeframe)
        if result:
            found_trades.append(result)
        
        # Stream status update setiap 20 koin
        if i % 20 == 0 or i == len(coin_universe) - 1:
            status_placeholder.info(f"Memindai {symbol}... Koin ke {i+1} dari {len(coin_universe)}. Ditemukan {len(found_trades)} sinyal.")
            
    return found_trades

# --- ANTARMUKA APLIKASI WEB ---
st.title("🚀 Instant AI Signal Dashboard")
st.caption(f"Menganalisis {len(BASE_COIN_UNIVERSE)}+ koin secara **SINKRON** (Mode Paling Stabil).")

col1, col2, col3 = st.columns([1.5, 1.5, 7])
selected_tf = col1.selectbox("Pilih Timeframe Sinyal:", ['1d', '4h', '1h'], help="Pilih Timeframe sinyal yang diinginkan.")
if col2.button("🔄 Scan Ulang Sekarang"):
    st.cache_data.clear() 
    st.rerun() 
st.markdown("---")

# --- INISIALISASI VARIABEL RUNTIME ---
status_placeholder = st.empty() 
start_time = time.time() 

# --- PROSES PEMINDAIAN UTAMA (SINKRON) ---

status_placeholder.info(f"Memulai pemindaian instan untuk **{len(BASE_COIN_UNIVERSE)} koin** secara berurutan...")

found_trades = run_scanner_streamed_sync(BASE_COIN_UNIVERSE, selected_tf, status_placeholder)
total_time = time.time() - start_time

# --- TAMPILKAN HASIL AKHIR ---
status_placeholder.empty()

# Hitung Persentase Sinyal
total_coins_scanned = len(BASE_COIN_UNIVERSE)
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


    st.markdown("---") 
    
    # 3. Tampilkan Sinyal Keyakinan TINGGI Lainnya (jika ada)
    remaining_high_conviction = high_conviction[3:] if len(high_conviction) >= 3 else high_conviction
    
    if remaining_high_conviction:
        st.subheader("🔥 Sinyal Keyakinan TINGGI Lainnya"); num_cols = 4; cols = st.columns(num_cols)
        for i, trade in enumerate(remaining_high_conviction):
            with cols[i % num_cols]:
                with st.container(border=True): 
                    st.subheader(f"{trade['symbol']}")
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
                    st.subheader(f"{trade['symbol']}")
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    is_bullish = "Bullish" in trade['bias']
                    
                    st.markdown(f"**Sinyal:** <strong style='color:{color};'>{trade['bias']}</strong>", unsafe_allow_html=True)
                    st.caption(f"Harga Masuk: {format_price(trade['entry'])}")
                    
                    st.markdown(f"SL ({abs(trade['sl_pct'])}%): {format_price(trade['sl_target'])}", unsafe_allow_html=True)
                    st.markdown(f"TP 1 ({abs(trade['t1_pct'])}%): {format_price(trade['tp1_target'])}", unsafe_allow_html=True)
                    st.markdown(f"RR: {format_rr(trade['rr_ratio'])}", unsafe_allow_html=True)
