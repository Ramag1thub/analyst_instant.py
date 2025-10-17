# File: analyst_instant.py
# Versi: 32.0 - SINYAL RENDAH DIJAMIN (Menggunakan Data Mover sebagai Fallback Sinyal)
# Tujuan: Memastikan sinyal Konviksi Rendah pasti muncul dengan menggunakan data pergerakan 24 jam sebagai kriteria fallback.

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests 
import json

# --- PENTING: PENGGUNAAN API PUBLIK BINANCE FUTURES ---
BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/klines"
REQUEST_TIMEOUT = 30 

# --- DAFTAR KOIN BLUE-CHIP (HANYA ~35 KOIN LIKUIDITAS TERTINGGI UNTUK STABILITAS) ---
BASE_COIN_UNIVERSE = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT',
    'DOT/USDT', 'MATIC/USDT', 'LTC/USDT', 'BCH/USDT', 'LINK/USDT', 'AVAX/USDT', 'UNI/USDT',
    'XLM/USDT', 'ETC/USDT', 'ALGO/USDT', 'ATOM/USDT', 'FIL/USDT', 'EGLD/USDT', 'TRX/USDT',
    'AAVE/USDT', 'COMP/USDT', 'MKR/USDT', 'YFI/USDT', 'DYDX/USDT', 'SUI/USDT', 'SEI/USDT',
    'APT/USDT', 'IMX/USDT', 'ARB/USDT', 'OP/USDT', 'RNDR/USDT', 'TIA/USDT', 'FET/USDT'
]

# --- DEKLARASI GLOBAL ---
total_coins_scanned = len(BASE_COIN_UNIVERSE)

# --- UI/UX CSS (DIPINDAHKAN KE ATAS) ---
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
    .low-conviction { color: #5c7e8e; } 
    .report-detail { font-size: 12px; margin-top: 5px; line-height: 1.5; }
    .mover-price { font-size: 1.1em; }
</style>
"""

# --- PENGATURAN HALAMAN & KONFIGURASI AWAL ---
st.set_page_config(layout="wide", page_title="Instant AI Analyst", initial_sidebar_state="collapsed")
st.markdown(INSTANT_CSS, unsafe_allow_html=True) 
# --- FUNGSI HEALTH CHECK API BINANCE ---
@st.cache_data(ttl=60)
def binance_health_check():
    """Memeriksa koneksi API dengan mengambil data BTC. Mengembalikan True jika berhasil."""
    try:
        symbol_id = 'BTCUSDT'
        params = {'symbol': symbol_id, 'interval': '1m', 'limit': 1}
        response = requests.get(BINANCE_API_URL, params=params, timeout=5) 
        response.raise_for_status()
        bars = response.json()
        if bars and len(bars) > 0:
            return True
        return False
    except requests.exceptions.RequestException:
        return False
    except Exception:
        return False

# --- FUNGSI UTAMA (SINKRON - MENGGUNAKAN REQUESTS) ---

@st.cache_data(show_spinner=True, ttl=60) 
def fetch_daily_data(symbol, days=7):
    """Mengambil data HANYA 7 HARI untuk beban API yang sangat ringan."""
    
    symbol_id = symbol.replace('/', '') 
    
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - 86400000 * days

    params = {
        'symbol': symbol_id,
        'interval': '1d',
        'startTime': start_time_ms,
        'limit': 1000 
    }
    
    df = None
    try:
        response = requests.get(BINANCE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() 
        
        bars = response.json()
        
        if bars and len(bars) > 0:
            df = pd.DataFrame.from_records(bars)
            
            df = df.iloc[:, 0:6] 
            df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') 
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
            df.set_index('timestamp', inplace=True)
            
    except requests.exceptions.RequestException:
        return None
    except Exception:
        pass
        
    return df

@st.cache_data(show_spinner=False)
def analyze_structure(df):
    """Analisis struktur dan mengembalikan status lengkap."""
    if df is None or len(df) < 7: 
        return {'structure': 'Data Tidak Cukup', 'current_price': None, 'fib_bias': 'Netral', 'high': None, 'low': None, 'proxy_low': None, 'proxy_high': None, 'change_pct': 0.0}

    structure = "Konsolidasi"
    current_close = df['Close'].iloc[-1]
    df_recent = df[['High', 'Low']].tail(14)

    # Logika Market Structure 
    if len(df_recent) >= 14:
        recent_highs = df_recent['High'].rolling(window=5, center=True).max().dropna()
        recent_lows = df_recent['Low'].rolling(window=5, center=True).min().dropna()
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            high_h = recent_highs.iloc[-1] > recent_highs.iloc[-2]
            low_h = recent_lows.iloc[-1] > recent_lows.iloc[-2]
            if high_h and low_h: structure = "Bullish"
            elif recent_highs.iloc[-1] < recent_highs.iloc[-2] and recent_lows.iloc[-1] < recent_lows.iloc[-2]: structure = "Bearish"

    # Logika Fibonacci 
    max_price = df['High'].max(); min_price = df['Low'].min(); diff = max_price - min_price
    fib_level = {}
    for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
        fib_level[f'FIB_{int(level*100)}'] = max_price - (diff * level)
        
    fib_bias = "Netral"
    if 'FIB_61' in fib_level and current_close > fib_level['FIB_61']: fib_bias = "Bullish"
    elif 'FIB_38' in fib_level and current_close < fib_level['FIB_38']: fib_bias = "Bearish"

    # Logika Perubahan 24H (FallBack Data Mover)
    change_pct = 0.0
    if len(df) >= 2:
        close_24h_ago = df['Close'].iloc[-2]
        if close_24h_ago != 0:
             change_pct = ((current_close - close_24h_ago) / close_24h_ago) * 100
        
    return {
        'structure': structure, 
        'current_price': current_close, 
        'fib_bias': fib_bias,
        'high': df['High'].max(), 
        'low': df['Low'].min(),
        'proxy_low': df['Low'].iloc[-14:].min() if len(df) >= 14 else current_close,
        'proxy_high': df['High'].iloc[-14:].max() if len(df) >= 14 else current_close,
        'change_pct': change_pct 
    }

def find_signal_resampled(symbol, user_timeframe):
    """Mengambil data dan menganalisis secara lokal (SINKRON)."""
    
    df_daily = fetch_daily_data(symbol) 
    
    if df_daily is None: 
        return {'symbol': symbol, 'conviction': 'Error', 'change_pct': 0.0, 'current_price': None}

    resample_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    
    analysis_d = analyze_structure(df_daily)
    current_price = analysis_d.get('current_price')
    change_pct = analysis_d.get('change_pct', 0.0)
    
    # Cek minimal data untuk analisis struktur dan resample (minimal 7 hari)
    if len(df_daily) < 7: 
        return {'symbol': symbol, 'conviction': 'Nihil', 'change_pct': change_pct, 'current_price': current_price,
                'report_w_structure': analysis_d.get('structure'), 
                'report_d_structure': analysis_d.get('structure')}


    try:
        
        df_w = df_daily.resample('W').agg(resample_rules).dropna()
        analysis_w = analyze_structure(df_w)
        
        if user_timeframe == '1d':
            analysis_user = analysis_d
        else:
            df_user = df_daily.resample(user_timeframe).agg(resample_rules).dropna()
            analysis_user = analyze_structure(df_user)
            
    except Exception:
        return {'symbol': symbol, 'conviction': 'Nihil', 'change_pct': change_pct, 'current_price': current_price,
                'report_w_structure': analysis_d.get('structure'), 
                'report_d_structure': analysis_d.get('structure')} 

    # Mengambil Status Struktur dan Data Proksi
    structure_w = analysis_w['structure']; structure_d = analysis_d['structure']; structure_user = analysis_user['structure']
    fib_bias_d = analysis_d['fib_bias']; 
    proxy_low = analysis_d['proxy_low']
    proxy_high = analysis_d['proxy_high']
    
    conviction = "Nihil"; bias = "Netral" 
    entry_price = current_price 
    sl_pct = 0; tp1_pct = 0; tp2_pct = 0
    
    # --- LOGIKA KONVIKSI & PENETAPAN ENTRY SMART MONEY (PALING LONGGAR) ---
    
    # 1. KRITERIA TINGGI (Paling ketat)
    if structure_w == 'Bullish' and structure_d == 'Bullish' and structure_user == 'Bullish' and fib_bias_d == 'Bullish': 
        conviction = "Tinggi"; bias = "Bullish Kuat"; tp1_pct = 5; tp2_pct = 10; sl_pct = 2.5 
        entry_price = proxy_low 
    elif structure_w == 'Bearish' and structure_d == 'Bearish' and structure_user == 'Bearish' and fib_bias_d == 'Bearish': 
        conviction = "Tinggi"; bias = "Bearish Kuat"; tp1_pct = -5; tp2_pct = -10; sl_pct = -2.5 
        entry_price = proxy_high
    
    # 2. KRITERIA SEDANG
    elif structure_d == 'Bullish' or structure_user == 'Bullish': 
        conviction = "Sedang"; bias = "Cenderung Bullish"
        tp1_pct = 3; tp2_pct = 7; sl_pct = 1.0 
        entry_price = proxy_low * 1.002
    elif structure_d == 'Bearish' or structure_user == 'Bearish': 
        conviction = "Sedang"; bias = "Cenderung Bearish"
        tp1_pct = -3; tp2_pct = -7; sl_pct = -1.0 
        entry_price = proxy_high * 0.998
    
    # 3. KRITERIA RENDAH (FINAL LONGGAR V27.0 - BERDASARKAN STRUKTUR)
    elif structure_user == 'Bullish' or structure_d == 'Bullish' or structure_w == 'Bullish':
        conviction = "Rendah"; bias = "Bullish Potensial"
        tp1_pct = 2; tp2_pct = 4; sl_pct = 1.0 
        entry_price = proxy_low * 1.005 
    elif structure_user == 'Bearish' or structure_d == 'Bearish' or structure_w == 'Bearish':
        conviction = "Rendah"; bias = "Bearish Potensial"
        tp1_pct = -2; tp2_pct = -4; sl_pct = -1.0 
        entry_price = proxy_high * 0.995 
        
    if conviction in ['Tinggi', 'Sedang', 'Rendah']:
        # Hitung ulang SL, TP, dan RR
        sl_multiplier = 1 + (sl_pct / 100) if sl_pct >= 0 else 1 - abs(sl_pct / 100)
        sl_target = entry_price * sl_multiplier
        
        tp1_target = entry_price * (1 + (tp1_pct / 100))
        tp2_target = entry_price * (1 + (tp2_pct / 100))
        
        risk_amount = abs(entry_price - sl_target)
        reward_amount = abs(tp1_target - entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'symbol': symbol, 'bias': bias, 'timeframe': user_timeframe, 'conviction': conviction,
            'entry': entry_price, 'sl_target': sl_target, 'tp1_target': tp1_target, 
            'tp2_target': tp2_target, 't1_pct': tp1_pct, 't2_pct': tp2_pct, 
            'sl_pct': sl_pct, 'rr_ratio': rr_ratio, 'change_pct': change_pct,
            
            # Data Analisis Komprehensif (REPORT)
            'report_w_structure': structure_w, 'report_d_structure': structure_d,
            'report_user_structure': structure_user, 'report_fib_bias': fib_bias_d,
            'report_current_price': current_price, 'report_proxy_low': proxy_low,
            'report_proxy_high': proxy_high,
        }
    
    # Jika sinyal Nihil, kembalikan data minimal untuk mover report
    return {
        'symbol': symbol, 'conviction': 'Nihil', 'change_pct': change_pct,
        'current_price': current_price,
        'report_w_structure': structure_w, 
        'report_d_structure': structure_d,
    }

def run_scanner_streamed_sync(coin_universe, timeframe, status_placeholder):
    """Menjalankan pemindaian secara SINKRON (berurutan) - SANGAT STABIL."""
    
    all_results = []
    
    for i, symbol in enumerate(coin_universe):
        result = find_signal_resampled(symbol, timeframe)
        if result:
            all_results.append(result)
        
        # Jeda 50ms per koin (maks 20 request/detik) 
        time.sleep(0.05) 
        
        if i % 5 == 0 or i == total_coins_scanned - 1: # Update status lebih sering karena koin lebih sedikit
            found_signals = len([r for r in all_results if r.get('conviction') not in ['Nihil', 'Error']])
            status_placeholder.info(f"Memindai {symbol}... Koin ke {i+1} dari {total_coins_scanned}. Ditemukan {found_signals} sinyal trading.")
            
    return all_results

# --- EKSEKUSI: HEALTH CHECK ---
is_connected = binance_health_check()

if not is_connected:
    st.error("🔴 **KONEKSI GAGAL TOTAL:** Gagal terhubung ke Binance API (BTC/USDT Health Check).")
    st.warning("Kendala jaringan eksternal (API Down/Rate Limit) terdeteksi. Silakan coba tombol **Scan Ulang Sekarang** di bawah, atau refresh browser Anda.")
    
    col1, col2 = st.columns([1.5, 8.5])
    if col1.button("🔄 Scan Ulang Sekarang (API Check)"):
        st.cache_data.clear() 
        st.rerun() 
    st.stop()
    
# Jika terhubung, lanjutkan ke UI dan Pemindaian
st.success(f"🟢 **KONEKSI API BERHASIL:** Binance API Health Check (BTC/USDT) berhasil. Melanjutkan pemindaian {total_coins_scanned} koin.")


# --- ANTARMUKA APLIKASI WEB ---
st.title("🚀 Instant AI Signal Dashboard")
st.caption(f"Menganalisis **{total_coins_scanned} koin Blue-Chip (Fokus Kualitas)** dengan **Smart Money Entry**.")

col1, col2, col3 = st.columns([1.5, 1.5, 7])
selected_tf = col1.selectbox("Pilih Timeframe Sinyal:", ['1d', '4h', '1h'], help="Pilih Timeframe sinyal yang diinginkan.")
if col2.button("🔄 Scan Ulang Sekarang (Data Penuh)"):
    st.cache_data.clear() 
    st.rerun() 
st.markdown("---")

# --- INISIALISASI VARIABEL RUNTIME ---
status_placeholder = st.empty() 
start_time = time.time() 

# --- PROSES PEMINDAIAN UTAMA (SINKRON) ---

status_placeholder.info(f"Memulai pemindaian instan untuk **{total_coins_scanned} koin** secara berurutan...")

all_results = run_scanner_streamed_sync(BASE_COIN_UNIVERSE, selected_tf, status_placeholder)
total_time = time.time() - start_time


# --- POST-PROCESSING SINYAL JAMINAN ---
# Filter untuk Mover Report
movers_for_guarantee = [r for r in all_results if r.get('current_price') is not None and r.get('change_pct') != 0.0 and r.get('conviction') != 'Error']
movers_for_guarantee.sort(key=lambda x: x['change_pct'], reverse=True)
top_bullish_movers = movers_for_guarantee[:3]
top_bearish_movers = movers_for_guarantee[-3:][::-1]


# Sinyal Jaminan: Memaksa sinyal rendah jika koin adalah Top Mover dan saat ini Nihil
for result in all_results:
    if result.get('conviction') == 'Nihil' and result.get('symbol') in [m['symbol'] for m in top_bullish_movers + top_bearish_movers]:
        
        is_bullish = result.get('symbol') in [m['symbol'] for m in top_bullish_movers]
        
        # Logika Override untuk Sinyal Rendah (Guaranteed)
        if is_bullish:
            result.update({
                'conviction': 'Rendah', 
                'bias': 'Bullish Mover (Jaminan)',
                'entry': result['current_price'] * 1.001, 
                't1_pct': 1.5, 't2_pct': 3.0, 'sl_pct': 1.0,
            })
        else: # Bearish
            result.update({
                'conviction': 'Rendah', 
                'bias': 'Bearish Mover (Jaminan)',
                'entry': result['current_price'] * 0.999, 
                't1_pct': -1.5, 't2_pct': -3.0, 'sl_pct': -1.0,
            })
            
        # Perhitungan SL/TP/RR harus dilakukan ulang (menggunakan persentase sederhana)
        entry_price = result['entry']
        sl_pct = result['sl_pct']
        tp1_pct = result['t1_pct']
        tp2_pct = result['t2_pct']
        
        sl_multiplier = 1 + (sl_pct / 100) if sl_pct >= 0 else 1 - abs(sl_pct / 100)
        sl_target = entry_price * sl_multiplier
        tp1_target = entry_price * (1 + (tp1_pct / 100))
        tp2_target = entry_price * (1 + (tp2_pct / 100))
        
        risk_amount = abs(entry_price - sl_target)
        reward_amount = abs(tp1_target - entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        result.update({'sl_target': sl_target, 'tp1_target': tp1_target, 'tp2_target': tp2_target, 'rr_ratio': rr_ratio})
        
# --- FUNGSI DISPLAY PEMBANTU (SAMA) ---
def format_price(price):
    if price is None: return "N/A"
    if price < 0.001: return f"{price:,.8f}"
    if price < 0.1: return f"{price:,.5f}"
    if price < 10: return f"{price:,.4f}"
    return f"{price:,.2f}"

def format_rr(ratio):
    return f"{ratio:.1f}:1" if ratio > 0 else "0:1"

def display_report_details(trade):
    tab1, tab2 = st.tabs(["📊 Trading Params", "🔬 Analisis Komprehensif"])

    with tab1:
        st.markdown(f"""
        **Entri Target (OB):** <span class='entry'>{format_price(trade['entry'])}</span>
        **SL ({abs(trade['sl_pct'])}% dari Entry):** <span class='loss'>{format_price(trade['sl_target'])}</span> | **R/R Aktual:** **{format_rr(trade['rr_ratio'])}**
        **TP 1 ({abs(trade['t1_pct'])}%):** <span class='profit'>{format_price(trade['tp1_target'])}</span>
        **TP 2 ({abs(trade['t2_pct'])}%):** <span class='profit'>{format_price(trade['tp2_target'])}</span>
        """, unsafe_allow_html=True)

    with tab2:
        def get_trend_color(trend):
            if 'Bullish' in trend or 'Bullish' in trade['bias']: return 'lime'
            if 'Bearish' in trend or 'Bearish' in trade['bias']: return 'salmon'
            return '#5c7e8e'
        
        ob_level = trade.get('report_proxy_low') if 'Bullish' in trade['bias'] else trade.get('report_proxy_high')
        ob_label = "Support OB (Low 14D)" if 'Bullish' in trade['bias'] else "Resistance OB (High 14D)"
        
        st.markdown(f"""
        <div class='report-detail'>
            <b>HARGA TERAKHIR:</b> {format_price(trade.get('current_price') or trade.get('report_current_price'))}<br>
            <b>TARGET OB/WALL ({ob_label}):</b> <span style='color:{get_trend_color(trade['bias'])};'>{format_price(ob_level) if ob_level else 'N/A'}</span><br>
            ---
            <b>STRUKTUR PASAR:</b><br>
            • Weekly (1W): <span style='color:{get_trend_color(trade.get('report_w_structure', 'N/A'))};'>{trade.get('report_w_structure', 'N/A')}</span><br>
            • Daily (1D): <span style='color:{get_trend_color(trade.get('report_d_structure', 'N/A'))};'>{trade.get('report_d_structure', 'N/A')}</span><br>
            • User TF ({selected_tf}): <span style='color:{get_trend_color(trade.get('report_user_structure', 'N/A'))};'>{trade.get('report_user_structure', 'N/A')}</span><br>
            <b>KONF. FIBONACCI:</b> {trade.get('report_fib_bias', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

# --- TAMPILKAN HASIL AKHIR ---
status_placeholder.empty()

# Filter sinyal trading yang valid
found_trades = [r for r in all_results if r.get('conviction') in ['Tinggi', 'Sedang', 'Rendah']]
total_signals = len(found_trades)
signal_percentage = (total_signals / total_coins_scanned) * 100 if total_coins_scanned > 0 else 0


# --- BAGIAN REPORT MOVER (SELALU TAMPIL) ---
st.header("⚡ Laporan Koin Penggerak (24 Jam) - Top Movers")
st.caption("Analisis ini hanya mencakup koin Blue-Chip teratas untuk stabilitas data.")

if movers_for_guarantee:
    top_bullish = top_bullish_movers
    top_bearish = top_bearish_movers

    cols_movers = st.columns(3)
    
    def get_trend_color(trend):
        if 'Bullish' in trend: return 'lime'
        if 'Bearish' in trend: return 'salmon'
        return '#5c7e8e'

    def get_mover_context(mover):
        w_struct = mover.get('report_w_structure', 'N/A')
        d_struct = mover.get('report_d_structure', 'N/A')
        
        w_color = get_trend_color(w_struct)
        d_color = get_trend_color(d_struct)

        return f"""
        <div style='font-size: 11px; margin-top: 5px;'>
            <b>KONTEKS STRUKTUR:</b><br>
            • 1W: <span style='color:{w_color};'>{w_struct}</span><br>
            • 1D: <span style='color:{d_color};'>{d_struct}</span>
        </div>
        """

    with cols_movers[0]:
        st.subheader("📈 Top 3 Bullish Movers")
        for mover in top_bullish:
            color = 'lime' if mover['change_pct'] > 0 else 'salmon'
            st.markdown(f"""
            <div class='signal-card'>
                **{mover['symbol']}**<br>
                <span style='color:{color}; font-weight:bold;'>{mover['change_pct']:.2f}%</span> (24H)<br>
                <p class='mover-price'>Harga: {format_price(mover['current_price'])}</p>
                {get_mover_context(mover)}
            </div>
            """, unsafe_allow_html=True)

    with cols_movers[1]:
        st.subheader("📉 Top 3 Bearish Movers")
        for mover in top_bearish:
            color = 'salmon' if mover['change_pct'] < 0 else 'lime'
            st.markdown(f"""
            <div class='signal-card'>
                **{mover['symbol']}**<br>
                <span style='color:{color}; font-weight:bold;'>{mover['change_pct']:.2f}%</span> (24H)<br>
                <p class='mover-price'>Harga: {format_price(mover['current_price'])}</p>
                {get_mover_context(mover)}
            </div>
            """, unsafe_allow_html=True)
    
    with cols_movers[2]:
         st.subheader("ℹ️ Keterangan Konteks")
         st.info("Koin ini memiliki pergerakan harga 24 jam tertinggi. Konteks Struktur 1W/1D menunjukkan bias jangka panjang dan menengah, yang dapat memvalidasi momentum pergerakan saat ini.")

else:
    st.error(f"Gagal memuat data harga dari {total_coins_scanned} koin. Data koin tidak cukup untuk analisis. Silakan coba 'Scan Ulang Sekarang'.")


st.markdown("---") 

# --- BAGIAN SINYAL TRADING (HANYA JIKA DITEMUKAN) ---

if not found_trades:
    st.warning("❌ Tidak ada sinyal Trading (Tinggi/Sedang/Rendah) yang ditemukan berdasarkan kriteria Market Structure saat ini. Silakan coba Timeframe lain.")

else:
    st.success(f"""
        ✅ Pemindaian selesai dalam **{total_time:.2f} detik**. 
        Ditemukan {total_signals} sinyal potensial dari {total_coins_scanned} koin.
        **Status Pasar (Sinyal Sedang/Tinggi/Rendah):** <span class='highlight-pct'>{signal_percentage:.2f}%</span> dari koin yang dipindai.
    """, unsafe_allow_html=True)
    
    # Pisahkan dan Urutkan
    high_conviction = [t for t in found_trades if t['conviction'] == 'Tinggi']
    medium_conviction = [t for t in found_trades if t['conviction'] == 'Sedang']
    low_conviction = [t for t in found_trades if t['conviction'] == 'Rendah']
    
    # Pisahkan Sinyal Jaminan Rendah
    guaranteed_low = [t for t in low_conviction if 'Mover (Jaminan)' in t['bias']]
    normal_low = [t for t in low_conviction if 'Mover (Jaminan)' not in t['bias']]
    
    all_signals = sorted(high_conviction, key=lambda x: x['symbol']) + \
                  sorted(medium_conviction, key=lambda x: x['symbol']) + \
                  sorted(normal_low, key=lambda x: x['symbol']) + \
                  sorted(guaranteed_low, key=lambda x: x['symbol']) # Sinyal Jaminan ditaruh paling akhir

    top_3_signals = all_signals[:3]
    
    # --- TAMPILAN TOP 3 KOIN ---
    if top_3_signals:
        st.header("🏆 Top 3 Sinyal Kuat (Rekomendasi Trading)")
        cols_top = st.columns(3)
        for i, trade in enumerate(top_3_signals):
            with cols_top[i]:
                with st.container(border=True):
                    st.markdown(f"**#{i+1}: {trade['symbol']}**", unsafe_allow_html=True) 
                    color = "lime" if "Bullish" in trade['bias'] else "salmon" if "Bearish" in trade['bias'] else "#5c7e8e"

                    st.markdown(f"**Konviksi:** <strong style='color:{color};'>{trade['conviction']} ({trade['bias']})</strong>", unsafe_allow_html=True)
                    st.caption(f"Timeframe: {selected_tf}")
                    
                    display_report_details(trade)


    st.markdown("---") 
    
    # 3. Tampilkan Sinyal Keyakinan TINGGI
    if high_conviction:
        st.subheader("🔥 Sinyal Keyakinan TINGGI"); num_cols = 4; cols = st.columns(num_cols)
        for i, trade in enumerate(high_conviction):
            with cols[i % num_cols]:
                with st.container(border=True): 
                    st.subheader(f"{trade['symbol']}")
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    st.markdown(f"**Sinyal:** <strong style='color:{color};'>{trade['bias']}</strong>", unsafe_allow_html=True)
                    display_report_details(trade)
                    
    # 4. Tampilkan Sinyal Keyakinan SEDANG
    if medium_conviction:
        st.subheader("👍 Sinyal Keyakinan SEDANG (Low Risk Entry)"); num_cols = 4; cols = st.columns(num_cols)
        for i, trade in enumerate(medium_conviction):
            with cols[i % num_cols]:
                with st.container(border=True):
                    st.subheader(f"{trade['symbol']}")
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    st.markdown(f"**Sinyal:** <strong style='color:{color};'>{trade['bias']}</strong>", unsafe_allow_html=True)
                    display_report_details(trade)
                    
    # 5. Tampilkan Sinyal Keyakinan RENDAH
    if low_conviction:
        st.subheader("⚪ Sinyal Keyakinan RENDAH (Potensial & Jaminan)"); num_cols = 4; cols = st.columns(num_cols)
        
        # Gabungkan Normal Low dan Guaranteed Low
        for i, trade in enumerate(normal_low + guaranteed_low):
            with cols[i % num_cols]:
                with st.container(border=True):
                    st.subheader(f"{trade['symbol']}")
                    color = "lime" if "Bullish" in trade['bias'] else "salmon"
                    
                    st.markdown(f"**Sinyal:** <strong style='color:{color};'>{trade['conviction']} ({trade['bias']})</strong>", unsafe_allow_html=True)
                    st.caption(f"Timeframe: {selected_tf}")
                    
                    display_report_details(trade)
