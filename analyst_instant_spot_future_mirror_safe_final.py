def analyze_df(df):
    """
    Analisis DataFrame harga OHLCV.
    Aman dari semua error tipe dan kolom hilang.
    """
    # pastikan benar-benar DataFrame
    if not isinstance(df, pd.DataFrame):
        return {
            "structure": "No Data",
            "fib_bias": "Netral",
            "support": None,
            "resistance": None,
            "current": None,
            "change": 0.0,
            "conviction": "Rendah"
        }

    # jika kosong
    if df.empty:
        return {
            "structure": "No Data",
            "fib_bias": "Netral",
            "support": None,
            "resistance": None,
            "current": None,
            "change": 0.0,
            "conviction": "Rendah"
        }

    # jika kolom Close tidak ada
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        else:
            return {
                "structure": "No Data",
                "fib_bias": "Netral",
                "support": None,
                "resistance": None,
                "current": None,
                "change": 0.0,
                "conviction": "Rendah"
            }

    # pastikan kolom numerik valid
    df = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # aman dropna
    try:
        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])
    except Exception:
        return {
            "structure": "No Data",
            "fib_bias": "Netral",
            "support": None,
            "resistance": None,
            "current": None,
            "change": 0.0,
            "conviction": "Rendah"
        }

    if len(df) < 5:
        return {
            "structure": "Data Kurang",
            "fib_bias": "Netral",
            "support": None,
            "resistance": None,
            "current": None,
            "change": 0.0,
            "conviction": "Rendah"
        }

    # isi kolom hilang
    for c in ["High", "Low"]:
        if c not in df.columns:
            df[c] = df["Close"]

    # hitung harga & persentase perubahan
    cur = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else cur
    change = ((cur - prev) / prev * 100) if prev != 0 else 0.0

    hi, lo = float(df["High"].max()), float(df["Low"].min())
    diff = hi - lo if hi and lo else 0
    fib_bias = "Netral"
    if diff > 0:
        fib61, fib38 = hi - diff * 0.618, hi - diff * 0.382
        if cur > fib61:
            fib_bias = "Bullish"
        elif cur < fib38:
            fib_bias = "Bearish"

    struct = "Konsolidasi"
    try:
        rec = df[["High", "Low"]].tail(20)
        if len(rec) >= 10:
            hh = rec["High"].rolling(5).max().dropna()
            ll = rec["Low"].rolling(5).min().dropna()
            if len(hh) >= 2 and len(ll) >= 2:
                if hh.iloc[-1] > hh.iloc[-2] and ll.iloc[-1] > ll.iloc[-2]:
                    struct = "Bullish"
                elif hh.iloc[-1] < hh.iloc[-2] and ll.iloc[-1] < ll.iloc[-2]:
                    struct = "Bearish"
    except Exception:
        pass

    support = float(df["Low"].rolling(20).min().iloc[-1])
    resistance = float(df["High"].rolling(20).max().iloc[-1])

    if struct in ["Bullish", "Bearish"] and fib_bias == struct:
        conviction = "Tinggi"
    elif struct in ["Bullish", "Bearish"] or fib_bias in ["Bullish", "Bearish"]:
        conviction = "Sedang"
    else:
        conviction = "Rendah"

    return {
        "structure": struct,
        "fib_bias": fib_bias,
        "support": support,
        "resistance": resistance,
        "current": cur,
        "change": change,
        "conviction": conviction
    }
