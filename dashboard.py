# ════════════════════════════════════
# TRADE DASHBOARD - ENHANCED VERSION
# Requirements: streamlit, yfinance, pandas, numpy, ta
# Optional in requirements.txt: schedule can remain, but is not used here.
# ════════════════════════════════════

import datetime as dt
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import ta
import yfinance as yf

# ════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════

NIFTY200 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS",
    "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS",
    "VBL.NS", "PREMIERENE.NS", "PIIND.NS", "HINDZINC.NS",
    "SHRIRAMFIN.NS", "CIPLA.NS", "TATACAP.NS", "DRREDDY.NS",
    "SOLARINDS.NS", "COALINDIA.NS", "ONGC.NS", "ADANIPORTS.NS",
    "HINDALCO.NS", "GRASIM.NS", "TRENT.NS", "TATAMOTORS.NS",
    "JSWSTEEL.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS"
]

NIFTY_SYMBOL = "^NSEI"
IST = ZoneInfo("Asia/Kolkata")
MIN_PROB_DEFAULT = 65

# Optional future-ready manual fields. Add stocks/results here when needed.
RESULTS_CALENDAR = {
    # "RELIANCE": "2026-04-30",
}

# ════════════════════════════════════
# DATA FETCHERS
# ════════════════════════════════════

def _normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance normal and MultiIndex columns safely."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df.dropna(how="all")

@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str = "9mo") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        df = _normalize_yf_columns(df)
        if df is None or df.empty or len(df) < 60:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=180)
def fetch_intraday_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period="1d", interval="5m", progress=False, auto_adjust=False)
        df = _normalize_yf_columns(df)
        if df is None or df.empty or len(df) < 20:
            return None
        return df
    except Exception:
        return None

def fetch_nifty_data() -> Optional[pd.DataFrame]:
    return fetch_stock_data(NIFTY_SYMBOL, period="6mo")

# ════════════════════════════════════
# MARKET PHASE
# ════════════════════════════════════

def get_market_phase() -> Tuple[str, str, dt.datetime]:
    now = dt.datetime.now(IST)
    weekday = now.weekday()  # Monday=0

    if weekday >= 5:
        return "WEEKEND", "🔴 Weekend / Market closed — showing last available session", now

    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    near_close_time = now.replace(hour=14, minute=45, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < open_time:
        return "PRE_MARKET", "🟡 Pre-market — showing last available session", now
    if open_time <= now < near_close_time:
        return "OPEN", "🟢 Market open — intraday scan active", now
    if near_close_time <= now <= close_time:
        return "NEAR_CLOSE", "🟠 Near close — BTST/STBT priority mode", now
    return "CLOSED", "🔴 Market closed — showing last completed session analysis", now

# ════════════════════════════════════
# INDICATOR CALCULATIONS
# ════════════════════════════════════

def safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator * 100)

def calculate_indicators(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or len(df) < 60:
        return None
    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        ema20 = ta.trend.EMAIndicator(close, 20).ema_indicator()
        ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, 200).ema_indicator() if len(df) >= 200 else pd.Series(np.nan, index=df.index)
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()

        bb = ta.volatility.BollingerBands(close, 20, 2)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        avg_vol20 = volume.rolling(20).mean()
        # Previous 10 sessions average excludes today, so today is compared fairly against last 10 completed days.
        avg_vol10_prev = volume.shift(1).rolling(10).mean()
        rel_vol = volume / avg_vol20
        vol_spike_10d = volume / avg_vol10_prev

        high20_prev = high.shift(1).rolling(20).max()
        low20_prev = low.shift(1).rolling(20).min()
        high5 = high.rolling(5).max()
        low5 = low.rolling(5).min()
        narrow_range = (high5 - low5) / close < 0.04

        body = (close - open_).abs()
        avg_body20 = body.rolling(20).mean()
        candle_range = (high - low).replace(0, np.nan)
        body_pct = body / candle_range
        close_location = (close - low) / candle_range  # 1 = close near high, 0 = close near low

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        sweep_low_reversal = (low < prev_low) & (close > prev_low) & (close > open_)
        sweep_high_reversal = (high > prev_high) & (close < prev_high) & (close < open_)

        structure_break_up = close > high20_prev
        structure_break_down = close < low20_prev
        displacement_up = (close > open_) & (body > avg_body20 * 1.25) & (close_location > 0.65)
        displacement_down = (close < open_) & (body > avg_body20 * 1.25) & (close_location < 0.35)

        c = float(close.iloc[-1])
        prev_c = float(close.iloc[-2])
        change_pct = safe_pct(c - prev_c, prev_c)
        ret20 = safe_pct(c - float(close.iloc[-20]), float(close.iloc[-20])) if len(close) >= 20 else 0.0
        atr_pct = safe_pct(float(atr.iloc[-1]), c)

        return {
            "close": c,
            "open": float(open_.iloc[-1]),
            "ema20": float(ema20.iloc[-1]),
            "ema50": float(ema50.iloc[-1]),
            "ema200": float(ema200.iloc[-1]) if not pd.isna(ema200.iloc[-1]) else np.nan,
            "rsi": float(rsi.iloc[-1]),
            "rel_vol": float(rel_vol.iloc[-1]) if not pd.isna(rel_vol.iloc[-1]) else 0.0,
            "vol_spike_10d": float(vol_spike_10d.iloc[-1]) if not pd.isna(vol_spike_10d.iloc[-1]) else 0.0,
            "avg_vol10_prev": float(avg_vol10_prev.iloc[-1]) if not pd.isna(avg_vol10_prev.iloc[-1]) else 0.0,
            "avg_vol": float(avg_vol20.iloc[-1]) if not pd.isna(avg_vol20.iloc[-1]) else 0.0,
            "volume": float(volume.iloc[-1]),
            "bb_width": float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0.0,
            "high20": float(high20_prev.iloc[-1]) if not pd.isna(high20_prev.iloc[-1]) else float(high.iloc[-1]),
            "low20": float(low20_prev.iloc[-1]) if not pd.isna(low20_prev.iloc[-1]) else float(low.iloc[-1]),
            "narrow_range": bool(narrow_range.iloc[-1]),
            "change_pct": change_pct,
            "ret20": ret20,
            "day_high": float(high.iloc[-1]),
            "day_low": float(low.iloc[-1]),
            "atr": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
            "atr_pct": atr_pct,
            "body_pct": float(body_pct.iloc[-1]) if not pd.isna(body_pct.iloc[-1]) else 0.0,
            "close_location": float(close_location.iloc[-1]) if not pd.isna(close_location.iloc[-1]) else 0.5,
            "displacement_up": bool(displacement_up.iloc[-1]),
            "displacement_down": bool(displacement_down.iloc[-1]),
            "structure_break_up": bool(structure_break_up.iloc[-1]),
            "structure_break_down": bool(structure_break_down.iloc[-1]),
            "sweep_low_reversal": bool(sweep_low_reversal.iloc[-1]),
            "sweep_high_reversal": bool(sweep_high_reversal.iloc[-1]),
        }
    except Exception:
        return None

def calculate_intraday_indicators(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or len(df) < 20:
        return None
    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)
        hlc3 = (high + low + close) / 3
        vwap = (hlc3 * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()
        avg_vol = volume.rolling(20).mean()
        rel_vol = volume / avg_vol
        atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()

        or_high = float(high.iloc[:3].max())
        or_low = float(low.iloc[:3].min())
        body = (close - open_).abs()
        avg_body20 = body.rolling(20).mean()
        candle_range = (high - low).replace(0, np.nan)
        close_location = (close - low) / candle_range
        displacement_up = (close > open_) & (body > avg_body20 * 1.25) & (close_location > 0.65)
        displacement_down = (close < open_) & (body > avg_body20 * 1.25) & (close_location < 0.35)

        c = float(close.iloc[-1])
        prev_c = float(close.iloc[-2])
        return {
            "close": c,
            "vwap": float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else c,
            "ema9": float(ema9.iloc[-1]),
            "ema21": float(ema21.iloc[-1]),
            "rsi": float(rsi.iloc[-1]),
            "rel_vol": float(rel_vol.iloc[-1]) if not pd.isna(rel_vol.iloc[-1]) else 0.0,
            "volume": float(volume.iloc[-1]),
            "or_high": or_high,
            "or_low": or_low,
            "change_pct": safe_pct(c - prev_c, prev_c),
            "atr": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
            "close_location": float(close_location.iloc[-1]) if not pd.isna(close_location.iloc[-1]) else 0.5,
            "displacement_up": bool(displacement_up.iloc[-1]),
            "displacement_down": bool(displacement_down.iloc[-1]),
        }
    except Exception:
        return None

# ════════════════════════════════════
# WEIGHTED SCORING ENGINE
# ════════════════════════════════════

def add_points(reasons: List[str], condition: bool, points: int, label: str) -> int:
    if condition:
        reasons.append(f"+{points} {label}")
        return points
    return 0

def probability_label(score: int) -> str:
    if score >= 80:
        return "A+ ≥80%"
    if score >= 70:
        return "A 70-79%"
    if score >= 65:
        return "B+ 65-69%"
    if score >= 55:
        return "Watch 55-64%"
    return "Avoid <55%"

def star_rating(score: int) -> str:
    if score >= 80:
        return "⭐⭐⭐⭐⭐"
    if score >= 70:
        return "⭐⭐⭐⭐"
    if score >= 65:
        return "⭐⭐⭐"
    if score >= 55:
        return "⭐⭐"
    return "⭐"

def result_boost(symbol_name: str) -> Tuple[int, str]:
    raw = RESULTS_CALENDAR.get(symbol_name)
    if not raw:
        return 0, ""
    try:
        result_date = dt.date.fromisoformat(raw)
        today = dt.datetime.now(IST).date()
        days = (result_date - today).days
        if 0 <= days <= 7:
            return 5, f"+5 Results in {days}d"
    except Exception:
        pass
    return 0, ""

def positional_score(ind: Dict, nifty_ret20: float, symbol_name: str) -> Dict:
    if not ind:
        return None

    c = ind["close"]
    rs_long = ind["ret20"] > nifty_ret20
    rs_short = ind["ret20"] < nifty_ret20
    squeeze = ind["bb_width"] < 0.08 or ind["narrow_range"]
    not_too_extended = ind["atr_pct"] <= 5.5

    long_reasons, short_reasons = [], []
    long_score = 0
    short_score = 0

    long_score += add_points(long_reasons, c > ind["ema20"] and c > ind["ema50"], 20, "trend above EMA20/50")
    long_score += add_points(long_reasons, c >= ind["high20"] * 0.98 or ind["structure_break_up"], 15, "near 20D high / breakout")
    long_score += add_points(long_reasons, ind.get("vol_spike_10d", ind["rel_vol"]) >= 1.25, 15, "volume spike vs 10D avg")
    long_score += add_points(long_reasons, 50 <= ind["rsi"] <= 68, 10, "healthy RSI")
    long_score += add_points(long_reasons, squeeze, 10, "squeeze / narrow range")
    long_score += add_points(long_reasons, rs_long, 15, "stronger than Nifty")
    long_score += add_points(long_reasons, not_too_extended, 10, "risk not extended")
    long_score += add_points(long_reasons, ind["sweep_low_reversal"] or ind["displacement_up"], 5, "Trader X reversal/displacement")

    short_score += add_points(short_reasons, c < ind["ema20"] and c < ind["ema50"], 20, "trend below EMA20/50")
    short_score += add_points(short_reasons, c <= ind["low20"] * 1.02 or ind["structure_break_down"], 15, "near 20D low / breakdown")
    short_score += add_points(short_reasons, ind.get("vol_spike_10d", ind["rel_vol"]) >= 1.25, 15, "volume spike vs 10D avg")
    short_score += add_points(short_reasons, 32 <= ind["rsi"] <= 50, 10, "bearish RSI")
    short_score += add_points(short_reasons, squeeze, 10, "squeeze / narrow range")
    short_score += add_points(short_reasons, rs_short, 15, "weaker than Nifty")
    short_score += add_points(short_reasons, not_too_extended, 10, "risk not extended")
    short_score += add_points(short_reasons, ind["sweep_high_reversal"] or ind["displacement_down"], 5, "Trader X reversal/displacement")

    boost, boost_reason = result_boost(symbol_name)
    if boost:
        long_score += boost
        short_score += boost
        long_reasons.append(boost_reason)
        short_reasons.append(boost_reason)

    bias = "LONG" if long_score > short_score else "SHORT" if short_score > long_score else "NEUTRAL"
    active_score = max(long_score, short_score)
    reasons = long_reasons if bias == "LONG" else short_reasons if bias == "SHORT" else []

    return {"bias": bias, "active_score": min(active_score, 100), "reasons": reasons[:4]}

def btst_score(ind: Dict, nifty_ret20: float, symbol_name: str) -> Dict:
    if not ind:
        return None

    c = ind["close"]
    rs_long = ind["ret20"] > nifty_ret20
    rs_short = ind["ret20"] < nifty_ret20
    close_near_high = ind["close_location"] >= 0.70 or c >= ind["day_high"] * 0.99
    close_near_low = ind["close_location"] <= 0.30 or c <= ind["day_low"] * 1.01
    gap_risk_ok = ind["atr_pct"] <= 6.0 and abs(ind["change_pct"]) <= 6.5

    btst_reasons, stbt_reasons = [], []
    btst = 0
    stbt = 0

    btst += add_points(btst_reasons, close_near_high, 20, "close near day high")
    btst += add_points(btst_reasons, ind["displacement_up"] or ind["body_pct"] >= 0.55, 20, "bullish displacement/body")
    btst += add_points(btst_reasons, ind.get("vol_spike_10d", ind["rel_vol"]) >= 1.5, 20, "volume spike vs 10D avg")
    btst += add_points(btst_reasons, c > ind["ema20"] and c > ind["ema50"], 15, "EMA trend support")
    btst += add_points(btst_reasons, 52 <= ind["rsi"] <= 72, 10, "momentum RSI")
    btst += add_points(btst_reasons, rs_long, 10, "relative strength")
    btst += add_points(btst_reasons, gap_risk_ok, 5, "gap risk acceptable")

    stbt += add_points(stbt_reasons, close_near_low, 20, "close near day low")
    stbt += add_points(stbt_reasons, ind["displacement_down"] or ind["body_pct"] >= 0.55, 20, "bearish displacement/body")
    stbt += add_points(stbt_reasons, ind.get("vol_spike_10d", ind["rel_vol"]) >= 1.5, 20, "volume spike vs 10D avg")
    stbt += add_points(stbt_reasons, c < ind["ema20"] and c < ind["ema50"], 15, "EMA trend support")
    stbt += add_points(stbt_reasons, 28 <= ind["rsi"] <= 48, 10, "bearish RSI")
    stbt += add_points(stbt_reasons, rs_short, 10, "relative weakness")
    stbt += add_points(stbt_reasons, gap_risk_ok, 5, "gap risk acceptable")

    boost, boost_reason = result_boost(symbol_name)
    if boost:
        btst += boost
        stbt += boost
        btst_reasons.append(boost_reason)
        stbt_reasons.append(boost_reason)

    bias = "BTST" if btst > stbt else "STBT" if stbt > btst else "NEUTRAL"
    active_score = max(btst, stbt)
    reasons = btst_reasons if bias == "BTST" else stbt_reasons if bias == "STBT" else []

    return {"bias": bias, "active_score": min(active_score, 100), "reasons": reasons[:4]}

def intraday_score(ind: Dict) -> Dict:
    if not ind:
        return None

    c = ind["close"]
    long_reasons, short_reasons = [], []
    long_score = 0
    short_score = 0

    long_score += add_points(long_reasons, c > ind["vwap"], 20, "above VWAP")
    long_score += add_points(long_reasons, c > ind["ema9"] and c > ind["ema21"], 15, "above EMA9/21")
    long_score += add_points(long_reasons, ind["ema9"] > ind["ema21"], 10, "EMA momentum")
    long_score += add_points(long_reasons, c > ind["or_high"], 15, "opening range breakout")
    long_score += add_points(long_reasons, 45 <= ind["rsi"] <= 68, 10, "RSI zone")
    long_score += add_points(long_reasons, ind["rel_vol"] >= 1.5, 15, "relative volume")
    long_score += add_points(long_reasons, ind["displacement_up"], 15, "5m displacement")

    short_score += add_points(short_reasons, c < ind["vwap"], 20, "below VWAP")
    short_score += add_points(short_reasons, c < ind["ema9"] and c < ind["ema21"], 15, "below EMA9/21")
    short_score += add_points(short_reasons, ind["ema9"] < ind["ema21"], 10, "EMA momentum")
    short_score += add_points(short_reasons, c < ind["or_low"], 15, "opening range breakdown")
    short_score += add_points(short_reasons, 32 <= ind["rsi"] <= 55, 10, "RSI zone")
    short_score += add_points(short_reasons, ind["rel_vol"] >= 1.5, 15, "relative volume")
    short_score += add_points(short_reasons, ind["displacement_down"], 15, "5m displacement")

    bias = "LONG" if long_score > short_score else "SHORT" if short_score > long_score else "NEUTRAL"
    active_score = max(long_score, short_score)
    reasons = long_reasons if bias == "LONG" else short_reasons if bias == "SHORT" else []

    return {"bias": bias, "active_score": min(active_score, 100), "reasons": reasons[:4]}

# ════════════════════════════════════
# MAIN SCREENER ENGINE
# ════════════════════════════════════

def make_entry(symbol_name: str, ind: Dict, score_obj: Dict) -> Dict:
    score = int(score_obj["active_score"])
    atr = ind.get("atr", 0.0)
    bias = score_obj["bias"]
    if bias in ["LONG", "BTST"]:
        sl = ind["close"] - atr * 1.2 if atr else ind["day_low"]
        target = ind["close"] + atr * 2.0 if atr else ind["day_high"]
    else:
        sl = ind["close"] + atr * 1.2 if atr else ind["day_high"]
        target = ind["close"] - atr * 2.0 if atr else ind["day_low"]

    risk = abs(ind["close"] - sl)
    reward = abs(target - ind["close"])
    rr = round(reward / risk, 2) if risk else 0

    return {
        "symbol": symbol_name,
        "price": round(ind["close"], 2),
        "change": round(ind["change_pct"], 2),
        "score": score,
        "stars": star_rating(score),
        "prob": probability_label(score),
        "rsi": round(ind["rsi"], 1),
        "rel_vol": round(ind.get("rel_vol", 0), 2),
        "vol_spike_10d": round(ind.get("vol_spike_10d", ind.get("rel_vol", 0)), 2),
        "close_location_pct": round(ind.get("close_location", 0.5) * 100, 1),
        "body_pct": round(ind.get("body_pct", 0) * 100, 1),
        "bias": bias,
        "entry_zone": f"{round(ind['close'] * 0.995, 2)} - {round(ind['close'] * 1.005, 2)}",
        "sl": round(sl, 2),
        "target": round(target, 2),
        "rr": rr,
        "reasons": " | ".join(score_obj.get("reasons", [])),
        "oi_pct": "N/A",
        "results": RESULTS_CALENDAR.get(symbol_name, "-")
    }

def run_screener(min_probability: int, scan_limit: int) -> Dict[str, List[Dict]]:
    results = {
        "positional_long": [],
        "positional_short": [],
        "btst": [],
        "stbt": [],
        "intraday_long": [],
        "intraday_short": []
    }

    nifty_df = fetch_nifty_data()
    nifty_ret20 = 0.0
    if nifty_df is not None and len(nifty_df) >= 20:
        nifty_ret20 = safe_pct(float(nifty_df["close"].iloc[-1]) - float(nifty_df["close"].iloc[-20]), float(nifty_df["close"].iloc[-20]))

    symbols = NIFTY200[:scan_limit]
    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        name = symbol.replace(".NS", "")
        status.text(f"Scanning {name}...")
        progress.progress((i + 1) / len(symbols))

        df_daily = fetch_stock_data(symbol)
        ind_daily = calculate_indicators(df_daily)
        if ind_daily:
            pos = positional_score(ind_daily, nifty_ret20, name)
            if pos and pos["active_score"] >= min_probability:
                entry = make_entry(name, ind_daily, pos)
                if pos["bias"] == "LONG":
                    results["positional_long"].append(entry)
                elif pos["bias"] == "SHORT":
                    results["positional_short"].append(entry)

            bt = btst_score(ind_daily, nifty_ret20, name)
            if bt and bt["active_score"] >= min_probability:
                entry = make_entry(name, ind_daily, bt)
                if bt["bias"] == "BTST":
                    results["btst"].append(entry)
                elif bt["bias"] == "STBT":
                    results["stbt"].append(entry)

        df_intra = fetch_intraday_data(symbol)
        ind_intra = calculate_intraday_indicators(df_intra)
        if ind_intra:
            intra = intraday_score(ind_intra)
            if intra and intra["active_score"] >= min_probability:
                # Intraday target/SL uses intraday ATR, so reuse make_entry-compatible fields.
                ind_intra_table = {
                    **ind_intra,
                    "day_high": ind_intra["or_high"],
                    "day_low": ind_intra["or_low"],
                    "atr_pct": safe_pct(ind_intra.get("atr", 0), ind_intra["close"]),
                }
                entry = make_entry(name, ind_intra_table, intra)
                if intra["bias"] == "LONG":
                    results["intraday_long"].append(entry)
                elif intra["bias"] == "SHORT":
                    results["intraday_short"].append(entry)

    for key in results:
        results[key].sort(key=lambda x: x["score"], reverse=True)

    progress.empty()
    status.empty()
    return results

# ════════════════════════════════════
# UI — ONE PAGE SIGNAL BOARD + DRILL-DOWN
# ════════════════════════════════════

BOARD_CSS = """
<style>
    .stApp {background: radial-gradient(circle at top left, #10213d 0, #020617 34%, #020617 100%);}
    .main-title {text-align:center;color:#f8fafc;margin-bottom:0;font-size:2.25rem;font-weight:950;letter-spacing:-0.04em;}
    .sub-title {text-align:center;color:#94a3b8;margin-top:0.35rem;margin-bottom:1.15rem;font-size:0.96rem;}
    [data-testid="stMetric"] {
        background:linear-gradient(145deg, rgba(15,23,42,0.92), rgba(30,41,59,0.64));
        border:1px solid rgba(148,163,184,0.18);border-radius:18px;padding:12px 14px;
        box-shadow:0 12px 28px rgba(0,0,0,0.25);
    }
    .board-box {
        min-height:380px;border:1px solid rgba(148,163,184,0.22);border-radius:26px;padding:16px;
        background:linear-gradient(160deg, rgba(15,23,42,0.94), rgba(2,6,23,0.96));
        box-shadow:0 18px 42px rgba(0,0,0,0.40);position:relative;overflow:hidden;
    }
    .board-box:before {content:"";position:absolute;top:-40px;right:-40px;width:120px;height:120px;border-radius:999px;background:rgba(34,197,94,0.12);filter:blur(2px);}
    .box-title {font-size:1.22rem;font-weight:950;color:#f8fafc;margin-bottom:4px;position:relative;z-index:1;}
    .box-sub {color:#94a3b8;font-size:0.83rem;margin-bottom:14px;min-height:32px;position:relative;z-index:1;}
    .signal-chip {
        display:flex;justify-content:space-between;align-items:center;gap:10px;border:1px solid rgba(148,163,184,0.18);
        border-radius:16px;padding:10px 12px;margin:8px 0;background:rgba(15,23,42,0.78);color:#e5e7eb;font-size:0.92rem;
    }
    .chip-left {font-weight:850;letter-spacing:0.01em;}
    .chip-right {font-weight:950;color:#facc15;background:rgba(250,204,21,0.10);border:1px solid rgba(250,204,21,0.18);padding:3px 8px;border-radius:999px;}
    .empty-box {border:1px dashed rgba(148,163,184,0.34);border-radius:16px;padding:26px 12px;margin-top:24px;text-align:center;color:#94a3b8;background:rgba(15,23,42,0.40);}
    .detail-card {
        border:1px solid rgba(148,163,184,0.28);border-radius:28px;padding:20px;
        background:linear-gradient(145deg, rgba(17,24,39,0.98), rgba(2,6,23,0.99));
        box-shadow:0 22px 52px rgba(0,0,0,0.42);margin-top:10px;
    }
    .detail-title {font-size:1.55rem;font-weight:950;margin-bottom:4px;color:#f8fafc;letter-spacing:-0.03em;}
    .badge-long,.badge-short {display:inline-block;padding:5px 12px;border-radius:999px;font-weight:950;font-size:0.78rem;margin-left:8px;}
    .badge-long {background:rgba(6,78,59,0.95);color:#a7f3d0;border:1px solid rgba(34,197,94,0.25);}
    .badge-short {background:rgba(127,29,29,0.95);color:#fecaca;border:1px solid rgba(248,113,113,0.25);}
    .metric-mini {border:1px solid rgba(148,163,184,0.18);border-radius:18px;padding:13px;background:rgba(15,23,42,0.82);text-align:center;box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);}
    .metric-label {color:#94a3b8;font-size:0.76rem;margin-bottom:5px;text-transform:uppercase;letter-spacing:0.06em;}
    .metric-value {color:#f8fafc;font-size:1.08rem;font-weight:950;}
    .reason-box {border-left:4px solid #22c55e;background:rgba(34,197,94,0.08);border-radius:14px;padding:12px 14px;color:#d1fae5;margin-top:12px;line-height:1.65;}
    .small-note {color:#94a3b8;font-size:0.84rem;margin-bottom:12px;}
    .section-label {color:#cbd5e1;font-size:0.9rem;font-weight:900;margin:18px 0 8px 0;}
</style>
"""

def category_items(results: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Convert engine buckets into four board boxes."""
    board = {
        "BTST": [{**x, "category": "BTST"} for x in results.get("btst", [])],
        "STBT": [{**x, "category": "STBT"} for x in results.get("stbt", [])],
        "INTRADAY": [
            *[{**x, "category": "INTRADAY BUY"} for x in results.get("intraday_long", [])],
            *[{**x, "category": "INTRADAY SELL"} for x in results.get("intraday_short", [])],
        ],
        "POSITIONAL": [
            *[{**x, "category": "POSITIONAL LONG"} for x in results.get("positional_long", [])],
            *[{**x, "category": "POSITIONAL SHORT"} for x in results.get("positional_short", [])],
        ],
    }
    for key in board:
        board[key].sort(key=lambda x: x["score"], reverse=True)
    return board

def all_signals(board: Dict[str, List[Dict]]) -> List[Dict]:
    rows = []
    for items in board.values():
        rows.extend(items)
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows

def render_signal_box(title: str, subtitle: str, items: List[Dict], box_key: str, max_names: int = 8):
    st.markdown("<div class='board-box'>", unsafe_allow_html=True)
    st.markdown(f"<div class='box-title'>{title} <span style='color:#facc15'>({len(items)})</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='box-sub'>{subtitle}</div>", unsafe_allow_html=True)

    if not items:
        st.markdown("<div class='empty-box'>No setups above filter</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for i, item in enumerate(items[:max_names]):
        st.markdown(
            f"""
            <div class='signal-chip'>
                <span class='chip-left'>#{i+1} {item['symbol']}</span>
                <span class='chip-right'>{item['score']}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Open {item['symbol']}", key=f"open_{box_key}_{item['symbol']}_{i}", use_container_width=True):
            st.session_state["selected_signal"] = item

    if len(items) > max_names:
        st.caption(f"+ {len(items) - max_names} more setups below filter ranking")
    st.markdown("</div>", unsafe_allow_html=True)

def render_detail_panel(signal: Optional[Dict]):
    if not signal:
        st.info("Select any stock from the four boxes above to see complete information here.")
        return

    bias_text = signal.get("category", signal.get("bias", "SIGNAL"))
    is_short = any(word in bias_text for word in ["SHORT", "STBT", "SELL"])
    badge_class = "badge-short" if is_short else "badge-long"

    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='detail-title'>{signal['symbol']} <span class='{badge_class}'>{bias_text}</span></div>
        <div class='small-note'>Complete setup information based on weighted probability engine</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Probability", f"{signal['score']}%"),
        ("Price", signal.get("price", "-")),
        ("Change", f"{signal.get('change', '-')}%"),
        ("Rating", signal.get("stars", "-")),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"<div class='metric-mini'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>TRADE PLAN</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    risk_metrics = [
        ("Entry Zone", signal.get("entry_zone", "-")),
        ("Stop Loss", signal.get("sl", "-")),
        ("Target", signal.get("target", "-")),
        ("RR", f"1:{signal.get('rr', '-')}")
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], risk_metrics):
        with col:
            st.markdown(f"<div class='metric-mini'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>SETUP QUALITY</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    indicator_metrics = [
        ("RSI", signal.get("rsi", "-")),
        ("Vol vs 20D", f"{signal.get('rel_vol', '-')}x"),
        ("Vol vs 10D", f"{signal.get('vol_spike_10d', '-')}x"),
        ("Close Location", f"{signal.get('close_location_pct', '-')}%"),
        ("Body", f"{signal.get('body_pct', '-')}%"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4, c5], indicator_metrics):
        with col:
            st.markdown(f"<div class='metric-mini'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>DATA CONTEXT</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"<div class='metric-mini'><div class='metric-label'>OI%</div><div class='metric-value'>{signal.get('oi_pct', 'Not connected')}</div></div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='metric-mini'><div class='metric-label'>Results Date</div><div class='metric-value'>{signal.get('results', '-')}</div></div>", unsafe_allow_html=True)

    reasons = signal.get("reasons", "No reasons captured")
    st.markdown(f"<div class='reason-box'><b>Why selected:</b><br>{reasons}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_compact_leaderboard(signals: List[Dict]):
    if not signals:
        return
    with st.expander("View full ranked list", expanded=False):
        df = pd.DataFrame(signals)
        df = df[["category", "symbol", "score", "price", "change", "rsi", "rel_vol", "vol_spike_10d", "close_location_pct", "entry_zone", "sl", "target", "reasons"]]
        df.columns = ["Category", "Stock", "Score", "Price", "Chg%", "RSI", "Vol20D", "Vol10D", "CloseLoc%", "Entry Zone", "SL", "Target", "Why"]
        st.dataframe(df, use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title="Trade Signal Board", page_icon="📈", layout="wide")
    st.markdown(BOARD_CSS, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>📈 TRADE SIGNAL BOARD</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>One-page view: BTST • STBT • Intraday • Positional. Click a stock for full details.</p>", unsafe_allow_html=True)

    phase, phase_text, now_ist = get_market_phase()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Market Phase", phase_text)
    with m2:
        st.metric("IST Time", now_ist.strftime("%d-%b %H:%M"))
    with m3:
        st.metric("Default Filter", f"≥{MIN_PROB_DEFAULT}%")
    with m4:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.caption("Closed market mode shows last available session data. OI% and delivery% remain placeholders until a better NSE/F&O data source is added.")

    st.sidebar.header("Scanner Controls")
    min_probability = st.sidebar.slider("Minimum probability score", 55, 90, MIN_PROB_DEFAULT, 5)
    scan_limit = st.sidebar.slider("Stocks to scan", 10, len(NIFTY200), min(40, len(NIFTY200)), 5)
    auto_refresh = st.sidebar.checkbox("Auto refresh during market hours", value=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Click a stock name button inside any box to load the complete setup below the board.")

    with st.spinner("Running enhanced screener..."):
        results = run_screener(min_probability=min_probability, scan_limit=scan_limit)

    board = category_items(results)
    signals = all_signals(board)
    total_setups = len(signals)
    best_signal = signals[0] if signals else None

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Setups", total_setups)
    with s2:
        st.metric("BTST", len(board["BTST"]))
    with s3:
        st.metric("STBT", len(board["STBT"]))
    with s4:
        st.metric("Best", f"{best_signal['symbol']} {best_signal['score']}%" if best_signal else "None")

    st.divider()

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        render_signal_box("🟢 BTST", "Overnight long candidates", board["BTST"], "btst")
    with b2:
        render_signal_box("🔴 STBT", "Overnight short candidates", board["STBT"], "stbt")
    with b3:
        render_signal_box("⚡ Intraday", "Same-day buy/sell candidates", board["INTRADAY"], "intra")
    with b4:
        render_signal_box("📊 Positional", "3–20 day swing candidates", board["POSITIONAL"], "pos")

    st.divider()
    st.subheader("🔍 Selected Stock Details")
    render_detail_panel(st.session_state.get("selected_signal"))

    render_compact_leaderboard(signals)

    if auto_refresh and phase in ["OPEN", "NEAR_CLOSE"]:
        st.caption("Auto-refresh enabled: app will rerun every 5 minutes during market hours.")
        import time
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()
