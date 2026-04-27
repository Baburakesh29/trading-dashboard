# ════════════════════════════════════
# INSTALL THESE FIRST
# pip install streamlit yfinance pandas 
# pip install ta requests schedule
# ════════════════════════════════════

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import schedule
import time
import datetime
from typing import Dict, List

# ════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════

# Nifty 200 stocks — add full list here
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

# ════════════════════════════════════
# DATA FETCHER
# ════════════════════════════════════

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol: str, period: str = "3mo") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, 
                        interval="1d", progress=False)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return None

@st.cache_data(ttl=300)
def fetch_intraday_data(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period="1d",
                        interval="5m", progress=False)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return None

def fetch_nifty_data() -> pd.DataFrame:
    return fetch_stock_data(NIFTY_SYMBOL)

# ════════════════════════════════════
# INDICATOR CALCULATIONS
# ════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 50:
        return None
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # EMAs
        ema20 = ta.trend.EMAIndicator(close, 20).ema_indicator()
        ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(close, 200).ema_indicator()

        # RSI
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, 20, 2)
        bb_width = (bb.bollinger_hband() - 
                   bb.bollinger_lband()) / bb.bollinger_mavg()

        # Volume
        avg_vol = volume.rolling(20).mean()
        rel_vol = volume / avg_vol

        # 20 day high/low
        high20 = high.rolling(20).max()
        low20 = low.rolling(20).min()

        # Narrow range
        high5 = high.rolling(5).max()
        low5 = low.rolling(5).min()
        narrow_range = (high5 - low5) / close < 0.04

        # Current values
        c = close.iloc[-1]
        return {
            'close': c,
            'ema20': ema20.iloc[-1],
            'ema50': ema50.iloc[-1],
            'ema200': ema200.iloc[-1],
            'rsi': rsi.iloc[-1],
            'rel_vol': rel_vol.iloc[-1],
            'avg_vol': avg_vol.iloc[-1],
            'volume': volume.iloc[-1],
            'bb_width': bb_width.iloc[-1],
            'high20': high20.iloc[-1],
            'low20': low20.iloc[-1],
            'narrow_range': narrow_range.iloc[-1],
            'change_pct': ((c - close.iloc[-2]) / 
                          close.iloc[-2] * 100),
            'day_high': high.iloc[-1],
            'day_low': low.iloc[-1]
        }
    except:
        return None

def calculate_intraday_indicators(df: pd.DataFrame) -> Dict:
    if df is None or len(df) < 20:
        return None
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        hlc3 = (high + low + close) / 3

        # VWAP
        vwap = (hlc3 * volume).cumsum() / volume.cumsum()

        # EMAs
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()

        # RSI
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        # Volume
        avg_vol = volume.rolling(20).mean()
        rel_vol = volume / avg_vol

        # Opening range (first 3 candles = 15 mins)
        or_high = high.iloc[:3].max()
        or_low = low.iloc[:3].min()

        c = close.iloc[-1]
        return {
            'close': c,
            'vwap': vwap.iloc[-1],
            'ema9': ema9.iloc[-1],
            'ema21': ema21.iloc[-1],
            'rsi': rsi.iloc[-1],
            'rel_vol': rel_vol.iloc[-1],
            'volume': volume.iloc[-1],
            'or_high': or_high,
            'or_low': or_low,
            'change_pct': ((c - close.iloc[-2]) / 
                          close.iloc[-2] * 100)
        }
    except:
        return None

# ════════════════════════════════════
# SCORING ENGINE
# ════════════════════════════════════

def positional_score(ind: Dict, nifty_ret: float) -> Dict:
    if not ind:
        return None

    c = ind['close']
    stock_ret = ind['change_pct']

    # Long conditions
    lc = {
        'EMA 20': c > ind['ema20'],
        'EMA 50': c > ind['ema50'],
        'RSI 50-70': 50 <= ind['rsi'] <= 70,
        'Rel Vol 1.5x': ind['rel_vol'] > 1.5,
        '20D High': c >= ind['high20'] * 0.98,
        'BB Squeeze': ind['bb_width'] < 0.06 or ind['narrow_range'],
        'Vol Confirm': ind['volume'] > ind['avg_vol'],
        'RS vs Nifty': stock_ret > nifty_ret
    }

    # Short conditions
    sc = {
        'EMA 20': c < ind['ema20'],
        'EMA 50': c < ind['ema50'],
        'RSI 30-50': 30 <= ind['rsi'] <= 50,
        'Rel Vol 1.5x': ind['rel_vol'] > 1.5,
        '20D Low': c <= ind['low20'] * 1.02,
        'BB Squeeze': ind['bb_width'] < 0.06 or ind['narrow_range'],
        'Vol Confirm': ind['volume'] > ind['avg_vol'],
        'RS vs Nifty': stock_ret < nifty_ret
    }

    long_score = sum(lc.values())
    short_score = sum(sc.values())

    return {
        'long_score': long_score,
        'short_score': short_score,
        'long_conditions': lc,
        'short_conditions': sc,
        'bias': 'LONG' if long_score > short_score else
                'SHORT' if short_score > long_score else 'NEUTRAL',
        'active_score': max(long_score, short_score)
    }

def btst_score(ind: Dict, nifty_ret: float) -> Dict:
    if not ind:
        return None

    c = ind['close']
    stock_ret = ind['change_pct']

    # BTST Long
    bl = {
        'EMA 20': c > ind['ema20'],
        'RSI 55-70': 55 <= ind['rsi'] <= 70,
        'Rel Vol 2x': ind['rel_vol'] > 2.0,
        'Change >1%': ind['change_pct'] > 1.0,
        'Near Day High': c >= ind['day_high'] * 0.99,
        'Above EMA50': c > ind['ema50'],
        'Vol Confirm': ind['volume'] > ind['avg_vol'],
        'RS vs Nifty': stock_ret > nifty_ret
    }

    # STBT Short
    sl = {
        'EMA 20': c < ind['ema20'],
        'RSI 30-45': 30 <= ind['rsi'] <= 45,
        'Rel Vol 2x': ind['rel_vol'] > 2.0,
        'Change <-1%': ind['change_pct'] < -1.0,
        'Near Day Low': c <= ind['day_low'] * 1.01,
        'Below EMA50': c < ind['ema50'],
        'Vol Confirm': ind['volume'] > ind['avg_vol'],
        'RS vs Nifty': stock_ret < nifty_ret
    }

    btst_score = sum(bl.values())
    stbt_score = sum(sl.values())

    return {
        'btst_score': btst_score,
        'stbt_score': stbt_score,
        'btst_conditions': bl,
        'stbt_conditions': sl,
        'bias': 'BTST' if btst_score > stbt_score else
                'STBT' if stbt_score > btst_score else 'NEUTRAL',
        'active_score': max(btst_score, stbt_score)
    }

def intraday_score(ind: Dict) -> Dict:
    if not ind:
        return None

    c = ind['close']

    # Intraday Long
    il = {
        'Above VWAP': c > ind['vwap'],
        'Above EMA9': c > ind['ema9'],
        'Above EMA21': c > ind['ema21'],
        'RSI 45-65': 45 <= ind['rsi'] <= 65,
        'Vol 2x': ind['rel_vol'] > 2.0,
        'OR Breakout': c > ind['or_high'],
        'EMA Cross': ind['ema9'] > ind['ema21'],
        'Price Action': ind['change_pct'] > 0
    }

    # Intraday Short
    is_ = {
        'Below VWAP': c < ind['vwap'],
        'Below EMA9': c < ind['ema9'],
        'Below EMA21': c < ind['ema21'],
        'RSI 35-55': 35 <= ind['rsi'] <= 55,
        'Vol 2x': ind['rel_vol'] > 2.0,
        'OR Breakdown': c < ind['or_low'],
        'EMA Cross': ind['ema9'] < ind['ema21'],
        'Price Action': ind['change_pct'] < 0
    }

    long_score = sum(il.values())
    short_score = sum(is_.values())

    return {
        'long_score': long_score,
        'short_score': short_score,
        'long_conditions': il,
        'short_conditions': is_,
        'bias': 'LONG' if long_score > short_score else
                'SHORT' if short_score > long_score else 'NEUTRAL',
        'active_score': max(long_score, short_score)
    }

# ════════════════════════════════════
# STAR RATING
# ════════════════════════════════════

def get_stars(score: int) -> str:
    if score >= 7: return "⭐⭐⭐⭐⭐"
    if score >= 6: return "⭐⭐⭐⭐"
    if score >= 5: return "⭐⭐⭐"
    if score >= 4: return "⭐⭐"
    return "⭐"

def get_probability(score: int) -> str:
    if score >= 7: return "≥70%"
    if score >= 6: return "65-69%"
    if score >= 5: return "60-64%"
    if score >= 4: return "55-59%"
    return "<55%"

def get_color(score: int, bias: str) -> str:
    if score >= 6:
        return "green" if bias in ['LONG','BTST'] else "red"
    if score >= 4:
        return "orange"
    return "gray"

# ════════════════════════════════════
# MAIN SCREENER ENGINE
# ════════════════════════════════════

def run_screener():
    results = {
        'positional_long': [],
        'positional_short': [],
        'btst': [],
        'stbt': [],
        'intraday_long': [],
        'intraday_short': []
    }

    # Get Nifty return for RS comparison
    nifty_df = fetch_nifty_data()
    nifty_ret = 0
    if nifty_df is not None:
        nifty_ret = ((nifty_df['close'].iloc[-1] -
                     nifty_df['close'].iloc[-20]) /
                    nifty_df['close'].iloc[-20] * 100)

    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(NIFTY200):
        status.text(f"Scanning {symbol}...")
        progress.progress((i + 1) / len(NIFTY200))

        # Daily data
        df_daily = fetch_stock_data(symbol)
        ind_daily = calculate_indicators(df_daily)

        # Intraday data
        df_intra = fetch_intraday_data(symbol)
        ind_intra = calculate_intraday_indicators(df_intra)

        name = symbol.replace('.NS', '')

        # Positional scoring
        pos = positional_score(ind_daily, nifty_ret)
        if pos and pos['active_score'] >= 5:
            entry = {
                'symbol': name,
                'price': round(ind_daily['close'], 2),
                'change': round(ind_daily['change_pct'], 2),
                'score': pos['active_score'],
                'stars': get_stars(pos['active_score']),
                'prob': get_probability(pos['active_score']),
                'rsi': round(ind_daily['rsi'], 1),
                'rel_vol': round(ind_daily['rel_vol'], 2),
                'bias': pos['bias']
            }
            if pos['bias'] == 'LONG':
                results['positional_long'].append(entry)
            elif pos['bias'] == 'SHORT':
                results['positional_short'].append(entry)

        # BTST/STBT scoring
        bt = btst_score(ind_daily, nifty_ret)
        if bt and bt['active_score'] >= 5:
            entry = {
                'symbol': name,
                'price': round(ind_daily['close'], 2),
                'change': round(ind_daily['change_pct'], 2),
                'score': bt['active_score'],
                'stars': get_stars(bt['active_score']),
                'prob': get_probability(bt['active_score']),
                'rsi': round(ind_daily['rsi'], 1),
                'rel_vol': round(ind_daily['rel_vol'], 2),
                'bias': bt['bias']
            }
            if bt['bias'] == 'BTST':
                results['btst'].append(entry)
            elif bt['bias'] == 'STBT':
                results['stbt'].append(entry)

        # Intraday scoring
        intra = intraday_score(ind_intra)
        if intra and intra['active_score'] >= 5:
            entry = {
                'symbol': name,
                'price': round(ind_intra['close'], 2),
                'change': round(ind_intra['change_pct'], 2),
                'score': intra['active_score'],
                'stars': get_stars(intra['active_score']),
                'prob': get_probability(intra['active_score']),
                'rsi': round(ind_intra['rsi'], 1),
                'rel_vol': round(ind_intra['rel_vol'], 2),
                'bias': intra['bias']
            }
            if intra['bias'] == 'LONG':
                results['intraday_long'].append(entry)
            elif intra['bias'] == 'SHORT':
                results['intraday_short'].append(entry)

    # Sort by score
    for key in results:
        results[key].sort(key=lambda x: x['score'],
                         reverse=True)

    progress.empty()
    status.empty()
    return results

# ════════════════════════════════════
# STREAMLIT DASHBOARD UI
# ════════════════════════════════════

def render_table(data: List[Dict], bias_type: str):
    if not data:
        st.info(f"No {bias_type} setups found")
        return

    df = pd.DataFrame(data)
    df = df[['symbol','price','change','stars',
             'prob','rsi','rel_vol','score']]
    df.columns = ['Stock','Price','Chg%','Rating',
                  'Prob','RSI','RelVol','Score']

    def color_rows(row):
        if row['Score'] >= 7:
            color = 'background-color: #1a5c1a'
        elif row['Score'] >= 6:
            color = 'background-color: #2d8c2d'
        elif row['Score'] >= 5:
            color = 'background-color: #8c6d00'
        else:
            color = ''
        return [color] * len(row)

    styled = df.style.apply(color_rows, axis=1)
    st.dataframe(styled, use_container_width=True,
                hide_index=True)

def main():
    st.set_page_config(
        page_title="Trade Dashboard",
        page_icon="📈",
        layout="wide"
    )

    # Header
    st.markdown("""
        <h1 style='text-align:center; color:#00ff88'>
        📈 POSITIONAL TRADE DASHBOARD
        </h1>
    """, unsafe_allow_html=True)

    # Market status
    now = datetime.datetime.now()
    market_open = (now.hour >= 9 and now.minute >= 15 and
                  now.hour < 15 or
                  (now.hour == 15 and now.minute <= 30))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Status",
                 "🟢 OPEN" if market_open else "🔴 CLOSED")
    with col2:
        st.metric("Last Updated", now.strftime("%H:%M:%S"))
    with col3:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()

    st.divider()

    # Auto refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)",
                                       value=True)
    min_stars = st.sidebar.slider("Min Score Filter", 
                                   3, 8, 5)

    # Run screener
    with st.spinner("Running screener on Nifty 200..."):
        results = run_screener()

    # Filter by minimum score
    for key in results:
        results[key] = [x for x in results[key]
                       if x['score'] >= min_stars]

    # ── POSITIONAL ──────────────────────────────
    st.subheader("📊 POSITIONAL TRADES (0-20 Days)")
    col1, col2 = st.columns(2)

    with col1:
        count = len(results['positional_long'])
        st.markdown(f"### 🟢 LONG ({count} setups)")
        render_table(results['positional_long'], "Long")

    with col2:
        count = len(results['positional_short'])
        st.markdown(f"### 🔴 SHORT ({count} setups)")
        render_table(results['positional_short'], "Short")

    st.divider()

    # ── BTST/STBT ───────────────────────────────
    st.subheader("🌙 BTST / STBT (Overnight)")
    col1, col2 = st.columns(2)

    with col1:
        count = len(results['btst'])
        st.markdown(f"### 🟢 BTST ({count} setups)")
        render_table(results['btst'], "BTST")

    with col2:
        count = len(results['stbt'])
        st.markdown(f"### 🔴 STBT ({count} setups)")
        render_table(results['stbt'], "STBT")

    st.divider()

    # ── INTRADAY ────────────────────────────────
    st.subheader("⚡ INTRADAY (Same Day)")
    col1, col2 = st.columns(2)

    with col1:
        count = len(results['intraday_long'])
        st.markdown(f"### 🟢 BUY ({count} setups)")
        render_table(results['intraday_long'], "Intraday Long")

    with col2:
        count = len(results['intraday_short'])
        st.markdown(f"### 🔴 SELL ({count} setups)")
        render_table(results['intraday_short'], "Intraday Short")

    st.divider()

    # Auto refresh
    if auto_refresh and market_open:
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    main()