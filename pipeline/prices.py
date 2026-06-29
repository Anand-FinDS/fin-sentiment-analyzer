# pipeline/prices.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Stock Price Data — yfinance
# Two modes:
#   1. fetch_historical() — 1 year on stock selection
#      saved to historical_prices table
#   2. get_price_window() — slice from DB for correlation
#      no API call needed
# ════════════════════════════════════════════════════════════

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from db.schema  import get_connection


# ════════════════════════════════════════════════════════════
# MODE 1 — Fetch + cache 1 year of price history
# Called once when user selects a stock
# ════════════════════════════════════════════════════════════

def fetch_historical(symbol: str,
                     years: int = 1,
                     force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch 1 year of daily price history from yfinance.
    Saves to historical_prices table.
    Skips fetch if data already exists (unless force_refresh).

    Args:
        symbol        : stock ticker e.g. 'AAPL'
        years         : how many years of history (default 1)
        force_refresh : re-fetch even if data exists

    Returns:
        df with full price history
    """
    # ── Check if already cached ───────────────────────────────
    if not force_refresh:
        existing = get_historical_from_db(symbol)
        if not existing.empty:
            print(f"✓ Price history loaded from cache "
                  f"({len(existing)} days)")
            return existing

    # ── Fetch from yfinance ───────────────────────────────────
    date_to   = datetime.now()
    date_from = date_to - timedelta(days=365 * years)

    # yfinance end date exclusive — add 1 day
    end = (date_to + timedelta(days=1)).strftime('%Y-%m-%d')
    start = date_from.strftime('%Y-%m-%d')

    print(f"Fetching {years}yr price history for "
          f"{symbol} ({start} → {date_to.strftime('%Y-%m-%d')})...")

    ticker = yf.Ticker(symbol)
    df     = ticker.history(start=start, end=end)

    if df.empty:
        print(f"⚠ No price data returned for {symbol}")
        return pd.DataFrame()

    # ── Clean + compute returns ───────────────────────────────
    df = df.reset_index()[
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)
    df         = df.sort_values('date').reset_index(drop=True)

    # Daily return %
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['price_direction']  = df['price_change_pct'].apply(
        lambda x: 'Up'   if x > 0.1
        else     ('Down'  if x < -0.1
        else      'Flat')
    )
    df['symbol'] = symbol

    # ── Save to DB ────────────────────────────────────────────
    save_historical_to_db(symbol, df)

    print(f"✓ {len(df)} trading days fetched and cached")
    print(f"  Range : {df['date'].min()} → {df['date'].max()}")
    print(f"  Close : ${df['close'].iloc[0]:.2f} → "
          f"${df['close'].iloc[-1]:.2f}")

    return df


def save_historical_to_db(symbol: str, df: pd.DataFrame):
    """Upsert price history into historical_prices table."""
    conn   = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO historical_prices
                (symbol, date, open, high, low, close,
                 volume, price_change_pct, price_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            str(row['date']),
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            int(row['volume']),
            row['price_change_pct'],
            row['price_direction'],
        ))

    conn.commit()
    conn.close()


def get_historical_from_db(symbol: str) -> pd.DataFrame:
    """Load full price history for symbol from DB."""
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT symbol, date, open, high, low, close,
               volume, price_change_pct, price_direction
        FROM historical_prices
        WHERE symbol = ?
        ORDER BY date
    """, conn, params=[symbol])
    conn.close()
    return df


def needs_refresh(symbol: str, max_age_days: int = 1) -> bool:
    """
    Check if price history needs refresh.
    Returns True if data is older than max_age_days
    or doesn't exist.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(fetched_at) FROM historical_prices
        WHERE symbol = ?
    """, (symbol,))
    row = cursor.fetchone()
    conn.close()

    if not row or not row[0]:
        return True

    fetched_at = datetime.fromisoformat(row[0])
    age_days   = (datetime.now() - fetched_at).days
    return age_days >= max_age_days


# ════════════════════════════════════════════════════════════
# MODE 2 — Slice price window for correlation
# Called after news sentiment is ready
# No API call — reads from DB cache
# ════════════════════════════════════════════════════════════

def get_price_window(symbol: str,
                     date_from: str,
                     date_to: str) -> pd.DataFrame:
    """
    Slice cached price history for a specific date range.
    Used for correlation analysis after sentiment scoring.

    Args:
        symbol    : stock ticker
        date_from : 'YYYY-MM-DD' start of news window
        date_to   : 'YYYY-MM-DD' end of news window

    Returns:
        df with price data for exact date range
    """
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT symbol, date, open, high, low, close,
               volume, price_change_pct, price_direction
        FROM historical_prices
        WHERE symbol = ?
          AND date  >= ?
          AND date  <= ?
        ORDER BY date
    """, conn, params=[symbol, date_from, date_to])
    conn.close()

    if df.empty:
        print(f"⚠ No price data in DB for {symbol} "
              f"{date_from} → {date_to}")
        print(f"  Run fetch_historical('{symbol}') first")
        return pd.DataFrame()

    print(f"✓ Price window loaded : {len(df)} trading days")
    print(f"  Range : {df['date'].min()} → {df['date'].max()}")

    return df


# ════════════════════════════════════════════════════════════
# QUICK STOCK INFO — for UI display on stock selection
# ════════════════════════════════════════════════════════════

def get_stock_info(symbol: str) -> dict:
    """
    Fetch quick stock info for UI display.
    Called when user selects stock from dropdown.
    Returns: name, sector, market cap, current price,
             52w high/low, 2-line company description
    """
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info

        # Current price
        hist  = ticker.history(period='2d')
        price = hist['Close'].iloc[-1] if not hist.empty else None
        prev  = hist['Close'].iloc[-2] if len(hist) > 1 else None
        chg   = ((price - prev) / prev * 100
                 if price and prev else None)

        return {
            'symbol'     : symbol,
            'name'       : info.get('longName', symbol),
            'sector'     : info.get('sector', 'N/A'),
            'industry'   : info.get('industry', 'N/A'),
            'market_cap' : info.get('marketCap'),
            'price'      : price,
            'change_pct' : chg,
            'week52_high': info.get('fiftyTwoWeekHigh'),
            'week52_low' : info.get('fiftyTwoWeekLow'),
            'description': info.get(
                'longBusinessSummary', ''
            )[:300],   # 2-3 lines
            'employees'  : info.get('fullTimeEmployees'),
            'website'    : info.get('website', ''),
        }
    except Exception as e:
        print(f"⚠ Could not fetch info for {symbol}: {e}")
        return {
            'symbol': symbol,
            'name'  : symbol,
            'error' : str(e)
        }


def format_market_cap(mcap: float) -> str:
    """Format market cap for display."""
    if not mcap:
        return 'N/A'
    if mcap >= 1e12:
        return f"${mcap/1e12:.2f}T"
    if mcap >= 1e9:
        return f"${mcap/1e9:.2f}B"
    return f"${mcap/1e6:.2f}M"