# pipeline/fetch.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Phase 1 — News fetch from Finnhub
# Input  : symbol, date_from, date_to
# Output : df_news (raw articles with trading dates)
# ════════════════════════════════════════════════════════════

import time
import finnhub
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from config import FINNHUB_API_KEY


# ── Finnhub client (singleton) ────────────────────────────────
_finnhub_client = None

def get_finnhub_client():
    global _finnhub_client
    if _finnhub_client is None:
        _finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    return _finnhub_client


# ── NYSE calendar (singleton) ────────────────────────────────
_trading_days_cache = {}

def get_trading_days(date_from: str, date_to: str) -> set:
    """
    Get NYSE trading days for date range.
    Cached to avoid repeated calendar lookups.
    """
    key = f"{date_from}_{date_to}"
    if key not in _trading_days_cache:
        # Buffer +7 days so weekend articles near
        # date_to always find a valid next trading day
        buffer_end = (
            pd.Timestamp(date_to) + pd.Timedelta(days=7)
        ).strftime('%Y-%m-%d')

        nyse     = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(
            start_date=date_from,
            end_date=buffer_end
        )
        _trading_days_cache[key] = set(schedule.index.date)

    return _trading_days_cache[key]


def roll_to_trading_day(date, trading_days: set):
    """Roll weekend/holiday date forward to next trading day."""
    d = pd.Timestamp(date)
    while d.date() not in trading_days:
        d += pd.Timedelta(days=1)
    return d.date()


# ── Core fetch function ───────────────────────────────────────
def fetch_news(symbol: str,
               date_from: str,
               date_to: str,
               progress_callback=None) -> pd.DataFrame:
    """
    Fetch news articles from Finnhub for a symbol + date range.
    Uses daily chunked fetch to stay under free tier cap.

    Args:
        symbol          : stock ticker e.g. 'AAPL'
        date_from       : 'YYYY-MM-DD'
        date_to         : 'YYYY-MM-DD'
        progress_callback: optional function(current, total, msg)
                          for Streamlit progress bar

    Returns:
        df_news: cleaned DataFrame with trading dates assigned
    """
    client = get_finnhub_client()

    # ── Daily chunked fetch ───────────────────────────────────
    all_articles = []
    current      = datetime.strptime(date_from, '%Y-%m-%d')
    end          = datetime.strptime(date_to,   '%Y-%m-%d')
    total_days   = (end - current).days

    print(f"Fetching news for {symbol} "
          f"({date_from} → {date_to}, ~{total_days} days)...")

    day_count = 0
    while current < end:
        chunk_end  = current + timedelta(days=1)
        chunk_from = current.strftime('%Y-%m-%d')
        chunk_to   = min(chunk_end, end).strftime('%Y-%m-%d')

        try:
            chunk = client.company_news(
                symbol,
                _from=chunk_from,
                to=chunk_to
            )
            all_articles.extend(chunk)
        except Exception as e:
            print(f"  ⚠ Error fetching {chunk_from}: {e}")

        day_count += 1
        if progress_callback:
            progress_callback(
                day_count, total_days,
                f"Fetching {chunk_from}... ({len(all_articles)} articles)"
            )

        current = chunk_end
        time.sleep(0.5)

    print(f"Raw articles fetched : {len(all_articles)}")

    if not all_articles:
        print(f"⚠ No articles found for {symbol} "
              f"in {date_from} → {date_to}")
        return pd.DataFrame()

    # ── Build DataFrame ───────────────────────────────────────
    df = pd.DataFrame([{
        'symbol'   : symbol,
        'Timestamp': datetime.fromtimestamp(
                         a['datetime']
                     ).strftime('%Y-%m-%d %H:%M'),
        'Date'     : datetime.fromtimestamp(
                         a['datetime']
                     ).date(),
        'Source'   : a.get('source',   ''),
        'Headline' : a.get('headline', ''),
        'Summary'  : a.get('summary',  ''),
        'URL'      : a.get('url',      ''),
    } for a in all_articles])

    # ── Deduplicate ───────────────────────────────────────────
    before = len(df)
    df     = df.drop_duplicates(subset='Headline').reset_index(drop=True)
    print(f"After dedup          : {len(df)}  "
          f"(removed {before - len(df)} duplicates)")

    # ── Date range filter ─────────────────────────────────────
    cutoff_start = pd.Timestamp(date_from).date()
    cutoff_end   = pd.Timestamp(date_to).date()
    df = df[
        (df['Date'] >= cutoff_start) &
        (df['Date'] <= cutoff_end)
    ].reset_index(drop=True)
    print(f"After date filter    : {len(df)}")

    # ── Trading date assignment ───────────────────────────────
    trading_days = get_trading_days(date_from, date_to)

    df['Is_Weekend']   = df['Date'].apply(
        lambda d: pd.Timestamp(d).weekday() >= 5
    )
    df['Trading_Date'] = df['Date'].apply(
        lambda d: roll_to_trading_day(d, trading_days)
    )

    # ── Sort ──────────────────────────────────────────────────
    df = df.sort_values(
        ['Trading_Date', 'Timestamp']
    ).reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n── Fetch complete ──────────────────────────────")
    print(f"  Symbol        : {symbol}")
    print(f"  Articles      : {len(df)}")
    print(f"  Date range    : {df['Date'].min()} → "
          f"{df['Date'].max()}")
    print(f"  Trading days  : "
          f"{df['Trading_Date'].nunique()}")
    print(f"\n  Articles per trading day (sample):")
    print(df.groupby('Trading_Date')['Headline']
            .count()
            .head(5)
            .to_string())

    return df
