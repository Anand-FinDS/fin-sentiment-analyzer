# setup_stocks.py
# ════════════════════════════════════════════════════════════
# ONE-TIME SETUP — Populate stocks table from Finnhub
# Run once: python -m setup_stocks
# ════════════════════════════════════════════════════════════
# Add these lines at the very top, before everything else
import sys
print(f"Python: {sys.version}")
print(f"Script starting...")

from config import FINNHUB_API_KEY
print(f"API Key loaded: {FINNHUB_API_KEY[:8]}...")


import finnhub
import time
from config import FINNHUB_API_KEY
from db.queries import insert_stocks, get_stock_count
from db.schema import create_tables


def fetch_nasdaq_stocks():
    client  = finnhub.Client(api_key=FINNHUB_API_KEY)
    print("Fetching stock symbols from Finnhub...")

    symbols = client.stock_symbols('US')
    print(f"Total US symbols returned : {len(symbols)}")

    # Filter to NASDAQ + NYSE common stocks
    quality = [
        s for s in symbols
        if s.get('mic') in ('XNAS', 'XNYS')    # ← both exchanges
        and s.get('type') == 'Common Stock'
    ]
    print(f"After NASDAQ + NYSE filter : {len(quality)}")

    stocks = [
        {
            'symbol'      : s['symbol'],
            'company_name': s['description'],
            'exchange'    : s.get('exchange', ''),
            'mic'         : s.get('mic', ''),
            'stock_type'  : s.get('type', 'Common Stock'),
        }
        for s in quality
        if s.get('symbol') and s.get('description')
    ]

    return stocks


def run_setup():
    # Ensure tables exist
    create_tables()

    # Check if already populated
    existing = get_stock_count()
    if existing > 0:
        print(f"✓ Stocks table already has {existing} rows — skipping fetch")
        print(f"  Delete data/financial_sentiment.db to re-run setup")
        return

    # Fetch from Finnhub
    stocks = fetch_nasdaq_stocks()

    if not stocks:
        print("✗ No stocks returned — check FINNHUB_API_KEY in config.py")
        return

    # Insert into DB
    print(f"\nInserting {len(stocks)} stocks into DB...")
    inserted = insert_stocks(stocks)
    final    = get_stock_count()

    print(f"\n── Setup complete ───────────────────────────────")
    print(f"  Stocks in DB : {final}")
    print(f"\nSample stocks:")

    # Preview first 10
    from db.queries import get_all_stocks
    df = get_all_stocks()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    run_setup()