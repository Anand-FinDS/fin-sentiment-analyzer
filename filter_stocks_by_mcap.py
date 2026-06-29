# filter_stocks_by_mcap.py — replace entire file with this
# ════════════════════════════════════════════════════════════
# Use curated S&P 500 + NASDAQ 100 list instead of mcap check
# All these are guaranteed $1B+ with good news coverage
# ════════════════════════════════════════════════════════════

import pandas as pd
from db.schema  import get_connection, get_db_stats

# S&P 500 + NASDAQ 100 curated list
QUALITY_STOCKS = [
    # Technology
    "AAPL","MSFT","NVDA","GOOGL","GOOG","META","AMZN",
    "TSLA","AMD","INTC","ORCL","CRM","ADBE","QCOM",
    "TXN","AVGO","MU","AMAT","LRCX","KLAC","SNPS",
    "CDNS","MRVL","NXPI","ON","STX","WDC","SWKS",
    "MPWR","ENPH","FSLR","SEDG",
    # Communication
    "NFLX","CMCSA","T","VZ","TMUS","DIS","CHTR",
    "PARA","WBD","FOXA","FOX","IPG","OMC",
    # Consumer
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TGT",
    "LOW","COST","WMT","BKNG","MAR","HLT","YUM",
    "CMG","ROST","TJX","ORLY","AZO","DLTR","DG",
    # Financial
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW",
    "AXP","V","MA","PYPL","COF","USB","PNC","TFC",
    "BK","STT","FITB","HBAN","RF","CFG","MTB",
    # Healthcare
    "UNH","JNJ","LLY","PFE","ABBV","MRK","TMO",
    "ABT","DHR","BMY","AMGN","GILD","REGN","VRTX",
    "ISRG","SYK","BSX","EW","ZBH","BAX","BDX",
    # Industrial
    "CAT","BA","HON","UPS","RTX","LMT","NOC","GD",
    "DE","MMM","EMR","ETN","PH","ROK","XYL","IR",
    # Energy
    "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO",
    "OXY","HAL","BKR","DVN","FANG","HES","APA",
    # Consumer Staples
    "PG","KO","PEP","PM","MO","MDLZ","CL","KMB",
    "GIS","K","CPB","SJM","CAG","MKC","HRL",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","SRE","PEG",
    "ED","ES","WEC","ETR","FE","PPL","CMS",
    # Real Estate
    "AMT","PLD","CCI","EQIX","PSA","DLR","O","SPG",
    "WELL","AVB","EQR","MAA","UDR","CPT","ESS",
    # Materials
    "LIN","APD","ECL","SHW","FCX","NEM","NUE","STLD",
    "ALB","MOS","CF","IFF","PPG","VMC","MLM",
]

# Remove duplicates
QUALITY_STOCKS = list(dict.fromkeys(QUALITY_STOCKS))

def filter_to_quality():
    print(f"Filtering stocks table to {len(QUALITY_STOCKS)} quality stocks...")

    conn   = get_connection()
    cursor = conn.cursor()

    # Keep only stocks in our curated list
    placeholders = ','.join(['?' for _ in QUALITY_STOCKS])
    cursor.execute(f"""
        DELETE FROM stocks
        WHERE symbol NOT IN ({placeholders})
    """, QUALITY_STOCKS)

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    print(f"\n── Filter complete ─────────────────────────────")
    print(f"  Deleted  : {deleted} stocks")
    print(f"  Remaining: {len(QUALITY_STOCKS)} quality stocks")
    get_db_stats()

    # Preview
    from db.queries import get_all_stocks
    df = get_all_stocks()
    print(f"\nSample stocks in DB:")
    print(df.sample(10).to_string(index=False))


if __name__ == "__main__":
    filter_to_quality()