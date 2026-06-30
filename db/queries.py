# db/queries.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# SQLite query helpers — all DB reads and writes go here
# ════════════════════════════════════════════════════════════

import sqlite3
import uuid
import pandas as pd
from datetime import datetime
from db.schema import get_connection


# ════════════════════════════════════════════════════════════
# STOCKS
# ════════════════════════════════════════════════════════════

def insert_stocks(stocks: list[dict]):
    """
    Bulk insert stock universe.
    Called once during setup.
    stocks = [{'symbol', 'company_name', 'exchange', 'mic', 'stock_type'}, ...]
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO stocks
            (symbol, company_name, exchange, mic, stock_type)
        VALUES
            (:symbol, :company_name, :exchange, :mic, :stock_type)
    """, stocks)
    conn.commit()
    inserted = cursor.rowcount
    conn.close()
    return inserted


def get_all_stocks() -> pd.DataFrame:
    """Return full stock list for UI dropdown."""
    conn = get_connection()
    df = pd.read_sql("""
        SELECT symbol, company_name, exchange, mic
        FROM stocks
        ORDER BY symbol
    """, conn)
    conn.close()
    return df


def stock_exists(symbol: str) -> bool:
    """Check if symbol exists in stocks table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM stocks WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def get_stock_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stocks")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# ════════════════════════════════════════════════════════════
# PIPELINE RUNS
# ════════════════════════════════════════════════════════════
def create_run(symbol: str, company_name: str,
               date_from: str, date_to: str) -> str:
    run_id = str(uuid.uuid4())
    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO pipeline_runs
                (run_id, symbol, company_name, date_from, date_to, status)
            VALUES (?, ?, ?, ?, ?, 'running')
        """, (run_id, symbol, company_name, date_from, date_to))
        conn.commit()
    except Exception:
        # Run already exists — get existing run_id
        cursor.execute("""
            SELECT run_id FROM pipeline_runs
            WHERE symbol=? AND date_from=? AND date_to=?
        """, (symbol, date_from, date_to))
        row = cursor.fetchone()
        if row:
            run_id = row[0]
    conn.close()
    return run_id


def get_cached_run(symbol: str,
                   date_from: str, date_to: str) -> dict | None:
    """
    Check if a completed run exists for this symbol + date range.
    Returns run dict or None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM pipeline_runs
        WHERE symbol   = ?
          AND date_from = ?
          AND date_to   = ?
          AND status    = 'complete'
        ORDER BY created_at DESC
        LIMIT 1
    """, (symbol, date_from, date_to))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_run_status(run_id: str, status: str, **kwargs):
    """
    Update run status and optional fields.
    kwargs: articles_fetched, articles_relevant, trading_days etc.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    fields = ["status = ?"]
    values = [status]

    if status == 'complete':
        fields.append("completed_at = ?")
        values.append(datetime.now().isoformat())

    for key, val in kwargs.items():
        fields.append(f"{key} = ?")
        values.append(val)

    values.append(run_id)
    cursor.execute(f"""
        UPDATE pipeline_runs
        SET {', '.join(fields)}
        WHERE run_id = ?
    """, values)
    conn.commit()
    conn.close()


def get_all_runs() -> pd.DataFrame:
    """Return all runs — for UI history view."""
    conn = get_connection()
    df = pd.read_sql("""
        SELECT run_id, symbol, company_name, date_from, date_to,
               status, articles_fetched, articles_relevant,
               trading_days, created_at, completed_at
        FROM pipeline_runs
        ORDER BY created_at DESC
    """, conn)
    conn.close()
    return df


# ════════════════════════════════════════════════════════════
# ARTICLES
# ════════════════════════════════════════════════════════════
def insert_articles(run_id: str, df: pd.DataFrame):
    """
    Insert fully scored articles into articles table.
    Called after complete pipeline (classify + sentiment).
    """
    df = df.copy()
    df['run_id'] = run_id

    # Normalize all columns to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Column mapping — pipeline names → DB schema names
    col_map = {
        'llm_sentiment'          : 'llm_label',
        'vader_score_gap'        : 'vader_low_conf',
        'price_drop_summary'     : 'price_drop_summary',
    }
    df = df.rename(columns=col_map)

    # All schema columns we want to save
    schema_cols = [
        'run_id', 'symbol', 'timestamp', 'date',
        'trading_date', 'source', 'headline', 'summary',
        'url', 'is_weekend',
        # Phase 1 — classification
        'relevance_class', 'relevance_reason', 'relevance_weight',
        # Phase 2a — VADER
        'vader_headline', 'vader_summary', 'vader_compound',
        'vader_label', 'vader_confidence', 'is_clickbait',
        'is_comparative', 'vader_low_conf', 'negated_positive',
        'price_drop_summary',
        # Phase 2b — FinBERT
        'finbert_headline_label', 'finbert_headline_score',
        'finbert_summary_label', 'finbert_summary_score',
        'finbert_label', 'finbert_compound',
        # Phase 2c — LLM
        'llm_label', 'llm_score', 'llm_reason',
    ]

    # Keep only columns that exist in df
    df = df[[c for c in schema_cols if c in df.columns]]

    conn = get_connection()
    df.to_sql('articles', conn, if_exists='append', index=False)
    conn.close()
    print(f"  ✓ {len(df)} articles inserted with full scoring")


def update_articles_classification(run_id: str, df: pd.DataFrame):
    """Update relevance classification columns after Groq classify."""
    conn   = get_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            UPDATE articles
            SET relevance_class  = ?,
                relevance_reason = ?,
                relevance_weight = ?
            WHERE run_id  = ?
              AND headline = ?
        """, (
            row.get('relevance_class'),
            row.get('relevance_reason'),
            row.get('relevance_weight'),
            run_id,
            row.get('Headline') or row.get('headline'),
        ))
    conn.commit()
    conn.close()


def update_articles_sentiment(run_id: str, df: pd.DataFrame):
    """Update all sentiment columns after VADER + FinBERT + LLM."""
    conn   = get_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            UPDATE articles
            SET vader_headline         = ?,
                vader_summary          = ?,
                vader_compound         = ?,
                vader_label            = ?,
                vader_confidence       = ?,
                is_clickbait           = ?,
                is_comparative         = ?,
                vader_low_conf         = ?,
                negated_positive       = ?,
                price_drop_summary     = ?,
                finbert_headline_label = ?,
                finbert_headline_score = ?,
                finbert_summary_label  = ?,
                finbert_summary_score  = ?,
                finbert_label          = ?,
                finbert_compound       = ?,
                models_agree           = ?,
                llm_candidate          = ?,
                llm_label              = ?,
                llm_score              = ?,
                llm_reason             = ?
            WHERE run_id  = ?
              AND headline = ?
        """, (
            row.get('vader_headline'),
            row.get('vader_summary'),
            row.get('vader_compound'),
            row.get('vader_label'),
            row.get('vader_confidence'),
            int(row.get('is_clickbait', 0)),
            int(row.get('is_comparative', 0)),
            int(row.get('vader_low_conf', 0)),
            int(row.get('negated_positive', 0)),
            int(row.get('price_drop_summary', 0)),
            row.get('finbert_headline_label'),
            row.get('finbert_headline_score'),
            row.get('finbert_summary_label'),
            row.get('finbert_summary_score'),
            row.get('finbert_label'),
            row.get('finbert_compound'),
            int(row.get('models_agree', 0)),
            int(row.get('llm_candidate', 0)),
            row.get('llm_label'),
            row.get('llm_score'),
            row.get('llm_reason'),
            run_id,
            row.get('Headline') or row.get('headline'),
        ))
    conn.commit()
    conn.close()


def get_articles(run_id: str,
                 relevance_filter: str = None) -> pd.DataFrame:
    """
    Load articles for a run.
    relevance_filter: 'Direct', 'Indirect', or None (all)
    """
    conn  = get_connection()
    query = "SELECT * FROM articles WHERE run_id = ?"
    params = [run_id]

    if relevance_filter:
        query  += " AND relevance_class = ?"
        params.append(relevance_filter)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

def get_relevant_articles(run_id: str) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT *
        FROM articles
        WHERE run_id = ?
          AND relevance_class IS NOT NULL
          AND relevance_class != 'Irrelevant'
        ORDER BY trading_date, timestamp
    """, conn, params=[run_id])
    conn.close()

    # Rename llm_label → llm_sentiment for UI consistency
    if 'llm_label' in df.columns:
        df = df.rename(columns={'llm_label': 'llm_sentiment'})

    return df

# ════════════════════════════════════════════════════════════
# DAILY SENTIMENT
# ════════════════════════════════════════════════════════════

def insert_daily_sentiment(run_id: str, df: pd.DataFrame):
    """Insert daily sentiment aggregate."""
    df = df.copy()
    df['run_id'] = run_id
    conn = get_connection()
    df.to_sql('daily_sentiment', conn, if_exists='append', index=False)
    conn.close()
    print(f"  ✓ {len(df)} trading days inserted")


def get_daily_sentiment(run_id: str) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT * FROM daily_sentiment
        WHERE run_id = ?
        ORDER BY date
    """, conn, params=[run_id])
    conn.close()
    return df


# ════════════════════════════════════════════════════════════
# STOCK PRICES
# ════════════════════════════════════════════════════════════

def insert_stock_prices(run_id: str, df: pd.DataFrame):
    """Insert yfinance stock prices."""
    df = df.copy()
    df['run_id'] = run_id
    conn = get_connection()
    df.to_sql('stock_prices', conn, if_exists='append', index=False)
    conn.close()
    print(f"  ✓ {len(df)} price rows inserted")


def get_stock_prices(run_id: str) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT * FROM stock_prices
        WHERE run_id = ?
        ORDER BY date
    """, conn, params=[run_id])
    conn.close()
    return df


# ════════════════════════════════════════════════════════════
# CORRELATION RESULTS
# ════════════════════════════════════════════════════════════

def insert_correlation(run_id: str, symbol: str, results: dict):
    conn   = get_connection()
    cursor = conn.cursor()
    for model, vals in results.items():
        # Only use numeric keys for best_lag calculation
        numeric_keys = ['same_day', 'lag_1', 'lag_2']
        best_lag  = max(
            numeric_keys,
            key=lambda k: abs(vals[k])
        )
        best_value = vals[best_lag]

        cursor.execute("""
            INSERT INTO correlation_results
                (run_id, symbol, model,
                 same_day, lag_1, lag_2,
                 best_lag, best_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, symbol, model,
            vals['same_day'], vals['lag_1'], vals['lag_2'],
            best_lag, best_value
        ))
    conn.commit()
    conn.close()
    print(f"  ✓ Correlation results inserted for {len(results)} models")


def get_correlation(run_id: str) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT model, same_day, lag_1, lag_2, best_lag, best_value
        FROM correlation_results
        WHERE run_id = ?
        ORDER BY model
    """, conn, params=[run_id])
    conn.close()
    return df


# ════════════════════════════════════════════════════════════
# FINDINGS REPORT
# ════════════════════════════════════════════════════════════

def insert_findings(run_id: str, symbol: str,
                    date_from: str, date_to: str,
                    report_text: str, model_used: str):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO findings_reports
            (run_id, symbol, date_from, date_to,
             report_text, model_used)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_id, symbol, date_from, date_to,
          report_text, model_used))
    conn.commit()
    conn.close()
    print(f"  ✓ Findings report saved")


def get_findings(run_id: str) -> str | None:
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT report_text FROM findings_reports
        WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    conn.close()
    return row['report_text'] if row else None


# ════════════════════════════════════════════════════════════
# MERGED VIEW — daily sentiment + prices joined
# ════════════════════════════════════════════════════════════

def get_merged(run_id: str) -> pd.DataFrame:
    """
    Join daily_sentiment + stock_prices on date.
    Returns the merged dataframe for correlation + charts.
    """
    conn = get_connection()
    df   = pd.read_sql("""
        SELECT
            d.date,
            d.article_count, d.direct_count, d.indirect_count,
            d.vader_compound, d.vader_sentiment,
            d.finbert_compound, d.finbert_sentiment,
            d.llm_compound, d.llm_sentiment,
            d.vader_positive, d.vader_negative, d.vader_neutral,
            d.finbert_positive, d.finbert_negative, d.finbert_neutral,
            d.llm_positive, d.llm_negative, d.llm_neutral,
            p.open, p.high, p.low, p.close, p.volume,
            p.price_change_pct, p.price_direction
        FROM daily_sentiment d
        JOIN stock_prices p
          ON d.date   = p.date
         AND d.run_id = p.run_id
        WHERE d.run_id = ?
        ORDER BY d.date
    """, conn, params=[run_id])
    conn.close()
    return df


if __name__ == "__main__":
    # Quick test — print all table counts
    from db.schema import get_db_stats
    get_db_stats()

def get_historical_prices(symbol: str,
                          date_from: str = None,
                          date_to: str = None) -> pd.DataFrame:
    """
    Load historical prices from DB.
    Optional date filter for correlation window.
    """
    conn  = get_connection()
    query = """
        SELECT * FROM historical_prices
        WHERE symbol = ?
    """
    params = [symbol]

    if date_from:
        query  += " AND date >= ?"
        params.append(date_from)
    if date_to:
        query  += " AND date <= ?"
        params.append(date_to)

    query += " ORDER BY date"
    df     = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def price_history_exists(symbol: str) -> bool:
    """Check if price history exists for symbol."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM historical_prices
        WHERE symbol = ?
    """, (symbol,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def insert_buzz_correlation(run_id: str, symbol: str, buzz_result: dict):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO buzz_correlation
            (run_id, symbol, trading_days, correlation,
             correlation_direct, interpretation)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_id, symbol, buzz_result['n'],
          buzz_result['correlation'],
          buzz_result['correlation_direct'],
          buzz_result['interpretation']))
    conn.commit()
    conn.close()


def get_buzz_correlation(run_id: str) -> dict:
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT trading_days, correlation, correlation_direct, interpretation
        FROM buzz_correlation WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'n': row[0], 'correlation': row[1],
                'correlation_direct': row[2], 'interpretation': row[3]}
    return None