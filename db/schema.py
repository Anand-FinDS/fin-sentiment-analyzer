# db/schema.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# SQLite schema — CREATE TABLE statements
# ════════════════════════════════════════════════════════════

import sqlite3
import os
from config import DB_PATH


def get_connection():
    """Get SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn


def create_tables():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── stocks ────────────────────────────────────────────────
    # Populated once via setup script
    # Source: Finnhub stock_symbols("US") filtered to NASDAQ
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol          TEXT PRIMARY KEY,
            company_name    TEXT NOT NULL,
            exchange        TEXT,
            mic             TEXT,
            stock_type      TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── pipeline_runs ─────────────────────────────────────────
    # One row per symbol + date range analysis
    # Acts as cache registry
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id              TEXT PRIMARY KEY,
            symbol              TEXT NOT NULL,
            company_name        TEXT NOT NULL,
            date_from           TEXT NOT NULL,
            date_to             TEXT NOT NULL,
            status              TEXT DEFAULT 'pending',
            articles_fetched    INTEGER DEFAULT 0,
            articles_relevant   INTEGER DEFAULT 0,
            articles_direct     INTEGER DEFAULT 0,
            articles_indirect   INTEGER DEFAULT 0,
            trading_days        INTEGER DEFAULT 0,
            created_at          TEXT DEFAULT (datetime('now')),
            completed_at        TEXT,
            UNIQUE(symbol, date_from, date_to)
        )
    """)

    # ── articles ──────────────────────────────────────────────
    # One row per news article
    # Populated in stages as pipeline runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                  TEXT NOT NULL,
            symbol                  TEXT NOT NULL,
            timestamp               TEXT,
            date                    TEXT,
            trading_date            TEXT,
            source                  TEXT,
            headline                TEXT,
            summary                 TEXT,
            url                     TEXT,
            is_weekend              INTEGER DEFAULT 0,

            -- Phase 1 : LLM classification
            relevance_class         TEXT,
            relevance_reason        TEXT,
            relevance_weight        REAL,

            -- Phase 2a : VADER
            vader_headline          REAL,
            vader_summary           REAL,
            vader_compound          REAL,
            vader_label             TEXT,
            vader_confidence        TEXT,
            is_clickbait            INTEGER DEFAULT 0,
            is_comparative          INTEGER DEFAULT 0,
            vader_low_conf          INTEGER DEFAULT 0,
            negated_positive        INTEGER DEFAULT 0,
            price_drop_summary      INTEGER DEFAULT 0,

            -- Phase 2b : FinBERT
            finbert_headline_label  TEXT,
            finbert_headline_score  REAL,
            finbert_summary_label   TEXT,
            finbert_summary_score   REAL,
            finbert_label           TEXT,
            finbert_compound        REAL,

            -- Phase 2c : LLM tiebreaker
            models_agree            INTEGER DEFAULT 0,
            llm_candidate           INTEGER DEFAULT 0,
            llm_label               TEXT,
            llm_score               REAL,
            llm_reason              TEXT,

            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # ── daily_sentiment ───────────────────────────────────────
    # One row per trading day per run
    # Weighted aggregate of all articles that day
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            symbol              TEXT NOT NULL,
            date                TEXT NOT NULL,
            article_count       INTEGER,
            direct_count        INTEGER,
            indirect_count      INTEGER,

            -- VADER daily
            vader_compound      REAL,
            vader_sentiment     TEXT,
            vader_positive      INTEGER,
            vader_negative      INTEGER,
            vader_neutral       INTEGER,

            -- FinBERT daily
            finbert_compound    REAL,
            finbert_sentiment   TEXT,
            finbert_positive    INTEGER,
            finbert_negative    INTEGER,
            finbert_neutral     INTEGER,

            -- LLM daily
            llm_compound        REAL,
            llm_sentiment       TEXT,
            llm_positive        INTEGER,
            llm_negative        INTEGER,
            llm_neutral         INTEGER,

            -- Quality flags
            low_conf_count      INTEGER,
            clickbait_count     INTEGER,
            llm_tiebreak_count  INTEGER,
            weekend_articles    INTEGER,

            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # ── stock_prices ──────────────────────────────────────────
    # One row per trading day per run
    # Source: yfinance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            symbol              TEXT NOT NULL,
            date                TEXT NOT NULL,
            open                REAL,
            high                REAL,
            low                 REAL,
            close               REAL,
            volume              INTEGER,
            price_change_pct    REAL,
            price_direction     TEXT,

            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # ── historical_prices ─────────────────────────────────────────
    # Fetched once when stock is selected
    # Full 1-year history — sliced per run for correlation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol              TEXT NOT NULL,
            date                TEXT NOT NULL,
            open                REAL,
            high                REAL,
            low                 REAL,
            close               REAL,
            volume              INTEGER,
            price_change_pct    REAL,
            price_direction     TEXT,
            fetched_at          TEXT DEFAULT (datetime('now')),
            UNIQUE(symbol, date)
        )
    """)



    # ── correlation_results ───────────────────────────────────
    # One row per model per run
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS correlation_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            model       TEXT NOT NULL,
            same_day    REAL,
            lag_1       REAL,
            lag_2       REAL,
            best_lag    TEXT,
            best_value  REAL,

            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    # ── findings_reports ──────────────────────────────────────
    # GPT-4o generated analyst report
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS findings_reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL UNIQUE,
            symbol      TEXT NOT NULL,
            date_from   TEXT NOT NULL,
            date_to     TEXT NOT NULL,
            report_text TEXT,
            model_used  TEXT,
            created_at  TEXT DEFAULT (datetime('now')),

            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
        )
    """)

    conn.commit()
    conn.close()
    print("✓ All tables created successfully")


def drop_tables():
    """Drop all tables — use only for dev reset."""
    conn = get_connection()
    cursor = conn.cursor()
    tables = [
        'findings_reports',
        'correlation_results',
        'stock_prices',
        'daily_sentiment',
        'articles',
        'pipeline_runs',
        'stocks',
    ]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    print("✓ All tables dropped")


def get_db_stats():
    conn = get_connection()
    cursor = conn.cursor()
    tables = [
        'stocks', 'pipeline_runs', 'articles',
        'daily_sentiment', 'stock_prices',
        'historical_prices',        
        'correlation_results', 'findings_reports'
    ]
    print("── DB Stats ─────────────────────────────────")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:<25} : {count:>6} rows")
    conn.close()


if __name__ == "__main__":
    # Run directly to initialize DB
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    create_tables()
    get_db_stats()