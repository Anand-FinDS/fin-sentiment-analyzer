# pipeline/correlate.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Phase 4 — Daily Aggregate + Correlation Analysis
# Input  : df_relevant (scored articles), price window
# Output : df_daily, df_merged, correlation_results
# ════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from config import LAG_WINDOWS, RELEVANCE_WEIGHTS


# ════════════════════════════════════════════════════════════
# PHASE 4a — DAILY AGGREGATE
# ════════════════════════════════════════════════════════════

def sentiment_label(score: float) -> str:
    if score >=  0.05: return 'Positive'
    if score <= -0.05: return 'Negative'
    return 'Neutral'


def weighted_avg(group: pd.DataFrame, col: str) -> float:
    """Weighted average using relevance_weight column."""
    weights = group['relevance_weight']
    values  = group[col]
    if weights.sum() == 0:
        return values.mean()
    return np.average(values, weights=weights)


def build_daily_sentiment(df_relevant: pd.DataFrame,
                          symbol: str,
                          run_id: str) -> pd.DataFrame:
    """
    Aggregate article-level sentiment to daily level.
    Weighted by relevance_weight (Direct=1.0, Indirect=0.5).

    Returns df_daily with one row per trading day.
    """
    if df_relevant.empty:
        print("⚠ Empty df_relevant — cannot aggregate")
        return pd.DataFrame()

    print(f"\nAggregating to daily sentiment...")

    df_daily = (
        df_relevant.groupby('Trading_Date')
        .apply(lambda g: pd.Series({
            'run_id'          : run_id,
            'symbol'          : symbol,
            'article_count'   : len(g),
            'direct_count'    : (g['relevance_class'] == 'Direct').sum(),
            'indirect_count'  : (g['relevance_class'] == 'Indirect').sum(),

            # VADER — weighted
            'vader_compound'  : weighted_avg(g, 'vader_compound'),
            'vader_positive'  : (g['vader_label'] == 'Positive').sum(),
            'vader_negative'  : (g['vader_label'] == 'Negative').sum(),
            'vader_neutral'   : (g['vader_label'] == 'Neutral').sum(),

            # FinBERT — weighted
            'finbert_compound': weighted_avg(g, 'finbert_compound'),
            'finbert_positive': (g['finbert_label'] == 'Positive').sum(),
            'finbert_negative': (g['finbert_label'] == 'Negative').sum(),
            'finbert_neutral' : (g['finbert_label'] == 'Neutral').sum(),

            # LLM — weighted
            'llm_compound'    : weighted_avg(g, 'llm_score'),
            'llm_positive'    : (g['llm_sentiment'] == 'Positive').sum(),
            'llm_negative'    : (g['llm_sentiment'] == 'Negative').sum(),
            'llm_neutral'     : (g['llm_sentiment'] == 'Neutral').sum(),

            # Quality flags
            'low_conf_count'  : g.get('vader_low_conf',
                                pd.Series([0]*len(g))).sum(),
            'clickbait_count' : g.get('is_clickbait',
                                pd.Series([0]*len(g))).sum(),
            'llm_tiebreak_count': 0,  # no tiebreaker in new pipeline
            'weekend_articles': g['Is_Weekend'].sum(),
        }))
        .reset_index()
        .rename(columns={'Trading_Date': 'date'})
    )

    # Add sentiment labels
    df_daily['vader_sentiment']   = df_daily['vader_compound'].apply(
        sentiment_label)
    df_daily['finbert_sentiment'] = df_daily['finbert_compound'].apply(
        sentiment_label)
    df_daily['llm_sentiment']     = df_daily['llm_compound'].apply(
        sentiment_label)

    # Ensure date is string for DB storage
    df_daily['date'] = df_daily['date'].astype(str)

    print(f"── Daily aggregate complete ────────────────────")
    print(f"  Trading days : {len(df_daily)}")
    print(f"  Date range   : {df_daily['date'].min()} → "
          f"{df_daily['date'].max()}")
    print(f"\n  Daily preview:")
    print(df_daily[[
        'date', 'article_count', 'direct_count',
        'vader_compound', 'finbert_compound', 'llm_compound',
        'vader_sentiment', 'finbert_sentiment', 'llm_sentiment'
    ]].to_string(index=False))

    return df_daily


# ════════════════════════════════════════════════════════════
# PHASE 4b — MERGE SENTIMENT + PRICES
# ════════════════════════════════════════════════════════════

def merge_sentiment_prices(df_daily: pd.DataFrame,
                           df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sentiment with stock prices on date.
    Returns df_merged for correlation analysis.
    """
    if df_daily.empty or df_prices.empty:
        print("⚠ Cannot merge — empty DataFrame")
        return pd.DataFrame()

    # Ensure date columns are strings
    df_daily  = df_daily.copy()
    df_prices = df_prices.copy()
    df_daily['date']  = df_daily['date'].astype(str)
    df_prices['date'] = df_prices['date'].astype(str)

    df_merged = pd.merge(
        df_daily, df_prices,
        on='date', how='inner'
    ).sort_values('date').reset_index(drop=True)

    print(f"\n── Merge complete ──────────────────────────────")
    print(f"  Merged rows  : {len(df_merged)}")
    print(f"  Date range   : {df_merged['date'].min()} → "
          f"{df_merged['date'].max()}")

    return df_merged


# ════════════════════════════════════════════════════════════
# PHASE 4c — CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════

def compute_correlations(df_merged: pd.DataFrame) -> dict:
    """
    Compute Pearson correlation between sentiment and
    price_change_pct at same-day, lag-1, lag-2.

    Returns:
        results = {
            'VADER'  : {'same_day': x, 'lag_1': x, 'lag_2': x},
            'FinBERT': {...},
            'LLM'    : {...}
        }
    """
    if df_merged.empty:
        print("⚠ Cannot correlate — empty DataFrame")
        return {}

    # Add lag columns
    for model in ['vader', 'finbert', 'llm']:
        col = f'{model}_compound'
        df_merged[f'{model}_lag1'] = df_merged[col].shift(1)
        df_merged[f'{model}_lag2'] = df_merged[col].shift(2)

    price = df_merged['price_change_pct']
    results = {}

    models = [
        ('VADER',   'vader_compound',   'vader_lag1',   'vader_lag2'),
        ('FinBERT', 'finbert_compound', 'finbert_lag1', 'finbert_lag2'),
        ('LLM',     'llm_compound',     'llm_lag1',     'llm_lag2'),
    ]

    print(f"\n── Correlation Analysis ────────────────────────")
    print(f"  n = {len(df_merged)} trading days")
    print(f"\n  {'Model':<12} {'Same-day':>10} "
          f"{'Lag-1':>10} {'Lag-2':>10} {'Best':>12}")
    print(f"  {'─'*58}")

    for model_name, same_col, lag1_col, lag2_col in models:
        same = df_merged[same_col].corr(price)
        lag1 = df_merged[lag1_col].corr(price)
        lag2 = df_merged[lag2_col].corr(price)

        vals = {'same_day': same, 'lag_1': lag1, 'lag_2': lag2}
        best_key = max(vals, key=lambda k: abs(vals[k]))
        best_val = vals[best_key]

        results[model_name] = vals
        results[model_name]['best_lag']   = best_key
        results[model_name]['best_value'] = best_val

        print(f"  {model_name:<12} {same:>10.3f} "
              f"{lag1:>10.3f} {lag2:>10.3f} "
              f"  {best_key}={best_val:+.3f}")

    return results, df_merged


def interpret_correlation(results: dict) -> str:
    """
    Generate plain-English interpretation of correlation results.
    Used in UI and findings report.
    """
    lines = []

    for model, vals in results.items():
        if 'best_lag' not in vals:
            continue
        best_lag = vals['best_lag']
        best_val = vals['best_value']

        # Strength
        abs_val = abs(best_val)
        if abs_val >= 0.5:
            strength = "strong"
        elif abs_val >= 0.3:
            strength = "moderate"
        elif abs_val >= 0.1:
            strength = "weak"
        else:
            strength = "negligible"

        # Direction
        direction = "positive" if best_val > 0 else "negative"

        # Lag interpretation
        lag_text = {
            'same_day': "same trading day",
            'lag_1'   : "1 day after publication",
            'lag_2'   : "2 days after publication",
        }.get(best_lag, best_lag)

        lines.append(
            f"{model}: {strength} {direction} correlation "
            f"({best_val:+.3f}) at {lag_text}"
        )

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# MAIN — run full correlation pipeline
# ════════════════════════════════════════════════════════════

def run_correlation(df_relevant: pd.DataFrame,
                    df_prices: pd.DataFrame,
                    symbol: str,
                    run_id: str) -> tuple:
    """
    Full correlation pipeline:
    Daily aggregate → Merge → Correlate

    Args:
        df_relevant : scored articles from sentiment.py
        df_prices   : price window from prices.py
        symbol      : stock ticker
        run_id      : current pipeline run ID

    Returns:
        df_daily    : daily sentiment aggregate
        df_merged   : sentiment + prices joined
        results     : correlation dict per model
        interpretation: plain English summary
    """
    # Phase 4a — daily aggregate
    df_daily = build_daily_sentiment(df_relevant, symbol, run_id)

    # Phase 4b — merge with prices
    df_merged = merge_sentiment_prices(df_daily, df_prices)

    # Phase 4c — correlate
    results, df_merged = compute_correlations(df_merged)

    # Interpret
    interpretation = interpret_correlation(results)
    print(f"\n── Interpretation ──────────────────────────────")
    print(interpretation)

    return df_daily, df_merged, results, interpretation