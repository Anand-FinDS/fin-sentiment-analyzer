# pipeline/report.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Phase 5 — GPT-4o Findings Report Generator
# Input  : correlation results + sentiment summary
# Output : structured analyst report (text)
# ════════════════════════════════════════════════════════════

import json
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_REPORT_MODEL
import pandas as pd


# ── OpenAI client (singleton) ─────────────────────────────────
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── System prompt ─────────────────────────────────────────────
REPORT_SYSTEM = """You are a senior financial data scientist writing 
an analysis report for a financial analyst end user.

Given quantitative findings from a GenAI Financial Sentiment Analyzer,
generate a structured report with these exact sections:

1. EXECUTIVE SUMMARY
   3-4 sentences, non-technical, for a financial analyst.
   What happened with this stock during this period?

2. KEY FINDINGS
   5 bullet points with specific numbers.
   What do the sentiment scores actually show?

3. MODEL COMPARISON
   Compare VADER vs FinBERT vs LLM performance.
   Strengths and weaknesses of each for this dataset.

4. SENTIMENT-PRICE RELATIONSHIP
   Interpret the correlation numbers.
   What does the lag analysis reveal about news digestion?

5. NOTABLE EVENTS
   Based on sentiment spikes or drops, what likely drove them?
   Reference specific dates if scores show anomalies.

6. LIMITATIONS
   Honest assessment — what can this data NOT tell us?
   Sample size, time window, confounding factors.

7. FUTURE OUTLOOK
   3 specific recommendations to improve the system.
   Based on findings, what should a fin analyst watch next?

8. ANALYST TAKEAWAY
   One paragraph — if you were a financial analyst,
   what action or watch-list decision would you make
   based on this sentiment analysis?

Rules:
- Use the actual numbers provided — no generic statements
- Be specific about which model performed best and why
- Acknowledge weak correlations honestly
- Write in professional but accessible language
- Keep each section concise — quality over length"""


# ── Build findings payload ────────────────────────────────────
def build_findings_payload(symbol: str,
                           company_name: str,
                           date_from: str,
                           date_to: str,
                           df_relevant: pd.DataFrame,
                           df_daily: pd.DataFrame,
                           df_merged: pd.DataFrame,
                           correlation_results: dict) -> dict:
    """
    Build structured findings dict from pipeline outputs.
    Sent to GPT-4o as context for report generation.
    """
    # Sentiment distribution
    sentiment_dist = {}
    for model, col in [
        ('VADER',   'vader_label'),
        ('FinBERT', 'finbert_label'),
        ('LLM',     'llm_sentiment'),
    ]:
        if col in df_relevant.columns:
            sentiment_dist[model] = (
                df_relevant[col].value_counts().to_dict()
            )

    # Score ranges per model
    score_ranges = {}
    for model, col in [
        ('VADER',   'vader_compound'),
        ('FinBERT', 'finbert_compound'),
        ('LLM',     'llm_score'),
    ]:
        if col in df_relevant.columns:
            score_ranges[model] = {
                'min' : round(float(df_relevant[col].min()), 3),
                'max' : round(float(df_relevant[col].max()), 3),
                'mean': round(float(df_relevant[col].mean()), 3),
                'std' : round(float(df_relevant[col].std()), 3),
            }

    # Most positive + negative days
    notable_days = {}
    if not df_merged.empty and 'llm_compound' in df_merged.columns:
        most_positive = df_merged.loc[
            df_merged['llm_compound'].idxmax()
        ]
        most_negative = df_merged.loc[
            df_merged['llm_compound'].idxmin()
        ]
        notable_days = {
            'most_positive_sentiment': {
                'date'         : str(most_positive['date']),
                'llm_compound' : round(most_positive['llm_compound'], 3),
                'price_change' : round(
                    most_positive.get('price_change_pct', 0), 3),
                'article_count': int(most_positive['article_count']),
            },
            'most_negative_sentiment': {
                'date'         : str(most_negative['date']),
                'llm_compound' : round(most_negative['llm_compound'], 3),
                'price_change' : round(
                    most_negative.get('price_change_pct', 0), 3),
                'article_count': int(most_negative['article_count']),
            },
        }

    # Price summary
    price_summary = {}
    if not df_merged.empty:
        price_summary = {
            'start_close'     : round(float(
                df_merged['close'].iloc[0]), 2),
            'end_close'       : round(float(
                df_merged['close'].iloc[-1]), 2),
            'total_return_pct': round(float(
                (df_merged['close'].iloc[-1] /
                 df_merged['close'].iloc[0] - 1) * 100), 2),
            'max_daily_gain'  : round(float(
                df_merged['price_change_pct'].max()), 2),
            'max_daily_loss'  : round(float(
                df_merged['price_change_pct'].min()), 2),
        }

    # Clean correlation results for JSON
    clean_corr = {}
    for model, vals in correlation_results.items():
        clean_corr[model] = {
            'same_day'  : round(vals.get('same_day', 0), 3),
            'lag_1'     : round(vals.get('lag_1', 0), 3),
            'lag_2'     : round(vals.get('lag_2', 0), 3),
            'best_lag'  : vals.get('best_lag', 'same_day'),
            'best_value': round(vals.get('best_value', 0), 3),
        }

    return {
        'stock'                 : symbol,
        'company'               : company_name,
        'period'                : f"{date_from} to {date_to}",
        'trading_days'          : len(df_daily),
        'total_articles'        : len(df_relevant),
        'direct_articles'       : int(
            (df_relevant['relevance_class'] == 'Direct').sum()),
        'indirect_articles'     : int(
            (df_relevant['relevance_class'] == 'Indirect').sum()),
        'sentiment_distribution': sentiment_dist,
        'score_ranges'          : score_ranges,
        'correlation_results'   : clean_corr,
        'price_summary'         : price_summary,
        'notable_days'          : notable_days,
        'pipeline_info'         : {
            'classification_model': 'llama-4-scout-17b (Groq)',
            'sentiment_models'    : ['VADER', 'FinBERT', 'LLM'],
            'relevance_weighting' : 'Direct=1.0, Indirect=0.5',
            'lag_windows_tested'  : ['same_day', 'lag_1', 'lag_2'],
        }
    }


# ── Generate report ───────────────────────────────────────────
def generate_report(symbol: str,
                    company_name: str,
                    date_from: str,
                    date_to: str,
                    df_relevant: pd.DataFrame,
                    df_daily: pd.DataFrame,
                    df_merged: pd.DataFrame,
                    correlation_results: dict) -> str:
    """
    Generate GPT-4o analyst findings report.

    Args:
        symbol              : stock ticker
        company_name        : full company name
        date_from / date_to : analysis period
        df_relevant         : scored articles
        df_daily            : daily sentiment aggregate
        df_merged           : sentiment + prices joined
        correlation_results : from correlate.py

    Returns:
        report_text : structured analyst report string
    """
    client = get_openai_client()

    # Build payload
    payload = build_findings_payload(
        symbol, company_name,
        date_from, date_to,
        df_relevant, df_daily,
        df_merged, correlation_results
    )

    print(f"\nGenerating findings report via {OPENAI_REPORT_MODEL}...")
    print(f"  Stock   : {symbol} ({company_name})")
    print(f"  Period  : {date_from} → {date_to}")
    print(f"  Articles: {payload['total_articles']}")
    print(f"  Days    : {payload['trading_days']}")

    try:
        resp = client.chat.completions.create(
            model       = OPENAI_REPORT_MODEL,
            messages    = [
                {"role": "system", "content": REPORT_SYSTEM},
                {"role": "user",
                 "content": json.dumps(payload, indent=2)}
            ],
            temperature = 0.3,
            max_tokens  = 2000,
        )

        report_text = resp.choices[0].message.content.strip()
        tokens_used = resp.usage.total_tokens

        print(f"\n✓ Report generated")
        print(f"  Tokens used : {tokens_used}")
        print(f"  Est. cost   : ~${tokens_used * 0.000005:.4f}")
        print(f"\n{'═'*65}")
        print(f"  {symbol} ({company_name}) — FINDINGS REPORT")
        print(f"  {date_from} → {date_to}")
        print(f"{'═'*65}")
        print(report_text)
        print(f"{'═'*65}")

        return report_text

    except Exception as e:
        print(f"⚠ Report generation failed: {e}")
        # Fallback — return structured summary without LLM
        return _fallback_report(payload)


# ── Fallback report (no LLM) ──────────────────────────────────
def _fallback_report(payload: dict) -> str:
    """
    Simple text report if OpenAI call fails.
    No LLM required — pure data summary.
    """
    corr = payload.get('correlation_results', {})
    dist = payload.get('sentiment_distribution', {})
    price = payload.get('price_summary', {})

    lines = [
        f"FINDINGS REPORT — {payload['stock']} "
        f"({payload['company']})",
        f"Period: {payload['period']}",
        f"{'─'*50}",
        f"\nARTICLES ANALYZED",
        f"  Total    : {payload['total_articles']}",
        f"  Direct   : {payload['direct_articles']}",
        f"  Indirect : {payload['indirect_articles']}",
        f"  Trading days: {payload['trading_days']}",
        f"\nCORRELATION RESULTS",
    ]

    for model, vals in corr.items():
        lines.append(
            f"  {model:<10} same={vals['same_day']:+.3f} "
            f"lag1={vals['lag_1']:+.3f} "
            f"lag2={vals['lag_2']:+.3f} "
            f"→ best={vals['best_value']:+.3f} ({vals['best_lag']})"
        )

    if price:
        lines += [
            f"\nPRICE SUMMARY",
            f"  Start : ${price.get('start_close', 0):.2f}",
            f"  End   : ${price.get('end_close', 0):.2f}",
            f"  Return: {price.get('total_return_pct', 0):+.2f}%",
        ]

    lines.append(
        f"\nNote: GPT-4o report unavailable — "
        f"showing raw summary."
    )

    return "\n".join(lines)