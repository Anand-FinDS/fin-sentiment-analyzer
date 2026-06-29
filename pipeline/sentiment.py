# pipeline/sentiment.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Phase 3 — VADER + FinBERT Sentiment Scoring
# Input  : df_relevant (Direct + Indirect articles)
# Output : df_relevant with vader + finbert columns added
# Note   : LLM sentiment already in df from classify.py
#          This adds rule-based + domain-aware layers
#          for comparison and validation
# ════════════════════════════════════════════════════════════

import re
import torch
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import (
    FINBERT_MODEL,
    FINBERT_BATCH_SIZE,
    FINANCIAL_LEXICON,
    CLICKBAIT_PATTERNS,
)

# ── Model singletons ──────────────────────────────────────────
_analyzer      = None
_finbert_pipe  = None
_device        = None


def get_device() -> str:
    """Returns cuda if available else cpu."""
    global _device
    if _device is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return _device


def get_vader() -> SentimentIntensityAnalyzer:
    """Load VADER + patch financial lexicon (once)."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCIAL_LEXICON)
        print(f"✓ VADER loaded — "
              f"{len(FINANCIAL_LEXICON)} custom lexicon words")
    return _analyzer


def get_finbert():
    """Load FinBERT on GPU (once)."""
    global _finbert_pipe
    if _finbert_pipe is None:
        from transformers import (
            BertTokenizer,
            BertForSequenceClassification,
            pipeline,
        )
        device = get_device()
        print(f"Loading FinBERT on {device}...")

        tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
        model     = BertForSequenceClassification.from_pretrained(
            FINBERT_MODEL
        )
        model     = model.to(device)
        model.eval()

        _finbert_pipe = pipeline(
            'text-classification',
            model     = model,
            tokenizer = tokenizer,
            device    = 0 if device == 'cuda' else -1,
            truncation= True,
            max_length= 512,
        )
        print(f"✓ FinBERT loaded on {device.upper()}")

        # Quick sanity test
        test = _finbert_pipe(
            "Apple shares fell 3% after earnings missed estimates"
        )
        print(f"  Sanity test : {test}")

    return _finbert_pipe


# ════════════════════════════════════════════════════════════
# PHASE 3a — VADER
# ════════════════════════════════════════════════════════════

def is_clickbait(headline: str) -> bool:
    """Detect clickbait headlines using regex patterns."""
    h = str(headline).lower()
    return any(re.search(p, h) for p in CLICKBAIT_PATTERNS)


def vader_label(score: float) -> str:
    if score >=  0.05: return 'Positive'
    if score <= -0.05: return 'Negative'
    return 'Neutral'


def smart_vader_combined(row) -> float:
    """
    Smart VADER combined score:
    - Empty summary    → headline only
    - Clickbait        → 20/80 headline/summary
    - Normal           → 70/30 headline/summary
    """
    h = row['vader_headline']
    s = row['vader_summary']

    if abs(s) < 0.05:
        return h
    if row['is_clickbait']:
        return 0.2 * h + 0.8 * s
    return 0.7 * h + 0.3 * s


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all articles with VADER.
    Adds columns: vader_headline, vader_summary,
                  vader_compound, vader_label,
                  is_clickbait, is_comparative,
                  vader_low_conf, negated_positive,
                  price_drop_summary, vader_confidence
    """
    analyzer = get_vader()
    df       = df.copy()

    # ── Clickbait flag ────────────────────────────────────────
    df['is_clickbait'] = df['Headline'].apply(is_clickbait)

    # ── Score headline + summary ──────────────────────────────
    df['vader_headline'] = df['Headline'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    df['vader_summary'] = df['Summary'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )

    # ── Smart combined ────────────────────────────────────────
    df['vader_compound'] = df.apply(smart_vader_combined, axis=1)
    df['vader_label']    = df['vader_compound'].apply(vader_label)

    # ── Reliability flags ─────────────────────────────────────
    comparative_triggers = [
        'than', 'vs', 'versus', 'compared to',
        'unlike', 'while', 'but', 'however'
    ]
    df['is_comparative'] = df['Headline'].apply(
        lambda x: any(t in str(x).lower()
                      for t in comparative_triggers)
    )
    df['vader_score_gap'] = (
        df['vader_headline'] - df['vader_summary']
    ).abs()
    df['vader_low_conf']  = df['vader_score_gap'] > 0.5

    df['negated_positive'] = df['Headline'].str.contains(
        r'\b(fails?|failed|unable|refuses?|refused)\s+to\b',
        case=False, regex=True, na=False
    )
    df['price_drop_summary'] = df['Summary'].apply(
        lambda x: any(
            w in str(x).lower()
            for w in ['fell', 'fallen', 'down',
                      'sliding', 'slipped', 'dropped']
        )
    )

    # ── Confidence tier ───────────────────────────────────────
    def vader_confidence(row):
        if (row['vader_low_conf'] or
            row['is_clickbait'] or
            row['negated_positive']):
            return 'Low'
        if row['is_comparative']:
            return 'Medium'
        return 'High'

    df['vader_confidence'] = df.apply(vader_confidence, axis=1)

    # ── Health check ──────────────────────────────────────────
    total = len(df)
    print(f"\n── VADER complete ──────────────────────────────")
    print(f"  Articles    : {total}")
    print(f"  Label split :")
    print(df['vader_label'].value_counts().to_string())
    print(f"\n  Confidence  :")
    for tier in ['High', 'Medium', 'Low']:
        n = (df['vader_confidence'] == tier).sum()
        print(f"    {tier:6} : {n:4}  ({n/total*100:.0f}%)")
    print(f"  Clickbait   : {df['is_clickbait'].sum()}")

    return df


# ════════════════════════════════════════════════════════════
# PHASE 3b — FinBERT
# ════════════════════════════════════════════════════════════

def finbert_compound(label: str) -> float:
    """Convert FinBERT label to compound-style score."""
    return {
        'Positive': 1.0,
        'Negative': -1.0,
        'Neutral' :  0.0
    }.get(label, 0.0)


def score_texts_finbert(texts: list,
                        progress_callback=None) -> tuple[list, list]:
    """
    Score a list of texts with FinBERT in batches.
    Returns (labels, scores).
    Key fix: FinBERT confidence always ~1.0
             key off label not score threshold.
    """
    pipe        = get_finbert()
    all_labels  = []
    all_scores  = []
    total       = len(texts)

    for i in range(0, total, FINBERT_BATCH_SIZE):
        batch   = texts[i : i + FINBERT_BATCH_SIZE]
        results = pipe(batch)

        for r in results:
            all_labels.append(r['label'].capitalize())
            all_scores.append(round(r['score'], 4))

        if progress_callback:
            progress_callback(
                min(i + FINBERT_BATCH_SIZE, total),
                total,
                f"FinBERT: {min(i+FINBERT_BATCH_SIZE, total)}/{total}"
            )
        elif i % (FINBERT_BATCH_SIZE * 10) == 0:
            print(f"  [{min(i+FINBERT_BATCH_SIZE, total)}/{total}]")

    return all_labels, all_scores


def run_finbert(df: pd.DataFrame,
                progress_callback=None) -> pd.DataFrame:
    """
    Score all articles with FinBERT.
    Adds columns: finbert_headline_label, finbert_headline_score,
                  finbert_summary_label,  finbert_summary_score,
                  finbert_label, finbert_compound
    """
    df = df.copy()

    # ── Score headlines ───────────────────────────────────────
    print(f"Scoring {len(df)} headlines with FinBERT...")
    h_labels, h_scores = score_texts_finbert(
        df['Headline'].fillna('').tolist(),
        progress_callback
    )
    print(f"✓ Headlines done")

    # ── Score summaries ───────────────────────────────────────
    print(f"Scoring {len(df)} summaries with FinBERT...")
    s_labels, s_scores = score_texts_finbert(
        df['Summary'].fillna('').tolist(),
        progress_callback
    )
    print(f"✓ Summaries done")

    df['finbert_headline_label'] = h_labels
    df['finbert_headline_score'] = h_scores
    df['finbert_summary_label']  = s_labels
    df['finbert_summary_score']  = s_scores

    # ── Smart combined ────────────────────────────────────────
    # Key fix: confidence always ~1.0 — key off label not score
    # Headline Neutral or clickbait → trust summary
    def finbert_smart_combined(row):
        h = row['finbert_headline_label']
        s = row['finbert_summary_label']
        if h == 'Neutral' or row['is_clickbait']:
            final = s
        else:
            final = h
        return final, finbert_compound(final)

    combined = df.apply(finbert_smart_combined, axis=1)
    df['finbert_label']    = [c[0] for c in combined]
    df['finbert_compound'] = [c[1] for c in combined]

    # ── Summary ───────────────────────────────────────────────
    print(f"\n── FinBERT complete ────────────────────────────")
    print(f"  Label split :")
    print(df['finbert_label'].value_counts().to_string())

    print(f"\n  VADER vs FinBERT crosstab:")
    print(pd.crosstab(
        df['vader_label'],
        df['finbert_label'],
        rownames=['VADER'],
        colnames=['FinBERT']
    ).to_string())

    return df


# ════════════════════════════════════════════════════════════
# PHASE 3c — Three-way comparison
# ════════════════════════════════════════════════════════════

def three_way_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print three-way label comparison:
    VADER vs FinBERT vs LLM (from classify.py)
    """
    print(f"\n── Three-way comparison ────────────────────────")
    print(f"{'Label':<12} {'VADER':>8} {'FinBERT':>8} {'LLM':>8}")
    print(f"{'─'*40}")
    for label in ['Positive', 'Negative', 'Neutral']:
        v = (df['vader_label']    == label).sum()
        f = (df['finbert_label']  == label).sum()
        l = (df['llm_sentiment']  == label).sum()
        print(f"{label:<12} {v:>8} {f:>8} {l:>8}")

    return df


# ════════════════════════════════════════════════════════════
# MAIN — run full sentiment pipeline
# ════════════════════════════════════════════════════════════

def run_sentiment(df_relevant: pd.DataFrame,
                  progress_callback=None) -> pd.DataFrame:
    """
    Run complete sentiment pipeline on relevant articles.
    VADER → FinBERT → three-way comparison

    Args:
        df_relevant      : Direct + Indirect articles from classify.py
        progress_callback: optional function(current, total, msg)

    Returns:
        df_relevant with all sentiment columns added
    """
    if df_relevant.empty:
        print("⚠ Empty DataFrame — nothing to score")
        return df_relevant

    print(f"\nRunning sentiment pipeline on "
          f"{len(df_relevant)} articles...")

    # Phase 3a — VADER
    df_relevant = run_vader(df_relevant)

    # Phase 3b — FinBERT
    df_relevant = run_finbert(df_relevant, progress_callback)

    # Phase 3c — Three-way comparison
    df_relevant = three_way_comparison(df_relevant)

    print(f"\n── Sentiment pipeline complete ─────────────────")
    print(f"  Columns added:")
    sentiment_cols = [
        'is_clickbait', 'vader_headline', 'vader_summary',
        'vader_compound', 'vader_label', 'vader_confidence',
        'finbert_headline_label', 'finbert_headline_score',
        'finbert_summary_label', 'finbert_summary_score',
        'finbert_label', 'finbert_compound',
    ]
    for col in sentiment_cols:
        if col in df_relevant.columns:
            print(f"    ✓ {col}")

    return df_relevant