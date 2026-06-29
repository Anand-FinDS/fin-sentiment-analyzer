# pipeline/classify.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Phase 2 — LLM Classification + Sentiment (Groq)
# Combined in ONE call: relevance + sentiment score
# Auto-fallback: llama-4-scout → llama-3.1-8b on quota hit
# Input  : df_news, symbol, company_name
# Output : df_news with relevance_class + llm_sentiment + llm_score
# ════════════════════════════════════════════════════════════

import json
import time
from groq import Groq
import pandas as pd
from config import (
    GROQ_API_KEY,
    GROQ_CLASSIFY_MODEL,
    GROQ_CLASSIFY_MODEL_FALLBACK,
    GROQ_MODELS,
    BATCH_SIZE,
    SUMMARY_TRUNCATE,
    RELEVANCE_WEIGHTS,
)

# ── Groq client (singleton) ───────────────────────────────────
_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Model state tracker ───────────────────────────────────────
_active_model   = GROQ_CLASSIFY_MODEL
_model_switched = False


def get_active_model() -> str:
    """Returns currently active model name."""
    return _active_model


def is_model_switched() -> bool:
    """Returns True if fallback model is active."""
    return _model_switched


def get_sleep_interval() -> int:
    """Returns sleep interval for currently active model."""
    return GROQ_MODELS[_active_model]['sleep_interval']


# ── Combined prompt — classify + sentiment ────────────────────
def get_combined_prompt(symbol: str, company_name: str) -> str:
    return f"""You are a financial news classifier and sentiment analyzer
for {company_name} ({symbol}) stock analysis.

For each article return:
1. relevance_class: Direct | Indirect | Irrelevant
2. sentiment: Positive | Negative | Neutral
3. sentiment_score: float between -1.0 and 1.0
4. reason: <8 words max>

Rules:
- relevance_class relative to {company_name} ({symbol}) only
- Direct: primarily about {company_name} products, financials,
  leadership, supply chain, regulatory issues
- Indirect: not primarily about {company_name} but materially
  impacts {symbol} — major shareholders, key suppliers,
  sector regulation, macro events for this industry
- Irrelevant: no meaningful connection to {symbol} stock
- sentiment = impact on {symbol} stock price only
- Irrelevant articles → sentiment: Neutral, score: 0.0
- Financial context: overweight/outperform/buy = Positive
- raises price target = Positive
- maintains rating = mildly Positive
- downgrades/underweight/sell = Negative
- Indirect impact counts: Berkshire selling = Negative for {symbol}

Reply ONLY with a JSON array, same order as input:
[{{
  "id": 0,
  "relevance_class": "Direct"|"Indirect"|"Irrelevant",
  "sentiment": "Positive"|"Negative"|"Neutral",
  "sentiment_score": float,
  "reason": "<8 words>"
}}, ...]"""


# ── Single batch call with auto-fallback ──────────────────────
def classify_batch(batch_df: pd.DataFrame,
                   system_prompt: str,
                   use_fallback: bool = False) -> list[tuple]:
    """
    Classify + score a batch in one Groq call.
    Auto-switches to fallback model on quota exceeded.
    Returns list of (relevance_class, sentiment, score, reason, model).
    """
    global _active_model, _model_switched

    client = get_groq_client()
    model  = (GROQ_CLASSIFY_MODEL_FALLBACK
              if use_fallback
              else GROQ_CLASSIFY_MODEL)

    articles_text = "\n\n".join([
        f"[{i}] Headline: {str(row['Headline'])}\n"
        f"Summary: {str(row['Summary'])[:SUMMARY_TRUNCATE]}"
        for i, (_, row) in enumerate(batch_df.iterrows())
    ])

    raw = ""
    try:
        resp = client.chat.completions.create(
            model    = model,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": articles_text}
            ],
            temperature = 0,
            max_tokens  = 600,
        )
        raw    = resp.choices[0].message.content.strip()
        # Remove any preamble before the JSON array
        if '[' in raw:
            raw = raw[raw.index('['):]
        if ']' in raw:
            raw = raw[:raw.rindex(']') + 1]
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)

        return [
            (
                p.get("relevance_class", "Irrelevant"),
                p.get("sentiment",       "Neutral"),
                float(p.get("sentiment_score", 0.0)),
                p.get("reason",          ""),
                model,
            )
            for p in parsed
        ]

    except Exception as e:
        error_str = str(e)

        # ── Quota exceeded → switch to fallback ───────────────
        if any(code in error_str.lower() for code in
               ['429', 'rate_limit', 'quota', 'exceeded',
                'too many requests']):

            if not use_fallback:
                _active_model   = GROQ_CLASSIFY_MODEL_FALLBACK
                _model_switched = True
                fallback_info   = GROQ_MODELS[GROQ_CLASSIFY_MODEL_FALLBACK]
                primary_info    = GROQ_MODELS[GROQ_CLASSIFY_MODEL]

                print(f"\n  ⚠ Quota exceeded on "
                      f"{primary_info['display_name']}")
                print(f"  → Switching to fallback: "
                      f"{fallback_info['display_name']}")
                print(f"  → New sleep interval : "
                      f"{fallback_info['sleep_interval']}s")
                print(f"  → Speed impact       : "
                      f"~{fallback_info['est_mins_per_1000_articles']} "
                      f"mins per 1K articles\n")

                time.sleep(30)  # cooldown before retry
                return classify_batch(batch_df, system_prompt,
                                      use_fallback=True)

        print(f"  ⚠ Batch error: {e} | raw: {raw[:80]}")
        return [
            ("Irrelevant", "Neutral", 0.0, "batch-error", model)
        ] * len(batch_df)


# ── Rescue misclassified articles ────────────────────────────
def rescue_misclassified(df_news: pd.DataFrame,
                         symbol: str,
                         company_name: str,
                         classes: list,
                         sentiments: list,
                         scores: list,
                         reasons: list,
                         models_used: list) -> tuple:
    """
    Re-classify articles marked Irrelevant that explicitly
    mention company/symbol — fixes rate limit fallbacks.
    """
    system_prompt = get_combined_prompt(symbol, company_name)
    use_fallback  = _model_switched

    rescue_mask = (
        (pd.Series(classes) == 'Irrelevant') &
        (df_news['Headline'].str.contains(
            f'{company_name}|{symbol}',
            case=False, na=False) |
         df_news['Summary'].str.contains(
            f'{company_name}|{symbol}',
            case=False, na=False))
    )

    rescue_idx = df_news[rescue_mask.values].index.tolist()
    print(f"  Rescue candidates : {len(rescue_idx)}")

    if not rescue_idx:
        return classes, sentiments, scores, reasons, models_used

    rescue_df = df_news.loc[rescue_idx].reset_index(drop=True)
    r_classes, r_sentiments  = [], []
    r_scores,  r_reasons     = [], []
    r_models                 = []

    for batch_start in range(0, len(rescue_df), BATCH_SIZE):
        batch   = rescue_df.iloc[batch_start : batch_start + BATCH_SIZE]
        results = classify_batch(batch, system_prompt, use_fallback)

        while len(results) < len(batch):
            results.append(
                ("Irrelevant", "Neutral", 0.0, "missing", _active_model)
            )

        for cls, sent, score, reason, mdl in results:
            r_classes.append(cls)
            r_sentiments.append(sent)
            r_scores.append(score)
            r_reasons.append(reason)
            r_models.append(mdl)

        time.sleep(get_sleep_interval())

    # Patch back into main lists
    for i, idx in enumerate(rescue_idx):
        classes[idx]     = r_classes[i]
        sentiments[idx]  = r_sentiments[i]
        scores[idx]      = r_scores[i]
        reasons[idx]     = r_reasons[i]
        models_used[idx] = r_models[i]

    rescued = sum(1 for c in r_classes if c != 'Irrelevant')
    print(f"  Rescued           : {rescued} articles")

    return classes, sentiments, scores, reasons, models_used


# ── Main function ─────────────────────────────────────────────
def classify_and_score(df_news: pd.DataFrame,
                       symbol: str,
                       company_name: str,
                       progress_callback=None) -> pd.DataFrame:
    """
    Classify relevance AND score sentiment in one Groq pass.
    Auto-fallback to llama-3.1-8b if scout quota exceeded.

    Args:
        df_news          : raw articles from fetch.py
        symbol           : stock ticker e.g. 'AAPL'
        company_name     : full name e.g. 'Apple Inc'
        progress_callback: optional function(current, total, msg)
                          for Streamlit progress bar

    Returns:
        df_news with columns:
            relevance_class  : Direct / Indirect / Irrelevant
            llm_sentiment    : Positive / Negative / Neutral
            llm_score        : float -1.0 to 1.0
            llm_reason       : one-line explanation
            llm_model_used   : which model scored this batch
            relevance_weight : 1.0 / 0.5 / 0.0
    """
    global _active_model, _model_switched

    # Reset model state for fresh run
    _active_model   = GROQ_CLASSIFY_MODEL
    _model_switched = False

    if df_news.empty:
        print("⚠ Empty DataFrame — nothing to classify")
        return df_news

    system_prompt = get_combined_prompt(symbol, company_name)
    total         = len(df_news)
    total_batches = (total // BATCH_SIZE) + 1
    primary_info  = GROQ_MODELS[GROQ_CLASSIFY_MODEL]

    print(f"\nClassifying + scoring {total} articles")
    print(f"Model   : {primary_info['display_name']}")
    print(f"Batches : {total_batches} × {BATCH_SIZE} articles")
    print(f"ETA     : ~{round(total_batches * primary_info['sleep_interval'] / 60, 1)} mins")
    print(f"Fallback: {GROQ_MODELS[GROQ_CLASSIFY_MODEL_FALLBACK]['display_name']} "
          f"(auto on quota)\n")

    classes    = []
    sentiments = []
    scores     = []
    reasons    = []
    models_used= []

    for batch_start in range(0, total, BATCH_SIZE):
        batch        = df_news.iloc[batch_start : batch_start + BATCH_SIZE]
        use_fallback = _model_switched
        results      = classify_batch(batch, system_prompt, use_fallback)

        while len(results) < len(batch):
            results.append(
                ("Irrelevant", "Neutral", 0.0, "missing", _active_model)
            )

        for cls, sent, score, reason, mdl in results:
            classes.append(cls)
            sentiments.append(sent)
            scores.append(score)
            reasons.append(reason)
            models_used.append(mdl)

        done = batch_start + len(batch)

        if progress_callback:
            dist = pd.Series(classes).value_counts().to_dict()
            progress_callback(
                done, total,
                f"[{_active_model.split('/')[-1]}] "
                f"{done}/{total} — {dist}"
            )
        elif (batch_start // BATCH_SIZE) % 10 == 0:
            dist = pd.Series(classes).value_counts().to_dict()
            switched_tag = " [FALLBACK]" if _model_switched else ""
            print(f"  [{done}/{total}]{switched_tag}  {dist}")

        time.sleep(get_sleep_interval())

    # ── Rescue pass ───────────────────────────────────────────
    print(f"\nRunning rescue pass...")
    (classes, sentiments,
     scores, reasons,
     models_used) = rescue_misclassified(
        df_news, symbol, company_name,
        classes, sentiments, scores,
        reasons, models_used
    )

    # ── Assign columns ────────────────────────────────────────
    df_news = df_news.copy()
    df_news['relevance_class']  = classes
    df_news['llm_sentiment']    = sentiments
    df_news['llm_score']        = scores
    df_news['llm_reason']       = reasons
    df_news['llm_model_used']   = models_used
    df_news['relevance_weight'] = df_news['relevance_class'].map(
        RELEVANCE_WEIGHTS
    ).fillna(0.0)

    # ── Summary ───────────────────────────────────────────────
    direct   = (df_news['relevance_class'] == 'Direct').sum()
    indirect = (df_news['relevance_class'] == 'Indirect').sum()
    irr      = (df_news['relevance_class'] == 'Irrelevant').sum()

    print(f"\n── Classification + Scoring complete ───────────")
    print(f"  Model used  : {_active_model.split('/')[-1]}")
    print(f"  Switched    : {'Yes → ' + GROQ_CLASSIFY_MODEL_FALLBACK if _model_switched else 'No'}")
    print(f"  Direct      : {direct}")
    print(f"  Indirect    : {indirect}")
    print(f"  Irrelevant  : {irr}")

    relevant_mask = df_news['relevance_class'] != 'Irrelevant'
    print(f"\n  LLM sentiment (relevant articles only):")
    print(df_news[relevant_mask]['llm_sentiment']
          .value_counts().to_string())

    return df_news


# ── Filter to relevant only ───────────────────────────────────
def get_relevant(df_news: pd.DataFrame) -> pd.DataFrame:
    """Filter to Direct + Indirect articles only."""
    df_relevant = df_news[
        df_news['relevance_class'] != 'Irrelevant'
    ].reset_index(drop=True)

    print(f"\n── df_relevant ─────────────────────────────────")
    print(f"  Total    : {len(df_relevant)} articles")
    print(f"  Direct   : "
          f"{(df_relevant['relevance_class'] == 'Direct').sum()}")
    print(f"  Indirect : "
          f"{(df_relevant['relevance_class'] == 'Indirect').sum()}")

    return df_relevant