# config.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Central configuration — API keys + constants ONLY
# Stock universe → fetched from Finnhub once → stored in SQLite
# ════════════════════════════════════════════════════════════

import os
from dotenv import load_dotenv 

# ── API Keys ──────────────────────────────────────────────────
load_dotenv()  # reads .env file automatically

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY",    "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY",  "")
# ── Deployment config ─────────────────────────────────────────
# True  = local machine with GPU (FinBERT enabled)
# False = cloud deployment (LLM fills FinBERT role)
USE_FINBERT     = os.environ.get("USE_FINBERT", "true").lower() == "true"


GROQ_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "display_name" : "Llama 4 Scout 17B",
        "rpd"          : 1_000,
        "tpm"          : 30_000,
        "tpd"          : 500_000,
        "sleep_interval": 3,
        "est_mins_per_1000_articles": 10,
        "tier"         : "primary",
    },
    "llama-3.1-8b-instant": {
        "display_name" : "Llama 3.1 8B Instant",
        "rpd"          : 14_400,
        "tpm"          : 6_000,
        "tpd"          : 500_000,
        "sleep_interval": 13,
        "est_mins_per_1000_articles": 43,
        "tier"         : "fallback",
    },
}

# ── Model config ──────────────────────────────────────────────
GROQ_CLASSIFY_MODEL          = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_CLASSIFY_MODEL_FALLBACK = "llama-3.1-8b-instant"
OPENAI_REPORT_MODEL   = "gpt-4o-mini"
FINBERT_MODEL         = "yiyanghkust/finbert-tone"

# ── Pipeline config ───────────────────────────────────────────
BATCH_SIZE           = 10    # articles per Groq call
SLEEP_INTERVAL       = 3    # seconds between Groq batches
SUMMARY_TRUNCATE     = 150   # chars of summary sent to LLM
FINBERT_BATCH_SIZE   = 32    # articles per FinBERT GPU batch
MIN_ARTICLES_PER_DAY = 5     # days below this excluded from correlation
LAG_WINDOWS          = [0, 1, 2]

# ── Stock filter — Finnhub fetch criteria ─────────────────────
STOCK_EXCHANGE    = "US"           # Finnhub exchange code
STOCK_MIC         = "XNAS"        # NASDAQ MIC code
                                   # XNYS = NYSE if needed later
MIN_MARKET_CAP    = 1_000_000_000  # $1B minimum — filters penny stocks

# ── Database ──────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "financial_sentiment.db")

# ── Relevance weighting ───────────────────────────────────────
RELEVANCE_WEIGHTS = {
    "Direct"  : 1.0,
    "Indirect": 0.5,
}

# ── Date range presets for UI ─────────────────────────────────
DATE_PRESETS = {
    "Last 30 days" : 30,
    "Last 60 days" : 60,
    "Last 90 days" : 90,
    "Last 6 months": 180,
}

# ── VADER financial lexicon ───────────────────────────────────
FINANCIAL_LEXICON = {
    'surges'        :  2.0, 'soars'         :  2.0,
    'rallies'       :  1.8, 'jumps'         :  1.5,
    'beats'         :  1.5, 'record'        :  1.2,
    'upgrades'      :  1.5, 'breakout'      :  1.3,
    'bullish'       :  2.0, 'outperform'    :  1.5,
    'gained'        :  1.5, 'gaining'       :  1.5,
    'climbed'       :  1.5, 'advances'      :  1.5,
    'rebounds'      :  1.8, 'recovered'     :  1.5,
    'overweight'    :  2.0, 'underweight'   : -2.0,
    'equalweight'   :  0.0, 'marketperform' :  0.0,
    'raises'        :  1.5, 'maintains'     :  0.2,
    'reiterates'    :  0.2, 'initiates'     :  0.3,
    'plunges'       : -2.0, 'crashes'       : -2.5,
    'tumbles'       : -1.8, 'slumps'        : -1.5,
    'misses'        : -1.5, 'downgrades'    : -1.5,
    'bearish'       : -2.0, 'underperform'  : -1.5,
    'recall'        : -1.8, 'lawsuit'       : -1.5,
    'layoffs'       : -1.3, 'bankruptcy'    : -3.0,
    'fell'          : -1.8, 'fallen'        : -1.8,
    'sliding'       : -1.5, 'slides'        : -1.5,
    'slipping'      : -1.5, 'slips'         : -1.5,
    'dropped'       : -1.5, 'drops'         : -1.5,
    'declining'     : -1.3, 'declines'      : -1.3,
    'sinking'       : -1.5, 'sinks'         : -1.5,
    'dragged'       : -1.2, 'weakness'      : -1.2,
    'weak'          : -1.0, 'disappointed'  : -1.8,
    'disappoints'   : -1.8, 'disappointing' : -1.8,
    'failed'        : -1.8, 'fails'         : -1.8,
    'questioning'   : -0.8, 'unimpressed'   : -1.5,
    'concerns'      : -1.0, 'volatile'      : -0.5,
    'cautious'      : -0.5, 'steady'        :  0.3,
    'stable'        :  0.5, 'clear'         :  0.0,
    'key'           :  0.0, 'major'         :  0.0,
    'important'     :  0.0, 'big'           :  0.0,
    'significant'   :  0.0, 'notable'       :  0.0,
    'shares'        :  0.0, 'surprises'     :  0.0,
    'surprise'      :  0.0, 'hoping'        :  0.0,
    'hopes'         :  0.0,
}

# ── Clickbait patterns ────────────────────────────────────────
CLICKBAIT_PATTERNS = [
    r'sends?\s+\w*\s*message',
    r"here'?s\s+why",
    r'what\s+you\s+need\s+to',
    r'this\s+is\s+what',
    r'the\s+real\s+reason',
    r'just\s+happened',
    r'analysts?\s+(say|weigh|react|warn|note|flag)',
    r'what\s+it\s+means',
    r'should\s+you\s+(buy|sell|worry)',
    r'is\s+it\s+(time|worth)',
]