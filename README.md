# 📈 GenAI Financial Sentiment Analyzer

A full-stack GenAI application that analyzes financial news sentiment 
and correlates it with stock price movements using a progressive 
3-stage NLP pipeline.

## 🎯 Project Overview

Built as a capstone project for GenAI & Agentic AI certification, 
this system demonstrates how Large Language Models can be integrated 
into a production-grade financial analysis pipeline.

## 🏗️ Architecture
Financial News (Finnhub API)
↓
┌─────────────────────────────────────┐
│  Stage 1: LLM Classification        │
│  Groq llama-4-scout-17b             │
│  → Direct / Indirect / Irrelevant   │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│  Stage 2a: VADER (Rule-based)       │
│  Custom financial lexicon           │
│  Clickbait detection                │
│  Smart headline/summary weighting   │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│  Stage 2b: FinBERT (Domain Model)   │
│  ProsusAI/finbert                   │
│  GPU accelerated (local)            │
│  LLM substitute (cloud)             │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│  Stage 3: Correlation Analysis      │
│  Pearson: Same-day, Lag-1, Lag-2    │
│  Stock price via yfinance           │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│  Stage 4: GPT-4o Findings Report    │
│  AI-generated analysis              │
│  Downloadable PDF/TXT               │
└─────────────────────────────────────┘
## ✨ Key Features

- **Stock-agnostic pipeline** — works for any of 179 quality US stocks
- **Progressive NLP** — VADER → FinBERT → LLM, each layer validates the prior
- **Smart caching** — SQLite stores results, second run is instant
- **Agentic classification** — LLM autonomously classifies article relevance
- **Weekend handling** — rolls Saturday/Sunday news to next NYSE trading day
- **Clickbait detection** — regex patterns flip headline/summary weights
- **Custom financial lexicon** — 75+ domain-specific VADER overrides
- **Cloud-safe deployment** — `USE_FINBERT` flag enables local/cloud switching

## 🖥️ Streamlit Dashboard

5-tab interface:
| Tab | Content |
|-----|---------|
| 📈 Sentiment & Price | Daily sentiment vs price movement chart |
| 🔥 Correlation | Heatmap + Pearson correlation table |
| 📰 News Feed | Classified articles with sentiment badges |
| 📋 Findings Report | GPT-4o generated analysis |
| 📉 Stock Trend | 1-year price history with MA20/MA50 |

## 🚀 Quick Start

### Local (with GPU — full pipeline)
```bash
git clone https://github.com/Anand-FinDS/fin-sentiment-analyzer.git
cd fin-sentiment-analyzer

# Create environment
conda create -n sentiment python=3.10 -y
conda activate sentiment
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize database
python -m db.schema
python setup_stocks.py
python filter_stocks_by_mcap.py

# Run
streamlit run app.py
```

### Cloud (without GPU)
```bash
# Same setup but in .env set:
USE_FINBERT=false
# LLM sentiment fills FinBERT role automatically
```

## 🔑 Environment Variables

Create `.env` file:
FINNHUB_API_KEY=your_finnhub_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
USE_FINBERT=true   # false for cloud deployment

Get API keys:
- Finnhub: https://finnhub.io (free tier)
- Groq: https://console.groq.com (free tier)
- OpenAI: https://platform.openai.com

## 🤖 Models Used

| Stage | Model | Provider |
|-------|-------|----------|
| Classification + Sentiment | llama-4-scout-17b-16e-instruct | Groq |
| Fallback | llama-3.1-8b-instant | Groq |
| Domain Sentiment | ProsusAI/finbert | HuggingFace |
| Findings Report | gpt-4o-mini | OpenAI |
| Baseline | VADER + custom lexicon | Local |

## 📊 Key Findings (AAPL Sample)

- **2,528** raw articles → **1,977** relevant (842 Direct, 1,135 Indirect)
- **VADER domain mismatch**: "overweight" scored -1.5 (medical English)
  vs +2.0 (analyst buy rating) — illustrates need for domain-specific NLP
- **Best correlation**: FinBERT Lag-2 = +0.166 (2-day news digestion)
- **LLM advantage**: Strongest same-day correlation (+0.436 for AMAT)

## 🏗️ Project Structure
fin-sentiment-analyzer/
├── app.py                    # Streamlit UI (5 tabs)
├── config.py                 # API keys, constants, lexicon
├── requirements.txt
├── setup_stocks.py           # Finnhub stock universe fetch
├── filter_stocks_by_mcap.py  # Filter to 179 quality stocks
├── pipeline/
│   ├── fetch.py              # Finnhub news + trading date
│   ├── classify.py           # Groq LLM classification
│   ├── sentiment.py          # VADER + FinBERT scoring
│   ├── prices.py             # yfinance price data
│   ├── correlate.py          # Pearson correlation
│   └── report.py             # GPT-4o report generation
├── db/
│   ├── schema.py             # SQLite schema + stats
│   └── queries.py            # All CRUD operations
└── charts/
└── visualizations.py     # 6 matplotlib charts

## ☁️ Deployment
**Live Demo**: http://34.72.212.182:8501 (GCP n1-standard-4)
**Cloud Architecture**:
GCP VM (n1-standard-4, 15GB RAM)
└── Ubuntu 22.04
└── Miniconda (Python 3.10)
└── Streamlit (port 8501)
└── SQLite (persistent DB)

## 🔮 Future Work
- LangSmith integration for LLM observability
- LangGraph agentic orchestration for multi-stock parallel analysis
- Social media sentiment (Reddit/Twitter) integration
- Real-time streaming pipeline
- GPU instance for FinBERT on cloud

## 📚 Tech Stack
Frontend  : Streamlit 1.35.0
Pipeline  : Python 3.10, pandas, numpy
NLP       : VADER, HuggingFace Transformers, Groq SDK
ML        : PyTorch (CUDA 12.8), scikit-learn
Data      : Finnhub API, yfinance
Storage   : SQLite
Charts    : Matplotlib, Seaborn
Cloud     : GCP Compute Engine
CI/CD     : GitHub

## 👤 Author
**Anand** — GenAI & Agentic AI Certification Capstone Project
[![GitHub](https://img.shields.io/badge/GitHub-Anand--FinDS-black)](https://github.com/Anand-FinDS)