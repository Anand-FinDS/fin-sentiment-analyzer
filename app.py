# app.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# Full Stack Streamlit Application
# ════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title = "Financial Sentiment Analyzer",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

from db.schema   import create_tables
from db.queries  import (
    get_all_stocks, get_cached_run,
    create_run, update_run_status,
    insert_articles, update_articles_classification,
    update_articles_sentiment, insert_daily_sentiment,
    insert_stock_prices, insert_correlation,
    insert_findings, get_findings,
    get_relevant_articles, get_daily_sentiment,
    get_stock_prices, get_merged, get_correlation,
    price_history_exists, get_all_runs,
    get_historical_prices,
)
from pipeline.fetch     import fetch_news
from pipeline.classify  import (
    classify_and_score, get_relevant,
    get_active_model, is_model_switched,
)
from pipeline.sentiment import run_sentiment
from pipeline.prices    import (
    fetch_historical, get_price_window, get_stock_info,
    format_market_cap,
)
from pipeline.correlate import run_correlation
from pipeline.report    import generate_report
from charts.visualizations import generate_all_charts
from config import GROQ_MODELS, GROQ_CLASSIFY_MODEL


@st.cache_resource
def init_db():
    create_tables()
    return True

init_db()


# ── Session state ─────────────────────────────────────────────
for key, default in {
    'run_id'          : None,
    'df_relevant'     : None,
    'df_daily'        : None,
    'df_merged'       : None,
    'corr_results'    : None,
    'stock_info'      : None,
    'last_symbol'     : None,
    'loaded_symbol'   : None,
    'loaded_date_from': None,
    'loaded_date_to'  : None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Sentiment Analyzer - US top stocks")
    st.markdown("---")

    st.markdown("### 🏢 Stock Selection")
    df_stocks = get_all_stocks()
    stock_options = {
        f"{row['symbol']:<6} — {row['company_name']}": row['symbol']
        for _, row in df_stocks.iterrows()
    }
    selected_display = st.selectbox(
        "Select Stock",
        options=list(stock_options.keys()),
        index=list(stock_options.values()).index('AAPL')
        if 'AAPL' in stock_options.values() else 0
    )
    symbol = stock_options[selected_display]


    st.markdown("### 📅 Date Range")
    date_preset = st.selectbox(
        "Preset",
        ["Last 30 days", "Last 60 days",
         "Last 90 days", "Custom range"]
    )
    if date_preset == "Custom range":
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(
                "From", value=datetime.now() - timedelta(days=30))
        with col2:
            date_to = st.date_input(
                "To", value=datetime.now())
    else:
        days_map = {
            "Last 30 days": 30,
            "Last 60 days": 60,
            "Last 90 days": 90,
        }
        days      = days_map[date_preset]
        date_to   = datetime.now().date()
        date_from = (datetime.now() - timedelta(days=days)).date()
        st.caption(f"📆 {date_from} → {date_to}")

    date_from_str = str(date_from)
    date_to_str   = str(date_to)

    st.markdown("---")

    analyze_btn = st.button(
        "🚀 Run Analysis",
        use_container_width=True,
        type="primary"
    )
    st.caption(
        "💡 Look for existing analysis in **Saved Analysis** "
        "below to save time on LLM processing."
    )

    st.markdown("---")

    # ── Saved Analysis ────────────────────────────────────────
    st.markdown("### 📂 Saved Analysis")
    df_runs = get_all_runs()

    if df_runs.empty:
        st.caption("No saved analysis yet")
    else:
        df_complete = df_runs[df_runs['status'] == 'complete'].copy()

        if df_complete.empty:
            st.caption("No completed analysis yet")
        else:
            df_complete['label'] = (
                df_complete['symbol'] + " | " +
                df_complete['date_from'] + " → " +
                df_complete['date_to'] + " (" +
                df_complete['articles_relevant'].astype(str) +
                " articles)"
            )

            selected_run = st.selectbox(
                "Load saved run",
                options=["-- Select --"] + df_complete['label'].tolist()
            )

            # ── SIMPLE DIRECT LOAD — no flags, no triggers ────
            if selected_run != "-- Select --":
                if st.button("📂 Load Analysis",
                             use_container_width=True):
                    idx       = df_complete['label'].tolist().index(
                        selected_run)
                    saved_row = df_complete.iloc[idx]

                    run_id       = saved_row['run_id']
                    saved_symbol = saved_row['symbol']
                    saved_from   = saved_row['date_from']
                    saved_to     = saved_row['date_to']

                    # Load all data into session
                    st.session_state.run_id      = run_id
                    st.session_state.df_relevant = get_relevant_articles(run_id)
                    st.session_state.df_daily    = get_daily_sentiment(run_id)
                    st.session_state.df_merged   = get_merged(run_id)

                    corr_df = get_correlation(run_id)
                    st.session_state.corr_results = {
                        row['model']: {
                            'same_day'  : row['same_day'],
                            'lag_1'     : row['lag_1'],
                            'lag_2'     : row['lag_2'],
                            'best_lag'  : row['best_lag'],
                            'best_value': row['best_value'],
                        }
                        for _, row in corr_df.iterrows()
                    }

                    st.session_state.loaded_symbol    = saved_symbol
                    st.session_state.loaded_date_from = saved_from
                    st.session_state.loaded_date_to   = saved_to
                    st.session_state.stock_info       = get_stock_info(saved_symbol)
                    st.session_state.last_symbol      = saved_symbol

                    st.rerun()  # ← direct rerun inside button click

    st.markdown("---")

    st.markdown("### ⚙ Pipeline Model")
    if is_model_switched():
        st.warning("⚠ Switched to fallback model")
        active_info = GROQ_MODELS.get('llama-3.1-8b-instant', {})
    else:
        st.success("✓ Llama 4 Scout 17B")
        active_info = GROQ_MODELS.get(GROQ_CLASSIFY_MODEL, {})

    st.caption(
        f"TPM: {active_info.get('tpm', 'N/A'):,} | "
        f"~{active_info.get('est_mins_per_1000_articles', '?')} "
        f"mins/1K articles"
    )
    for stage, model in {
        "Classify + Score" : "Llama 4 Scout (Groq)",
        "Domain Sentiment" : "FinBERT (GPU)",
        "Baseline"         : "VADER + Lexicon",
        "Report"           : "GPT-4o",
    }.items():
        st.caption(f"**{stage}**: {model}")

    st.markdown("---")
    st.caption("GenAI Financial Sentiment Analyzer v1.0")


# ════════════════════════════════════════════════════════════
# DISPLAY CONTEXT
# ════════════════════════════════════════════════════════════
display_symbol    = st.session_state.get('loaded_symbol')    or symbol
display_date_from = st.session_state.get('loaded_date_from') or date_from_str
display_date_to   = st.session_state.get('loaded_date_to')   or date_to_str


# ════════════════════════════════════════════════════════════
# STOCK INFO
# ════════════════════════════════════════════════════════════
if display_symbol != st.session_state.get('last_symbol'):
    with st.spinner(f"Loading {display_symbol} info..."):
        st.session_state.stock_info  = get_stock_info(display_symbol)
        st.session_state.last_symbol = display_symbol
        if not price_history_exists(display_symbol):
            fetch_historical(display_symbol)


# ════════════════════════════════════════════════════════════
# COMPANY CARD
# ════════════════════════════════════════════════════════════
info      = st.session_state.stock_info or {}
name      = info.get('name',        display_symbol)
sector    = info.get('sector',      'N/A')
price     = info.get('price')
chg       = info.get('change_pct')
mcap      = format_market_cap(info.get('market_cap'))
w52h      = info.get('week52_high')
w52l      = info.get('week52_low')
raw_desc  = info.get('description', '')[:600]
last_dot  = raw_desc.rfind('.')
desc      = raw_desc[:last_dot + 1] if last_dot > 0 else raw_desc
emp       = info.get('employees')

price_str = f"${price:.2f}" if price else "N/A"
chg_str   = f"{chg:+.2f}%"  if chg   else ""
chg_color = "#66bb6a" if chg and chg > 0 else "#ef5350"

st.markdown(f"""
<div style="background:#1a1a1a; border:1px solid #333;
            border-radius:8px; padding:16px; margin-bottom:16px;">
    <div style="display:flex; justify-content:space-between;
                align-items:center;">
        <div>
            <span style="font-size:24px; font-weight:bold;
                         color:#fff;">{display_symbol}</span>
            <span style="color:#888; margin-left:12px;
                         font-size:14px;">{name}</span>
            <span style="color:#888; margin-left:8px;
                         font-size:12px;">• {sector}</span>
        </div>
        <div style="text-align:right;">
            <span style="font-size:28px; font-weight:bold;
                         color:#fff;">{price_str}</span>
            <span style="color:{chg_color}; margin-left:8px;
                         font-size:14px;">{chg_str}</span>
        </div>
    </div>
    <div style="color:#888; font-size:12px; margin-top:8px;">
        {desc}
    </div>
    <div style="display:flex; gap:24px; margin-top:12px;
                font-size:12px; color:#aaa;">
        <span>52W High: <b style="color:#fff;">
            {"${:.2f}".format(w52h) if w52h else "N/A"}</b></span>
        <span>52W Low: <b style="color:#fff;">
            {"${:.2f}".format(w52l) if w52l else "N/A"}</b></span>
        <span>Mkt Cap: <b style="color:#fff;">{mcap}</b></span>
        <span>Employees: <b style="color:#fff;">
            {f"{emp:,}" if emp else "N/A"}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# RUN ANALYSIS PIPELINE
# ════════════════════════════════════════════════════════════
if analyze_btn:
    st.session_state.loaded_symbol    = None
    st.session_state.loaded_date_from = None
    st.session_state.loaded_date_to   = None
    display_symbol    = symbol
    display_date_from = date_from_str
    display_date_to   = date_to_str

    cached = get_cached_run(symbol, date_from_str, date_to_str)
    if cached:
        st.success(f"✓ Loading cached results for "
                   f"{symbol} {date_from_str} → {date_to_str}")
        run_id = cached['run_id']
        st.session_state.run_id      = run_id
        st.session_state.df_relevant = get_relevant_articles(run_id)
        st.session_state.df_daily    = get_daily_sentiment(run_id)
        st.session_state.df_merged   = get_merged(run_id)
        corr_df = get_correlation(run_id)
        st.session_state.corr_results = {
            row['model']: {
                'same_day'  : row['same_day'],
                'lag_1'     : row['lag_1'],
                'lag_2'     : row['lag_2'],
                'best_lag'  : row['best_lag'],
                'best_value': row['best_value'],
            }
            for _, row in corr_df.iterrows()
        }
        st.rerun()

    else:
        run_id = create_run(symbol, name, date_from_str, date_to_str)
        st.session_state.run_id = run_id
        progress_bar = st.progress(0)
        status_text  = st.empty()

        try:
            status_text.text("📡 Stage 1/6 — Fetching news...")
            progress_bar.progress(5)
            df_news = fetch_news(symbol, date_from_str, date_to_str)
            if df_news.empty:
                st.error(f"No news found for {symbol}.")
                update_run_status(run_id, 'failed')
                st.stop()
            update_run_status(run_id, 'running',
                              articles_fetched=len(df_news))
            progress_bar.progress(15)

            status_text.text("🤖 Stage 2/6 — Classifying with LLM...")
            def classify_progress(current, total, msg):
                pct = 15 + int((current / total) * 35)
                progress_bar.progress(pct)
                status_text.text(f"🤖 Stage 2/6 — {msg}")

            df_news     = classify_and_score(
                df_news, symbol, name,
                progress_callback=classify_progress)
            df_relevant = get_relevant(df_news)
            update_run_status(
                run_id, 'running',
                articles_relevant = len(df_relevant),
                articles_direct   = int(
                    (df_relevant['relevance_class'] == 'Direct').sum()),
                articles_indirect = int(
                    (df_relevant['relevance_class'] == 'Indirect').sum()),
            )
            progress_bar.progress(50)

            status_text.text("🧠 Stage 3/6 — Running VADER + FinBERT...")
            def sentiment_progress(current, total, msg):
                pct = 50 + int((current / total) * 20)
                progress_bar.progress(pct)
                status_text.text(f"🧠 Stage 3/6 — {msg}")

            df_relevant = run_sentiment(
                df_relevant,
                progress_callback=sentiment_progress)
            progress_bar.progress(70)

            status_text.text("💹 Stage 4/6 — Fetching stock prices...")
            df_prices = get_price_window(
                symbol, date_from_str, date_to_str)
            if df_prices.empty:
                fetch_historical(symbol)
                df_prices = get_price_window(
                    symbol, date_from_str, date_to_str)
            progress_bar.progress(80)

            status_text.text("📊 Stage 5/6 — Computing correlations...")
            df_daily, df_merged, corr_results, interpretation = \
                run_correlation(
                    df_relevant, df_prices, symbol, run_id)
            progress_bar.progress(88)

            status_text.text("📋 Stage 6/6 — Generating findings report...")
            report_text = generate_report(
                symbol, name,
                date_from_str, date_to_str,
                df_relevant, df_daily,
                df_merged, corr_results)
            progress_bar.progress(95)

            status_text.text("💾 Saving to database...")
            insert_articles(run_id, df_relevant)
            insert_daily_sentiment(run_id, df_daily)
            insert_stock_prices(run_id, df_prices)
            insert_correlation(run_id, symbol, corr_results)
            insert_findings(
                run_id, symbol,
                date_from_str, date_to_str,
                report_text, 'gpt-4o')
            update_run_status(run_id, 'complete',
                              trading_days=len(df_daily))

            st.session_state.df_relevant  = df_relevant
            st.session_state.df_daily     = df_daily
            st.session_state.df_merged    = df_merged
            st.session_state.corr_results = corr_results

            progress_bar.progress(100)
            status_text.text("✓ Analysis complete!")
            st.success(f"✓ Analysis complete — "
                       f"{len(df_relevant)} articles, "
                       f"{len(df_daily)} trading days")
            st.rerun()

        except Exception as e:
            update_run_status(run_id, 'failed')
            st.error(f"Pipeline failed: {e}")
            st.exception(e)
            st.stop()


# ════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════
df_relevant  = st.session_state.df_relevant
df_daily     = st.session_state.df_daily
df_merged    = st.session_state.df_merged
corr_results = st.session_state.corr_results

if df_relevant is None:
    st.info("👆 Select a stock and date range, "
            "then click **Run Analysis** to start.")
    st.stop()

# Normalize column names
for df in [df_relevant, df_daily, df_merged]:
    if df is not None:
        df.columns = [c.lower() for c in df.columns]


# ════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════
st.markdown("### 📊 Analysis Overview")
m1, m2, m3, m4, m5, m6 = st.columns(6)

best_corr_val = 0
if corr_results and len(corr_results) > 0:
    best_corr_val = max(
        [v.get('best_value', 0) for v in corr_results.values()],
        key=abs)

llm_sent = "N/A"
if df_relevant is not None and 'llm_sentiment' in df_relevant.columns:
    llm_sent = df_relevant['llm_sentiment'].mode()[0]
elif df_relevant is not None and 'llm_label' in df_relevant.columns:
    llm_sent = df_relevant['llm_label'].mode()[0]

direct_count   = (df_relevant['relevance_class'] == 'Direct').sum()
indirect_count = (df_relevant['relevance_class'] == 'Indirect').sum()

for col, label, value in [
    (m1, "Articles",      f"{len(df_relevant):,}"),
    (m2, "Trading Days",  f"{len(df_daily)}"),
    (m3, "Direct News",   f"{direct_count:,}"),
    (m4, "Indirect News", f"{indirect_count:,}"),
    (m5, "LLM Sentiment", llm_sent),
    (m6, "Best Corr",     f"{best_corr_val:+.3f}"),
]:
    col.metric(label, value)

st.markdown("---")


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sentiment & Price",
    "🔥 Correlation",
    "📰 News Feed",
    "📋 Findings Report",
    "📉 Stock Trend",
])

charts = generate_all_charts(
    df_relevant, df_daily, df_merged,
    display_symbol, display_date_from, display_date_to
)


# ════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("#### Daily Sentiment vs Price Movement")
        if 'sentiment_vs_price' in charts:
            st.pyplot(charts['sentiment_vs_price'])
    with col_r:
        st.markdown("#### Model Comparison")
        if 'distribution' in charts:
            st.pyplot(charts['distribution'])
        st.markdown("**Score Ranges**")
        score_rows = []
        for model, col in [
            ('VADER',   'vader_compound'),
            ('FinBERT', 'finbert_compound'),
            ('LLM',     'llm_score'),
        ]:
            if col in df_relevant.columns:
                score_rows.append({
                    'Model': model,
                    'Min'  : round(df_relevant[col].min(), 3),
                    'Max'  : round(df_relevant[col].max(), 3),
                    'Avg'  : round(df_relevant[col].mean(), 3),
                })
        if score_rows:
            st.dataframe(pd.DataFrame(score_rows),
                         hide_index=True,
                         use_container_width=True)
    st.markdown("#### Article Volume per Trading Day")
    if 'article_volume' in charts:
        st.pyplot(charts['article_volume'])


# ════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Correlation Heatmap")
        if 'correlation_heatmap' in charts:
            st.pyplot(charts['correlation_heatmap'])
    with col2:
        st.markdown("#### Results Table")
        if corr_results and len(corr_results) > 0:
            corr_df = pd.DataFrame([
                {
                    'Model'     : model,
                    'Same-day'  : round(vals['same_day'], 3),
                    'Lag-1'     : round(vals['lag_1'], 3),
                    'Lag-2'     : round(vals['lag_2'], 3),
                    'Best Lag'  : vals.get('best_lag', ''),
                    'Best Value': round(vals.get('best_value', 0), 3),
                }
                for model, vals in corr_results.items()
            ])
            st.dataframe(corr_df, hide_index=True,
                         use_container_width=True)
        st.markdown("#### Scatter Plot")
        if 'scatter' in charts:
            st.pyplot(charts['scatter'])


# ════════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### News Feed — Classified + Scored")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_rel = st.multiselect(
            "Relevance", ["Direct", "Indirect"],
            default=["Direct", "Indirect"])
    with fc2:
        filter_sent = st.multiselect(
            "Sentiment", ["Positive", "Negative", "Neutral"],
            default=["Positive", "Negative", "Neutral"])
    with fc3:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Score (highest)", "Score (lowest)"])

    df_feed = df_relevant.copy()
    if filter_rel:
        df_feed = df_feed[df_feed['relevance_class'].isin(filter_rel)]

    # Handle both llm_sentiment and llm_label column names
    sent_col = 'llm_sentiment' if 'llm_sentiment' in df_feed.columns \
               else 'llm_label'
    if filter_sent and sent_col in df_feed.columns:
        df_feed = df_feed[df_feed[sent_col].isin(filter_sent)]

    if sort_by == "Date (newest)":
        df_feed = df_feed.sort_values('trading_date', ascending=False)
    elif sort_by == "Score (highest)":
        df_feed = df_feed.sort_values('llm_score', ascending=False)
    else:
        df_feed = df_feed.sort_values('llm_score')

    st.caption(f"Showing {len(df_feed)} articles")

    for _, row in df_feed.head(50).iterrows():
        relevance = row.get('relevance_class', '')
        sentiment = row.get(sent_col, 'Neutral')
        score     = float(row.get('llm_score', 0) or 0)
        headline  = row.get('headline', '')
        source    = row.get('source', '')
        date      = str(row.get('trading_date', ''))
        reason    = row.get('llm_reason', '')

        border_color   = ('#66bb6a' if sentiment == 'Positive'
                          else '#ef5350' if sentiment == 'Negative'
                          else '#78909c')
        score_color    = '#66bb6a' if score > 0 else '#ef5350'
        rel_color      = '#1a3a5c' if relevance == 'Direct' else '#3a2a0a'
        rel_text_color = '#4fc3f7' if relevance == 'Direct' else '#ffb74d'

        st.markdown(f"""
        <div style="background:#1a1a1a;
                    border-left:3px solid {border_color};
                    padding:10px 14px; margin-bottom:8px;
                    border-radius:0 6px 6px 0;">
            <div style="display:flex; justify-content:space-between;
                        align-items:flex-start;">
                <div style="flex:1;">
                    <span style="background:{rel_color};
                                 color:{rel_text_color};
                                 padding:2px 8px; border-radius:12px;
                                 font-size:11px; font-weight:bold;
                                 margin-right:6px;">{relevance}</span>
                    <span style="background:#1a1a1a; color:{border_color};
                                 padding:2px 8px; border-radius:12px;
                                 font-size:11px; font-weight:bold;
                                 border:1px solid {border_color};">
                                 {sentiment}</span>
                    <span style="font-size:11px; color:#666;
                                 margin-left:8px;">{date} • {source}</span>
                    <div style="color:#fff; font-size:14px;
                                margin-top:6px; font-weight:500;">
                                {headline}</div>
                    <div style="color:#888; font-size:11px;
                                margin-top:4px;">💬 {reason}</div>
                </div>
                <div style="text-align:right; min-width:60px;
                            margin-left:16px;">
                    <span style="font-size:20px; font-weight:bold;
                                 color:{score_color};">{score:+.2f}</span>
                    <div style="font-size:10px; color:#666;">LLM Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 4
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 📋 AI-Generated Findings Report (GPT-4o)")
    run_id      = st.session_state.run_id
    report_text = get_findings(run_id) if run_id else None

    if report_text:
        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            st.markdown(report_text)
        with col_r2:
            st.markdown("**Report Metadata**")
            st.markdown(f"""
| | |
|---|---|
| Model | GPT-4o |
| Symbol | {display_symbol} |
| Period | {display_date_from} → {display_date_to} |
| Articles | {len(df_relevant):,} |
| Trading Days | {len(df_daily)} |
            """)
            st.download_button(
                "📥 Download Report (TXT)",
                data=report_text,
                file_name=f"{display_symbol}_report_{display_date_from}.txt",
                mime="text/plain",
                use_container_width=True)
            st.download_button(
                "📥 Download Sentiment Data (CSV)",
                data=df_relevant.to_csv(index=False),
                file_name=f"{display_symbol}_sentiment_{display_date_from}.csv",
                mime="text/csv",
                use_container_width=True)
    else:
        st.info("Report will appear here after analysis completes.")


# ════════════════════════════════════════════════════════════
# TAB 5
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"#### {display_symbol} — 1 Year Price History")
    df_hist = get_historical_prices(display_symbol)

    if df_hist.empty:
        st.info(f"No historical data for {display_symbol}.")
    else:
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        current_price = df_hist['close'].iloc[-1]
        start_price   = df_hist['close'].iloc[0]
        yr_return     = ((current_price / start_price) - 1) * 100
        yr_high       = df_hist['high'].max()
        yr_low        = df_hist['low'].min()
        avg_volume    = df_hist['volume'].mean()

        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Current Price", f"${current_price:.2f}")
        h2.metric("1Y Return",     f"{yr_return:+.2f}%")
        h3.metric("52W High",      f"${yr_high:.2f}")
        h4.metric("52W Low",       f"${yr_low:.2f}")
        h5.metric("Avg Volume",    f"{avg_volume/1e6:.1f}M")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True)
        fig.patch.set_facecolor('#0f0f0f')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#ccc')
            ax.grid(True, alpha=0.2, color='#333')

        ax1.plot(df_hist['date'], df_hist['close'],
                 color='#4fc3f7', linewidth=1.5, label='Close Price')
        ax1.fill_between(df_hist['date'], df_hist['close'],
                         df_hist['close'].min(),
                         color='#4fc3f7', alpha=0.1)

        df_hist['ma20'] = df_hist['close'].rolling(20).mean()
        df_hist['ma50'] = df_hist['close'].rolling(50).mean()
        ax1.plot(df_hist['date'], df_hist['ma20'],
                 color='#ffb74d', linewidth=1,
                 linestyle='--', label='MA20', alpha=0.8)
        ax1.plot(df_hist['date'], df_hist['ma50'],
                 color='#81c784', linewidth=1,
                 linestyle='--', label='MA50', alpha=0.8)

        if df_merged is not None and not df_merged.empty:
            try:
                window_start = pd.to_datetime(df_merged['date'].min())
                window_end   = pd.to_datetime(df_merged['date'].max())
                ax1.axvspan(window_start, window_end,
                            color='#ffb74d', alpha=0.1,
                            label='Analysis window')
                ax1.axvline(window_start, color='#ffb74d',
                            linewidth=0.8, linestyle=':')
                ax1.axvline(window_end, color='#ffb74d',
                            linewidth=0.8, linestyle=':')
            except Exception:
                pass

        ax1.set_ylabel('Price ($)', color='#ccc')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.set_title(
            f'{display_symbol} — 1 Year Price History  '
            f'${start_price:.2f} → ${current_price:.2f} '
            f'({yr_return:+.2f}%)',
            color='#ccc', fontsize=12)

        vol_colors = ['#66bb6a' if r > 0 else '#ef5350'
                      for r in df_hist['price_change_pct'].fillna(0)]
        ax2.bar(df_hist['date'], df_hist['volume'] / 1e6,
                color=vol_colors, alpha=0.7, width=1)
        ax2.set_ylabel('Volume (M)', color='#ccc')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, color='#ccc')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Daily Returns Distribution")
        fig2, ax = plt.subplots(figsize=(10, 3))
        fig2.patch.set_facecolor('#0f0f0f')
        ax.set_facecolor('#1a1a1a')
        returns = df_hist['price_change_pct'].dropna()
        ax.hist(returns, bins=50, color='#4fc3f7',
                alpha=0.7, edgecolor='#333')
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
        ax.axvline(returns.mean(), color='#ffb74d',
                   linewidth=1.5, linestyle='--',
                   label=f'Mean: {returns.mean():.2f}%')
        ax.set_xlabel('Daily Return %', color='#ccc')
        ax.set_ylabel('Frequency', color='#ccc')
        ax.tick_params(colors='#ccc')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, color='#333')
        ax.set_title('Distribution of Daily Returns', color='#ccc')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        with st.expander("📊 View Raw Price Data"):
            st.dataframe(
                df_hist[['date', 'open', 'high', 'low', 'close',
                          'volume', 'price_change_pct',
                          'price_direction']]
                .sort_values('date', ascending=False).round(2),
                hide_index=True, use_container_width=True)