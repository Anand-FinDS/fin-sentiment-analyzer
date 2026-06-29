# charts/visualizations.py
# ════════════════════════════════════════════════════════════
# GenAI Financial Sentiment Analyzer
# All chart functions — called by Streamlit UI
# Returns matplotlib figures (not plt.show())
# ════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns


# ── Theme ─────────────────────────────────────────────────────
DARK_BG    = '#0f0f0f'
PANEL_BG   = '#1a1a1a'
GRID_COLOR = '#2a2a2a'
TEXT_COLOR = '#cccccc'

COLORS = {
    'vader'  : '#4fc3f7',   # blue
    'finbert': '#81c784',   # green
    'llm'    : '#ffb74d',   # orange
    'price'  : '#ef5350',   # red
    'pos'    : '#66bb6a',   # green
    'neg'    : '#ef5350',   # red
    'neu'    : '#78909c',   # grey
    'direct' : '#ab47bc',   # purple
    'indirect': '#26c6da',  # cyan
}


def apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor' : DARK_BG,
        'axes.facecolor'   : PANEL_BG,
        'axes.edgecolor'   : '#333333',
        'axes.labelcolor'  : TEXT_COLOR,
        'text.color'       : TEXT_COLOR,
        'xtick.color'      : TEXT_COLOR,
        'ytick.color'      : TEXT_COLOR,
        'grid.color'       : GRID_COLOR,
        'grid.linewidth'   : 0.5,
        'font.family'      : 'monospace',
        'legend.facecolor' : PANEL_BG,
        'legend.edgecolor' : '#333333',
    })


# ════════════════════════════════════════════════════════════
# CHART 1 — Dual Axis: Sentiment vs Price over time
# ════════════════════════════════════════════════════════════

def chart_sentiment_vs_price(df_merged: pd.DataFrame,
                              symbol: str,
                              date_from: str,
                              date_to: str) -> plt.Figure:
    """
    Triple panel chart — one per model.
    Each panel: sentiment line + price bars (dual axis).
    """
    apply_dark_theme()
    dates  = pd.to_datetime(df_merged['date'])
    models = [
        ('vader',   'VADER',   'vader_compound'),
        ('finbert', 'FinBERT', 'finbert_compound'),
        ('llm',     'LLM',     'llm_compound'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f'{symbol} — Sentiment vs Price Movement\n'
        f'{date_from} → {date_to}',
        fontsize=14, fontweight='bold', y=0.98,
        color=TEXT_COLOR
    )

    for ax, (key, label, col) in zip(axes, models):
        # Price bars
        ax2 = ax.twinx()
        bar_colors = [
            COLORS['pos'] if x > 0 else COLORS['neg']
            for x in df_merged['price_change_pct'].fillna(0)
        ]
        ax2.bar(dates, df_merged['price_change_pct'],
                color=bar_colors, alpha=0.3,
                width=0.6, label='Price Change %')
        ax2.set_ylabel('Price Change %',
                       color=COLORS['price'], fontsize=9)
        ax2.tick_params(axis='y', colors=COLORS['price'])
        ax2.axhline(0, color=COLORS['price'],
                    linewidth=0.5, alpha=0.5)

        # Sentiment line
        ax.plot(dates, df_merged[col],
                color=COLORS[key], linewidth=2,
                label=label, zorder=3)
        ax.fill_between(dates, df_merged[col], 0,
                        color=COLORS[key], alpha=0.1)
        ax.axhline(0,     color='#555555',
                   linewidth=0.8, linestyle='--')
        ax.axhline(0.05,  color='#555555',
                   linewidth=0.4, linestyle=':')
        ax.axhline(-0.05, color='#555555',
                   linewidth=0.4, linestyle=':')
        ax.set_ylabel(f'{label} Score',
                      color=COLORS[key], fontsize=9)
        ax.tick_params(axis='y', colors=COLORS[key])
        ax.grid(True, alpha=0.3)

        # Correlation annotation
        corr = df_merged[col].corr(
            df_merged['price_change_pct']
        )
        ax.text(0.02, 0.88, f'r = {corr:+.3f}',
                transform=ax.transAxes, fontsize=9,
                color=COLORS[key],
                bbox=dict(boxstyle='round',
                          facecolor=PANEL_BG,
                          edgecolor=COLORS[key],
                          alpha=0.8))
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter('%b %d'))
    axes[-1].xaxis.set_major_locator(
        mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# CHART 2 — Correlation Heatmap
# ════════════════════════════════════════════════════════════

def chart_correlation_heatmap(df_merged: pd.DataFrame,
                               symbol: str) -> plt.Figure:
    """
    3×3 heatmap — models vs lag windows.
    Green = positive, Red = negative correlation.
    """
    apply_dark_theme()

    corr_data = {}
    for model, col in [
        ('VADER',   'vader_compound'),
        ('FinBERT', 'finbert_compound'),
        ('LLM',     'llm_compound'),
    ]:
        if col not in df_merged.columns:
            continue
        lag1_col = col.replace('compound', 'lag1').replace(
            'llm_compound', 'llm_lag1')
        lag2_col = col.replace('compound', 'lag2').replace(
            'llm_compound', 'llm_lag2')

        # Add lags if not present
        if lag1_col not in df_merged.columns:
            df_merged[lag1_col] = df_merged[col].shift(1)
        if lag2_col not in df_merged.columns:
            df_merged[lag2_col] = df_merged[col].shift(2)

        price = df_merged['price_change_pct']
        corr_data[model] = {
            'Same-day': df_merged[col].corr(price),
            'Lag-1'   : df_merged[lag1_col].corr(price),
            'Lag-2'   : df_merged[lag2_col].corr(price),
        }

    corr_df = pd.DataFrame(corr_data).T

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    sns.heatmap(
        corr_df,
        annot=True, fmt='.3f',
        cmap='RdYlGn', center=0,
        vmin=-0.5, vmax=0.5,
        linewidths=0.5,
        annot_kws={'size': 12, 'weight': 'bold'},
        ax=ax
    )
    ax.set_title(
        f'{symbol} Sentiment-Price Correlation\n'
        f'(Pearson r, {len(df_merged)} trading days)',
        fontsize=13, fontweight='bold', pad=15,
        color=TEXT_COLOR
    )
    ax.set_xlabel('Lag Window', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# CHART 3 — Daily Model Comparison (grouped bar)
# ════════════════════════════════════════════════════════════

def chart_model_comparison(df_merged: pd.DataFrame,
                            symbol: str,
                            date_from: str,
                            date_to: str) -> plt.Figure:
    """
    Grouped bar chart — VADER vs FinBERT vs LLM per trading day.
    Shows score variance across models clearly.
    """
    apply_dark_theme()

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    x     = np.arange(len(df_merged))
    width = 0.25

    ax.bar(x - width, df_merged['vader_compound'],
           width, label='VADER',
           color=COLORS['vader'], alpha=0.85)
    ax.bar(x,         df_merged['finbert_compound'],
           width, label='FinBERT',
           color=COLORS['finbert'], alpha=0.85)
    ax.bar(x + width, df_merged['llm_compound'],
           width, label='LLM',
           color=COLORS['llm'], alpha=0.85)

    ax.axhline(0,    color='#555555',
               linewidth=0.8, linestyle='--')
    ax.axhline(0.05, color='#555555',
               linewidth=0.4, linestyle=':')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(d)[5:] for d in df_merged['date']],
        rotation=45, ha='right', fontsize=7
    )
    ax.set_ylabel('Sentiment Score', fontsize=11)
    ax.set_title(
        f'{symbol} — Daily Sentiment by Model\n'
        f'{date_from} → {date_to}',
        fontsize=13, fontweight='bold',
        color=TEXT_COLOR
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# CHART 4 — Scatter: Sentiment vs Price Change
# ════════════════════════════════════════════════════════════

def chart_scatter(df_merged: pd.DataFrame,
                  symbol: str) -> plt.Figure:
    """
    3-panel scatter — one per model.
    X = sentiment score, Y = price change %.
    Trend line + correlation shown.
    """
    apply_dark_theme()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        f'{symbol} — Sentiment Score vs Same-Day Price Change',
        fontsize=13, fontweight='bold',
        color=TEXT_COLOR
    )

    models = [
        ('vader',   'VADER',   'vader_compound'),
        ('finbert', 'FinBERT', 'finbert_compound'),
        ('llm',     'LLM',     'llm_compound'),
    ]

    for ax, (key, label, col) in zip(axes, models):
        if col not in df_merged.columns:
            continue

        x    = df_merged[col]
        y    = df_merged['price_change_pct']
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]

        corr = x.corr(y)

        ax.scatter(x, y, color=COLORS[key],
                   alpha=0.7, s=60, zorder=3)

        # Trend line
        if len(x) > 2:
            z      = np.polyfit(x, y, 1)
            p      = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line),
                    color='white', linewidth=1.5,
                    linestyle='--', alpha=0.6)

        ax.axhline(0, color='#555555',
                   linewidth=0.8, linestyle='--')
        ax.axvline(0, color='#555555',
                   linewidth=0.8, linestyle='--')
        ax.set_xlabel(f'{label} Score', fontsize=10)
        ax.set_ylabel('Price Change %', fontsize=10)
        ax.set_title(
            f'{label}\nr = {corr:+.3f}',
            fontsize=11, color=COLORS[key],
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# CHART 5 — Sentiment Distribution (stacked bar)
# ════════════════════════════════════════════════════════════

def chart_distribution(df_relevant: pd.DataFrame,
                        symbol: str,
                        date_from: str,
                        date_to: str) -> plt.Figure:
    """
    Stacked bar — Positive/Neutral/Negative per model.
    Shows how each model distributes sentiment labels.
    """
    apply_dark_theme()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    categories = ['VADER', 'FinBERT', 'LLM']
    cols       = ['vader_label', 'finbert_label', 'llm_sentiment']

    positives, neutrals, negatives = [], [], []
    for col in cols:
        if col in df_relevant.columns:
            vc = df_relevant[col].value_counts()
            positives.append(vc.get('Positive', 0))
            neutrals.append(vc.get('Neutral',  0))
            negatives.append(vc.get('Negative', 0))
        else:
            positives.append(0)
            neutrals.append(0)
            negatives.append(0)

    x = np.arange(len(categories))
    ax.bar(x, positives, label='Positive',
           color=COLORS['pos'], alpha=0.85)
    ax.bar(x, neutrals,  label='Neutral',
           color=COLORS['neu'], alpha=0.85,
           bottom=positives)
    ax.bar(x, negatives, label='Negative',
           color=COLORS['neg'], alpha=0.85,
           bottom=[p + n for p, n in zip(positives, neutrals)])

    # Count labels inside bars
    for i, (p, nu, ne) in enumerate(
            zip(positives, neutrals, negatives)):
        if p  > 0: ax.text(i, p/2,        str(int(p)),
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color='white')
        if nu > 0: ax.text(i, p + nu/2,   str(int(nu)),
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color='white')
        if ne > 0: ax.text(i, p + nu + ne/2, str(int(ne)),
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('Article Count', fontsize=11)
    ax.set_title(
        f'{symbol} — Sentiment Distribution by Model\n'
        f'{date_from} → {date_to}',
        fontsize=13, fontweight='bold',
        color=TEXT_COLOR
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# CHART 6 — Article Volume + Direct/Indirect split
# ════════════════════════════════════════════════════════════

def chart_article_volume(df_daily: pd.DataFrame,
                          symbol: str) -> plt.Figure:
    """
    Bar chart showing article volume per trading day.
    Stacked by Direct vs Indirect classification.
    Helps analyst see news density over time.
    """
    apply_dark_theme()

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor(DARK_BG)

    dates    = range(len(df_daily))
    direct   = df_daily['direct_count'].fillna(0)
    indirect = df_daily['indirect_count'].fillna(0)

    ax.bar(dates, direct,
           label='Direct',   color=COLORS['direct'],  alpha=0.85)
    ax.bar(dates, indirect,
           label='Indirect', color=COLORS['indirect'], alpha=0.85,
           bottom=direct)

    ax.set_xticks(list(dates))
    ax.set_xticklabels(
        [str(d)[5:] for d in df_daily['date']],
        rotation=45, ha='right', fontsize=7
    )
    ax.set_ylabel('Article Count', fontsize=10)
    ax.set_title(
        f'{symbol} — Daily Article Volume '
        f'(Direct vs Indirect)',
        fontsize=12, fontweight='bold',
        color=TEXT_COLOR
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# GENERATE ALL CHARTS — called by app.py
# ════════════════════════════════════════════════════════════

def generate_all_charts(df_relevant: pd.DataFrame,
                         df_daily: pd.DataFrame,
                         df_merged: pd.DataFrame,
                         symbol: str,
                         date_from: str,
                         date_to: str) -> dict:
    """
    Generate all charts and return as dict of figures.
    Called by Streamlit app.py — renders with st.pyplot(fig).

    Returns:
        {
            'sentiment_vs_price' : fig,
            'correlation_heatmap': fig,
            'model_comparison'   : fig,
            'scatter'            : fig,
            'distribution'       : fig,
            'article_volume'     : fig,
        }
    """
    print("Generating charts...")
    charts = {}

    try:
        charts['sentiment_vs_price'] = chart_sentiment_vs_price(
            df_merged, symbol, date_from, date_to)
        print("  ✓ Chart 1 — Sentiment vs Price")
    except Exception as e:
        print(f"  ✗ Chart 1 failed: {e}")

    try:
        charts['correlation_heatmap'] = chart_correlation_heatmap(
            df_merged.copy(), symbol)
        print("  ✓ Chart 2 — Correlation Heatmap")
    except Exception as e:
        print(f"  ✗ Chart 2 failed: {e}")

    try:
        charts['model_comparison'] = chart_model_comparison(
            df_merged, symbol, date_from, date_to)
        print("  ✓ Chart 3 — Model Comparison")
    except Exception as e:
        print(f"  ✗ Chart 3 failed: {e}")

    try:
        charts['scatter'] = chart_scatter(df_merged, symbol)
        print("  ✓ Chart 4 — Scatter")
    except Exception as e:
        print(f"  ✗ Chart 4 failed: {e}")

    try:
        charts['distribution'] = chart_distribution(
            df_relevant, symbol, date_from, date_to)
        print("  ✓ Chart 5 — Distribution")
    except Exception as e:
        print(f"  ✗ Chart 5 failed: {e}")

    try:
        charts['article_volume'] = chart_article_volume(
            df_daily, symbol)
        print("  ✓ Chart 6 — Article Volume")
    except Exception as e:
        print(f"  ✗ Chart 6 failed: {e}")

    print(f"\n✓ {len(charts)}/6 charts generated")
    return charts