"""
Visualization module: generates and saves all figures for the report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

SUBREDDIT_COLORS = {
    "conservative":    "#C0392B",
    "liberal":         "#2980B9",
    "politics":        "#8E44AD",
    "neutralpolitics": "#27AE60",
}

SUBREDDIT_LABELS = {
    "conservative":    "r/conservative",
    "liberal":         "r/liberal",
    "politics":        "r/politics",
    "neutralpolitics": "r/neutralpolitics",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def fig1_sentiment_timeseries(ts_df: pd.DataFrame, score_col: str = "mean_sentiment"):
    """Figure 1 – Daily sentiment time series per subreddit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes = axes.flatten()
    election_date = pd.Timestamp("2024-11-05")

    for ax, sub in zip(axes, ["conservative", "liberal", "politics", "neutralpolitics"]):
        sub_df = ts_df[ts_df["subreddit"] == sub].copy()
        sub_df = sub_df.set_index("date").sort_index()

        # 7-day rolling average
        rolled = sub_df[score_col].rolling(7, min_periods=1).mean()
        ax.plot(rolled.index, rolled.values,
                color=SUBREDDIT_COLORS[sub], linewidth=1.8)
        ax.fill_between(rolled.index, rolled.values, 0,
                        alpha=0.15, color=SUBREDDIT_COLORS[sub])
        ax.axvline(election_date, color="black", linestyle="--",
                   linewidth=1.2, label="Election Day (Nov 5)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(SUBREDDIT_LABELS[sub])
        ax.set_ylabel("Mean VADER Compound")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", rotation=30)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Daily Sentiment Trends Across Political Communities\n"
                 "(7-day rolling average; dashed line = Election Day)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig1_sentiment_timeseries.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig2_prepost_comparison(shift_df: pd.DataFrame):
    """Figure 2 – Pre vs post mean sentiment per subreddit (grouped bar)."""
    subs   = ["conservative", "liberal", "politics", "neutralpolitics"]
    labels = [SUBREDDIT_LABELS[s] for s in subs]
    pre_vals  = [shift_df.loc[shift_df["subreddit"] == s, "pre_mean" ].values[0] for s in subs]
    post_vals = [shift_df.loc[shift_df["subreddit"] == s, "post_mean"].values[0] for s in subs]
    p_vals    = [shift_df.loc[shift_df["subreddit"] == s, "p_value"  ].values[0] for s in subs]

    x     = np.arange(len(subs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_pre  = ax.bar(x - width/2, pre_vals,  width, label="Pre-Election",
                       color=[SUBREDDIT_COLORS[s] for s in subs], alpha=0.55, edgecolor="black")
    bars_post = ax.bar(x + width/2, post_vals, width, label="Post-Election",
                       color=[SUBREDDIT_COLORS[s] for s in subs], alpha=0.95, edgecolor="black")

    # Significance stars
    for i, (pre, post, p) in enumerate(zip(pre_vals, post_vals, p_vals)):
        y_max = max(abs(pre), abs(post)) + 0.04
        star  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(x[i], y_max + 0.01, star, ha="center", va="bottom", fontsize=11)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean VADER Compound Score")
    ax.set_title("Sentiment Comparison: Pre- vs Post-Election\n"
                 "(* p<0.05, ** p<0.01, *** p<0.001)", fontweight="bold")
    ax.legend()
    ax.set_ylim(-0.40, 0.52)
    plt.tight_layout()
    path = FIGURES_DIR / "fig2_prepost_comparison.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig3_topic_heatmap(df: pd.DataFrame, topic_names: list[str]):
    """Figure 3 – Heatmap of mean topic weights by subreddit and period."""
    subs    = ["conservative", "liberal", "politics", "neutralpolitics"]
    periods = ["pre", "post"]
    row_labels = [f"{SUBREDDIT_LABELS[s]}\n({p}-election)" for s in subs for p in periods]

    mat = []
    for sub in subs:
        for period in periods:
            sub_per = df[(df["subreddit"] == sub) & (df["period"] == period)]
            vecs = np.vstack(sub_per["topic_vector"].values)  # (N, K)
            mat.append(vecs.mean(axis=0))
    mat = np.array(mat)   # (8, 6)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.35)
    ax.set_xticks(range(len(topic_names)))
    ax.set_xticklabels(topic_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if mat[i,j] < 0.25 else "white")
    plt.colorbar(im, ax=ax, label="Mean Topic Weight")
    ax.set_title("LDA Topic Distribution by Community and Election Period",
                 fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_topic_heatmap.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig4_engagement(eng_df: pd.DataFrame):
    """Figure 4 – Engagement metrics (mean score & comments) pre vs post."""
    subs    = ["conservative", "liberal", "politics", "neutralpolitics"]
    labels  = [SUBREDDIT_LABELS[s] for s in subs]
    metrics = [("mean_score", "Mean Post Score"), ("mean_comments", "Mean Comment Count")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (col, title) in zip(axes, metrics):
        pre_vals  = [eng_df.loc[(eng_df["subreddit"]==s)&(eng_df["period"]=="pre"),  col].values[0] for s in subs]
        post_vals = [eng_df.loc[(eng_df["subreddit"]==s)&(eng_df["period"]=="post"), col].values[0] for s in subs]
        x     = np.arange(len(subs))
        width = 0.35
        ax.bar(x - width/2, pre_vals,  width, label="Pre-Election",
               color=[SUBREDDIT_COLORS[s] for s in subs], alpha=0.50, edgecolor="black")
        ax.bar(x + width/2, post_vals, width, label="Post-Election",
               color=[SUBREDDIT_COLORS[s] for s in subs], alpha=0.90, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Community Engagement Metrics: Pre- vs Post-Election", fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig4_engagement.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig5_validation(val_results: dict):
    """Figure 5 – VADER vs TextBlob scatter on manual validation set."""
    vader_scores = val_results["vader_scores"]
    tb_scores    = val_results["tb_scores"]
    true_labels  = val_results["true"]
    color_map    = {"positive": "#2ECC71", "negative": "#E74C3C", "neutral": "#95A5A6"}
    colors       = [color_map[l] for l in true_labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(vader_scores, tb_scores, c=colors, s=70, edgecolors="black", linewidths=0.4)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("TextBlob Polarity")
    ax.set_title(
        f"VADER vs TextBlob on Manual Validation Set (n=30)\n"
        f"VADER acc={val_results['vader_accuracy']:.0%}  |  "
        f"TextBlob acc={val_results['textblob_accuracy']:.0%}",
        fontweight="bold",
    )
    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=l.capitalize())
               for l, c in color_map.items()]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "fig5_validation.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path
