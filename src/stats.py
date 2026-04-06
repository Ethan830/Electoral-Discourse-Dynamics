"""
Statistical analysis: t-tests, effect sizes, engagement metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d effect size."""
    n1, n2 = len(a), len(b)
    s_pool = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * b.std(ddof=1)**2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / (s_pool + 1e-12)


def sentiment_shift_tests(df: pd.DataFrame, score_col: str = "vader_compound") -> pd.DataFrame:
    """
    For each subreddit, run an independent-samples t-test comparing
    pre-election vs post-election sentiment scores.
    Returns a summary DataFrame.
    """
    records = []
    for sub in sorted(df["subreddit"].unique()):
        sub_df = df[df["subreddit"] == sub]
        pre  = sub_df[sub_df["period"] == "pre" ][score_col].values
        post = sub_df[sub_df["period"] == "post"][score_col].values

        t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
        d = cohens_d(post, pre)   # positive d means post > pre

        records.append({
            "subreddit":   sub,
            "pre_mean":    pre.mean(),
            "post_mean":   post.mean(),
            "delta":       post.mean() - pre.mean(),
            "t_stat":      t_stat,
            "p_value":     p_val,
            "cohens_d":    d,
            "significant": p_val < 0.05,
        })
    return pd.DataFrame(records)


def engagement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean score and comment count per subreddit/period."""
    return (
        df.groupby(["subreddit", "period"])
          .agg(
              mean_score    = ("score",        "mean"),
              mean_comments = ("num_comments", "mean"),
              post_count    = ("post_id",      "count"),
          )
          .reset_index()
    )


def weekly_sentiment(df: pd.DataFrame, score_col: str = "vader_compound") -> pd.DataFrame:
    """Daily mean sentiment per subreddit for time-series plotting."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.floor("D")
    return (
        df.groupby(["subreddit", "date"])[score_col]
          .mean()
          .reset_index()
          .rename(columns={score_col: "mean_sentiment"})
    )


def topic_chi_square(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of topic distribution independence per subreddit."""
    records = []
    for sub in sorted(df["subreddit"].unique()):
        sub_df = df[df["subreddit"] == sub]
        pre_counts  = sub_df[sub_df["period"] == "pre" ]["dominant_topic"].value_counts()
        post_counts = sub_df[sub_df["period"] == "post"]["dominant_topic"].value_counts()
        all_topics = sorted(set(pre_counts.index) | set(post_counts.index))
        table = np.array([
            [pre_counts.get(t, 0)  for t in all_topics],
            [post_counts.get(t, 0) for t in all_topics],
        ])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        records.append({"subreddit": sub, "chi2": chi2, "p_value": p, "dof": dof})
    return pd.DataFrame(records)
