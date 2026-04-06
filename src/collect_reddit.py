"""
Optional: live data collection from Reddit via PRAW.
Requires valid API credentials set in environment variables:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Usage:
    python src/collect_reddit.py

Collected data is saved to data/reddit_posts.csv in the same schema
used by generate_data.py, so the full pipeline works with either source.
"""

import os
import sys
import pandas as pd
from datetime import datetime

try:
    import praw
except ImportError:
    print("praw not installed. Run: pip install praw")
    sys.exit(1)

# Election windows
PRE_START  = datetime(2024, 8,  7).timestamp()
PRE_END    = datetime(2024, 11, 4).timestamp()
POST_START = datetime(2024, 11, 6).timestamp()
POST_END   = datetime(2025, 2,  3).timestamp()

SUBREDDITS = ["conservative", "liberal", "politics", "neutralpolitics"]
LIMIT_PER_SUBREDDIT = 500   # adjust as needed


def in_window(ts: float) -> str | None:
    if PRE_START <= ts <= PRE_END:
        return "pre"
    if POST_START <= ts <= POST_END:
        return "post"
    return None


def collect(reddit: praw.Reddit) -> pd.DataFrame:
    rows = []
    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        print(f"Collecting from r/{sub_name}...")
        for submission in subreddit.top(time_filter="year", limit=LIMIT_PER_SUBREDDIT):
            period = in_window(submission.created_utc)
            if period is None:
                continue
            rows.append({
                "post_id":      submission.id,
                "subreddit":    sub_name,
                "period":       period,
                "timestamp":    pd.to_datetime(submission.created_utc, unit="s"),
                "text":         submission.title + " " + (submission.selftext or ""),
                "score":        submission.score,
                "num_comments": submission.num_comments,
            })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    client_id     = os.environ.get("REDDIT_CLIENT_ID",     "YOUR_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
    user_agent    = os.environ.get("REDDIT_USER_AGENT",    "cs470-project by u/your_username")

    if "YOUR_" in client_id:
        print("Set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT "
              "environment variables before running.")
        sys.exit(1)

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    df = collect(reddit)
    out = "data/reddit_posts.csv"
    df.to_csv(out, index=False)
    print(f"Collected {len(df)} posts → {out}")
    print(df.groupby(["subreddit", "period"]).size())
