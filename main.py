"""
CS 470 Project – 2024 Election Sentiment Analysis
Main pipeline: generate data → preprocess → sentiment → topics → stats → figures

Run:
    python main.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/reddit_posts.csv")


def main():
    print("=" * 60)
    print("CS 470 – 2024 Election Sentiment Analysis Pipeline")
    print("=" * 60)

    # ── Step 1: Data ─────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        print(f"  Loaded {len(df)} posts from {DATA_PATH}")
    else:
        print("  No CSV found – generating synthetic dataset...")
        from generate_data import generate_dataset
        df = generate_dataset(n_per_group=350, seed=42)
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"  Generated {len(df)} posts -> {DATA_PATH}")

    print(df.groupby(["subreddit", "period"]).size().to_string())

    # ── Step 2: Preprocess ───────────────────────────────────────
    print("\n[2/6] Preprocessing text...")
    from preprocess import preprocess_dataframe
    df = preprocess_dataframe(df)
    print(f"  After cleaning: {len(df)} posts remain")

    # ── Step 3: Sentiment analysis ───────────────────────────────
    print("\n[3/6] Running sentiment analysis (VADER + TextBlob)...")
    from sentiment import score_dataframe, run_validation
    df = score_dataframe(df)
    val = run_validation()
    print(f"  VADER accuracy on validation set:    {val['vader_accuracy']:.1%}")
    print(f"  TextBlob accuracy on validation set: {val['textblob_accuracy']:.1%}")
    print("\n  VADER classification report:")
    print(val["vader_report"])

    # ── Step 4: Topic modeling ───────────────────────────────────
    print("\n[4/6] Running LDA topic modeling (6 topics, 10 passes)...")
    from topics import run_topic_modeling, TOPIC_NAMES
    model, vectorizer, dtm, topic_matrix, topic_name_map, df = run_topic_modeling(df)
    feature_names = vectorizer.get_feature_names_out()
    print("  Topic mapping:")
    for tid, name in topic_name_map.items():
        top_idx   = model.components_[tid].argsort()[-6:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        print(f"    Topic {tid} -> {name}: {', '.join(top_words)}")

    # ── Step 5: Statistical analysis ─────────────────────────────
    print("\n[5/6] Statistical analysis...")
    from stats import (sentiment_shift_tests, engagement_summary,
                       weekly_sentiment, topic_chi_square)

    shift_df = sentiment_shift_tests(df)
    eng_df   = engagement_summary(df)
    ts_df    = weekly_sentiment(df)
    chi_df   = topic_chi_square(df)

    print("\n  Sentiment shift (t-test, VADER compound):")
    print(shift_df.to_string(index=False, float_format="{:.4f}".format))

    print("\n  Topic distribution chi-square tests:")
    print(chi_df.to_string(index=False, float_format="{:.4f}".format))

    # ── Step 6: Visualizations ───────────────────────────────────
    print("\n[6/6] Generating figures...")
    from visualize import (fig1_sentiment_timeseries, fig2_prepost_comparison,
                           fig3_topic_heatmap, fig4_engagement, fig5_validation)

    topic_names_ordered = [topic_name_map[i] for i in range(len(topic_name_map))]

    fig1_sentiment_timeseries(ts_df)
    fig2_prepost_comparison(shift_df)
    fig3_topic_heatmap(df, topic_names_ordered)
    fig4_engagement(eng_df)
    fig5_validation(val)

    # Save processed data for reference
    df.drop(columns=["topic_vector", "tokens"], errors="ignore").to_csv(
        "data/reddit_processed.csv", index=False
    )

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for _, row in shift_df.iterrows():
        sig = "[SIG]" if row["significant"] else "[ns]"
        direction = "more positive" if row["delta"] > 0 else "more negative"
        print(f"  r/{row['subreddit']:20s}: delta={row['delta']:+.3f}  "
              f"cohen_d={row['cohens_d']:+.3f}  p={row['p_value']:.4f}  "
              f"{sig}  ({direction})")
    print("\nFigures saved to figures/")
    print("Compile report: cd report && pdflatex main.tex")
    print("=" * 60)

    return df, shift_df, eng_df, val


if __name__ == "__main__":
    main()
