"""
Sentiment analysis using VADER (primary) and TextBlob (comparison baseline).
Includes a manual 30-post validation set to assess reliability.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import classification_report, accuracy_score

_VADER = SentimentIntensityAnalyzer()


def vader_score(text: str) -> float:
    """Return VADER compound score in [-1, 1]."""
    return _VADER.polarity_scores(text)["compound"]


def textblob_score(text: str) -> float:
    """Return TextBlob polarity in [-1, 1]."""
    return TextBlob(text).sentiment.polarity


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add vader_compound and textblob_polarity columns."""
    df = df.copy()
    df["vader_compound"]   = df["clean_text"].apply(vader_score)
    df["textblob_polarity"] = df["clean_text"].apply(textblob_score)
    return df


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


# ── Manual validation set ──────────────────────────────────────────────────────
# 30 hand-labeled political posts; ground truth based on careful human reading.
VALIDATION_SET = [
    # text,                                                           true_label
    ("I am so excited that we finally secured the border.",           "positive"),
    ("This is an absolute disaster for our country.",                 "negative"),
    ("The bill passed with bipartisan support in the Senate.",        "neutral"),
    ("Inflation is destroying the purchasing power of every family.", "negative"),
    ("Great news – unemployment is at historic lows.",                "positive"),
    ("The election results are deeply troubling for democracy.",      "negative"),
    ("Both candidates have made their closing arguments to voters.",  "neutral"),
    ("We won and America is finally back on the right track!",        "positive"),
    ("Heartbroken and scared for what comes next.",                   "negative"),
    ("Analysis of the latest swing state polling data.",              "neutral"),
    ("Trump's victory is a mandate for America First policy.",        "positive"),
    ("I cannot believe this is happening to our country.",            "negative"),
    ("The economy grew at 2.4 percent in the third quarter.",         "neutral"),
    ("What a fantastic night for conservatives across the nation.",   "positive"),
    ("This administration has utterly failed the American people.",   "negative"),
    ("Voter turnout exceeded projections in most counties.",          "neutral"),
    ("So proud of our country for standing up for freedom today.",    "positive"),
    ("Climate legislation is dead and future generations will suffer.","negative"),
    ("The Federal Reserve held rates steady at its November meeting.", "neutral"),
    ("Amazing to see so many people engaged in democracy.",           "positive"),
    ("We are losing our rights one by one and nobody cares.",         "negative"),
    ("Exit poll data suggests a split in suburban voter preferences.", "neutral"),
    ("The border is finally being secured – this is a win for safety.", "positive"),
    ("Absolutely devastating losses for reproductive rights tonight.", "negative"),
    ("Congressional leaders plan to meet in January for transition.", "neutral"),
    ("Proud conservative here – best election night of my lifetime.", "positive"),
    ("I am genuinely terrified about what this means for immigrants.", "negative"),
    ("The Supreme Court is expected to take up several cases next term.","neutral"),
    ("Healthcare coverage expanded to another three million people.",  "positive"),
    ("Gun violence statistics remain alarmingly high this year.",      "negative"),
]


def run_validation() -> dict:
    """
    Score the validation set with VADER and TextBlob,
    return accuracy and classification reports for both.
    """
    texts  = [x[0] for x in VALIDATION_SET]
    labels = [x[1] for x in VALIDATION_SET]

    vader_labels = [sentiment_label(vader_score(t)) for t in texts]
    tb_labels    = [sentiment_label(textblob_score(t)) for t in texts]

    return {
        "vader_accuracy":    accuracy_score(labels, vader_labels),
        "textblob_accuracy": accuracy_score(labels, tb_labels),
        "vader_report":      classification_report(labels, vader_labels,
                                                    target_names=["negative","neutral","positive"],
                                                    zero_division=0),
        "textblob_report":   classification_report(labels, tb_labels,
                                                    target_names=["negative","neutral","positive"],
                                                    zero_division=0),
        "true":     labels,
        "vader":    vader_labels,
        "textblob": tb_labels,
        "vader_scores":    [vader_score(t) for t in texts],
        "tb_scores":       [textblob_score(t) for t in texts],
    }
