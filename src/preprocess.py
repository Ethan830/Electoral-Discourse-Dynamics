"""
Text preprocessing: cleaning, tokenization, lemmatization, bot filtering.
"""

import re
import string
import nltk
import pandas as pd

# Download NLTK data (idempotent)
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english")) | {
    "would", "could", "should", "also", "get", "got", "getting",
    "one", "like", "really", "think", "know", "right", "say",
    "going", "make", "just", "even", "much", "many", "well", "way",
}

_LEMMATIZER = WordNetLemmatizer()

URL_RE    = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
PUNCT_RE  = re.compile(r"[" + re.escape(string.punctuation) + r"]+")
SPACE_RE  = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return lightly-cleaned text suitable for VADER (preserves case & punctuation)."""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, lemmatize, remove stopwords."""
    text = clean_text(text).lower()
    text = PUNCT_RE.sub(" ", text)
    tokens = text.split()
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return tokens


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'clean_text' and 'tokens' columns; apply bot filter."""
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"]     = df["text"].apply(tokenize)
    # Bot heuristic: remove posts with score <= 0 or text shorter than 10 chars
    df = df[df["score"] > 0]
    df = df[df["text"].str.len() >= 10]
    df = df.reset_index(drop=True)
    return df
