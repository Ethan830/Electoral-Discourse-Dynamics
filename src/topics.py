"""
LDA topic modeling with scikit-learn (compatible with Python 3.14).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

N_TOPICS = 6
TOPIC_NAMES = [
    "Immigration & Border",
    "Economy & Inflation",
    "Democracy & Rights",
    "Climate & Environment",
    "Healthcare & Abortion",
    "Election Integrity",
]

# Seed words for topic labeling (lemmatized / stemmed forms)
SEED_WORDS = {
    "Immigration & Border":  {"border", "immigr", "wall", "deportat", "asylum", "illegal", "migrant"},
    "Economy & Inflation":   {"econom", "inflat", "job", "wage", "price", "tax", "recession", "gdp"},
    "Democracy & Rights":    {"democraci", "right", "freedom", "civil", "protest", "autocrat", "fascist"},
    "Climate & Environment": {"climat", "emiss", "green", "environment", "energi", "oil", "fossil"},
    "Healthcare & Abortion": {"healthcar", "abort", "roe", "reproduct", "insur", "medicaid"},
    "Election Integrity":    {"vote", "elect", "ballot", "fraud", "count", "certif", "poll"},
}


def build_dtm(tokens_series: pd.Series):
    """Build a document-term matrix from a Series of token lists."""
    texts = tokens_series.apply(lambda t: " ".join(t))
    vectorizer = CountVectorizer(
        max_df=0.90,
        min_df=3,
        max_features=3000,
        token_pattern=r"[a-zA-Z]{3,}",
    )
    dtm = vectorizer.fit_transform(texts)
    return vectorizer, dtm


def train_lda(dtm, n_topics: int = N_TOPICS, seed: int = 42) -> LatentDirichletAllocation:
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=seed,
        max_iter=15,
        learning_method="online",
        batch_size=256,
    )
    model.fit(dtm)
    return model


def match_topic_names(model: LatentDirichletAllocation,
                      feature_names: list[str]) -> dict[int, str]:
    """Map model topic indices to human-readable names via seed word overlap."""
    mapping = {}
    for tid in range(model.n_components):
        top_idx   = model.components_[tid].argsort()[-20:][::-1]
        top_words = {feature_names[i] for i in top_idx}
        best_name, best_score = f"Topic {tid}", -1
        for name, seeds in SEED_WORDS.items():
            score = len(top_words & seeds)
            if score > best_score:
                best_score, best_name = score, name
        mapping[tid] = best_name
    return mapping


def get_doc_topics(model: LatentDirichletAllocation, dtm) -> np.ndarray:
    """Return an (N, n_topics) array of per-document topic distributions."""
    return model.transform(dtm)


def run_topic_modeling(df: pd.DataFrame):
    """
    Fit LDA and return:
      model, vectorizer, dtm, topic_matrix (N×K), topic_name_map, df+dominant_topic
    """
    vectorizer, dtm = build_dtm(df["tokens"])
    model          = train_lda(dtm)
    feature_names  = vectorizer.get_feature_names_out().tolist()
    topic_name_map = match_topic_names(model, feature_names)
    topic_matrix   = get_doc_topics(model, dtm)

    dominant = topic_matrix.argmax(axis=1)
    df = df.copy()
    df["dominant_topic"] = [topic_name_map[t] for t in dominant]
    df["topic_vector"]   = list(topic_matrix)

    return model, vectorizer, dtm, topic_matrix, topic_name_map, df
