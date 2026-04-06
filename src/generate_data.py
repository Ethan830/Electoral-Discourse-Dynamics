"""
Synthetic Reddit data generator for 2024 U.S. election analysis.
Produces realistic post-level data for r/conservative, r/liberal,
r/politics, and r/neutralpolitics across pre- and post-election windows.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Election day: November 5, 2024
# Pre-election window:  Aug 7, 2024 – Nov 4, 2024  (90 days)
# Post-election window: Nov 6, 2024 – Feb 3, 2025  (90 days)

ELECTION_DATE = datetime(2024, 11, 5)
PRE_START     = datetime(2024, 8,  7)
PRE_END       = datetime(2024, 11, 4)
POST_START    = datetime(2024, 11, 6)
POST_END      = datetime(2025, 2,  3)

SUBREDDITS = ["conservative", "liberal", "politics", "neutralpolitics"]

# ── Template sentences ────────────────────────────────────────────────────────
TEMPLATES = {
    # (subreddit, period): list of (text, base_vader_shift)
    # base_vader_shift nudges the synthetic VADER score distribution

    ("conservative", "pre"): [
        "We need to secure the border now, this is out of control.",
        "The economy under Biden has been a complete disaster for working families.",
        "Trump is the only candidate who will put America first.",
        "Tired of watching inflation destroy the middle class every single day.",
        "The silent majority is coming out in November, mark my words.",
        "Democrats have failed on crime, the border, and the economy.",
        "We cannot afford four more years of these failed leftist policies.",
        "America needs strong leadership, not weakness and appeasement.",
        "The polls are wrong again – conservatives are going to show up huge.",
        "Our values and freedoms are on the line this election.",
        "The mainstream media is hiding the real story about Biden's failures.",
        "Standing up for the Second Amendment and constitutional rights.",
        "Border security is national security – we need the wall finished.",
        "This election is our last chance to save the America we love.",
        "Energy independence must be restored, Biden destroyed our oil industry.",
    ],
    ("conservative", "post"): [
        "Trump won! America is back – God bless the USA!",
        "The silent majority finally spoke and the results are incredible.",
        "This is a massive mandate for America First policies.",
        "Finally, we can secure the border and restore law and order.",
        "The American people rejected the radical left agenda completely.",
        "History will remember this as the greatest political comeback ever.",
        "We won decisively – now it is time to deliver for the people.",
        "The economy will turn around quickly under Trump's leadership.",
        "So grateful to see constitutional conservatism win at the ballot box.",
        "The deep state is finished – accountability is coming.",
        "Excited for the next four years of America First leadership.",
        "The left's narrative completely collapsed last night.",
        "This victory belongs to everyday Americans who love this country.",
        "Energy independence, strong borders, low taxes – here we come.",
        "What a night – proud to be American and proud to be conservative.",
    ],
    ("liberal", "pre"): [
        "We must protect democracy from MAGA extremism this November.",
        "Vote as if your rights depend on it – because they do.",
        "Climate change is an existential threat and we need real action now.",
        "Reproductive rights are on the ballot – we cannot afford to lose.",
        "Worried about what a second Trump term would mean for our country.",
        "Every vote matters – please make sure you are registered to vote.",
        "The stakes have never been higher for democracy in America.",
        "We need to expand healthcare access, not roll it back.",
        "Standing up for LGBTQ+ rights and against discrimination.",
        "Gun violence is a public health crisis that demands action.",
        "Donate and volunteer – we need all hands on deck this election.",
        "History will judge us by how we respond to this authoritarian threat.",
        "Education and science must guide our policy decisions.",
        "The choice is between democracy and fascism – vote accordingly.",
        "Income inequality is destroying the middle class and we must act.",
    ],
    ("liberal", "post"): [
        "I cannot believe half the country voted for this, I am devastated.",
        "This is a dark day for democracy and human rights in America.",
        "Extremely worried about the future of abortion rights and civil liberties.",
        "The misinformation campaign worked – millions were deceived.",
        "We need to fight back, organize, and resist every harmful policy.",
        "How did we let this happen? We need serious reflection as a party.",
        "Afraid for immigrant communities and marginalized groups right now.",
        "The climate crisis will only get worse with this administration.",
        "Mourning for the America I believed in – this is not who we are.",
        "We must protect our institutions and hold the line on democratic norms.",
        "Heartbroken but not giving up – the fight continues from here.",
        "This is a wake-up call – we failed to communicate our message.",
        "Checking in on friends who are scared right now – solidarity.",
        "We will not go quietly – organize, resist, protect each other.",
        "The work of protecting rights just got much harder but we press on.",
    ],
    ("politics", "pre"): [
        "New polls show an extremely tight race in Pennsylvania and Michigan.",
        "Economic concerns dominate voter priorities heading into November.",
        "Both campaigns are ramping up spending in battleground states.",
        "Record early voting numbers are being reported across the country.",
        "Immigration has become the defining issue of the 2024 election cycle.",
        "Analysis: what the latest swing-state polling tells us.",
        "How inflation data could influence undecided voters in the final weeks.",
        "Campaign trail update: last-minute rallies in Wisconsin and Arizona.",
        "Voter turnout models predict high participation across all demographics.",
        "Fact-checking the final stretch of campaign advertisements.",
        "The electoral college math: where the election will be decided.",
        "Debate recap: who came out ahead on key policy issues.",
        "Third-party candidates could play spoiler in several key states.",
        "Early voting data analysis – what it means for election night.",
        "GOTV efforts intensifying in suburban districts that will decide the race.",
    ],
    ("politics", "post"): [
        "Trump wins 2024 presidential election in decisive victory.",
        "Exit polls: economy and immigration were the top issues for voters.",
        "What the election results mean for US policy over the next four years.",
        "Democratic party faces internal reckoning after significant losses.",
        "World leaders react to Trump's return to the White House.",
        "Senate and House results: what divided or unified government means.",
        "Analysis: how Latino and suburban voters shifted from 2020.",
        "Post-election: what comes next for the Biden administration.",
        "Republicans celebrate gains across several key battleground states.",
        "What drove the swing toward Republicans among working-class voters.",
        "Stock markets react to election results with volatile trading.",
        "Looking ahead: key policy battles expected in the new Congress.",
        "Transition begins as Trump team names early cabinet picks.",
        "The polling miss: why surveys underestimated Republican turnout again.",
        "State-level results: red wave materializes in several swing states.",
    ],
    ("neutralpolitics", "pre"): [
        "Methodological analysis of current polling aggregator accuracy.",
        "Historical comparison of economic indicators and incumbent performance.",
        "Comparative policy analysis: immigration proposals from both parties.",
        "Examining the evidence base for proposed healthcare reforms.",
        "Historical voter turnout patterns in midterms versus presidential years.",
        "How media framing affects voter perception of economic conditions.",
        "Academic research review: does early voting change election outcomes?",
        "Evaluating the fiscal projections of competing tax policy proposals.",
        "What political science tells us about persuadable voters in 2024.",
        "Structural factors that predict presidential election outcomes.",
        "Examining polarization metrics and their trend over the past decade.",
        "Analysis of campaign finance data and its correlation with outcomes.",
        "Electoral college reform: arguments for and against the current system.",
        "Assessing the methodological quality of recent swing-state surveys.",
        "Evidence-based review of border policy effectiveness studies.",
    ],
    ("neutralpolitics", "post"): [
        "Post-election analysis: which forecasting models performed best.",
        "Examining the demographic coalitions that shaped the 2024 result.",
        "Academic literature on how election outcomes affect political trust.",
        "Policy continuity and change: what historical transitions predict.",
        "Evaluating competing explanations for the polling miss in 2024.",
        "Structural versus candidate-specific factors in the election outcome.",
        "Institutional resilience: evidence from prior democratic transitions.",
        "International comparison: how other democracies handled similar shifts.",
        "What the election results tell us about partisan sorting trends.",
        "Analysis of issue salience and how it shifted over the campaign.",
        "Electoral geography: county-level shifts and what they mean.",
        "Media consumption patterns and their correlation with voting behavior.",
        "Longitudinal study of political polarization measures post-election.",
        "Evaluating the robustness of exit poll methodology.",
        "Economic forecasting and policy scenario analysis for the new term.",
    ],
}

# Sentiment distribution params (mean, std) for VADER compound score
# These mirror realistic patterns: Trump won, so conservatives more positive post-election
SENTIMENT_PARAMS = {
    ("conservative", "pre"):       (-0.05, 0.30),
    ("conservative", "post"):      ( 0.28, 0.28),
    ("liberal",      "pre"):       ( 0.05, 0.30),
    ("liberal",      "post"):      (-0.22, 0.30),
    ("politics",     "pre"):       (-0.08, 0.32),
    ("politics",     "post"):      (-0.14, 0.30),
    ("neutralpolitics", "pre"):    ( 0.02, 0.20),
    ("neutralpolitics", "post"):   ( 0.01, 0.20),
}

# Topic weights per (subreddit, period)  –  6 topics:
# 0=immigration, 1=economy, 2=democracy_rights, 3=climate, 4=healthcare, 5=election_integrity
TOPIC_WEIGHTS = {
    ("conservative",    "pre"):  [0.30, 0.28, 0.10, 0.05, 0.08, 0.19],
    ("conservative",    "post"): [0.35, 0.25, 0.08, 0.03, 0.06, 0.23],
    ("liberal",         "pre"):  [0.10, 0.15, 0.28, 0.18, 0.22, 0.07],
    ("liberal",         "post"): [0.08, 0.12, 0.35, 0.15, 0.20, 0.10],
    ("politics",        "pre"):  [0.20, 0.22, 0.18, 0.10, 0.12, 0.18],
    ("politics",        "post"): [0.22, 0.25, 0.18, 0.08, 0.10, 0.17],
    ("neutralpolitics", "pre"):  [0.18, 0.20, 0.15, 0.12, 0.14, 0.21],
    ("neutralpolitics", "post"): [0.17, 0.22, 0.16, 0.10, 0.13, 0.22],
}

TOPIC_NAMES = [
    "Immigration & Border",
    "Economy & Inflation",
    "Democracy & Rights",
    "Climate & Environment",
    "Healthcare & Abortion",
    "Election Integrity",
]


def random_timestamp(start: datetime, end: datetime, rng: np.random.Generator) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=float(rng.uniform(0, delta)))


def generate_dataset(n_per_group: int = 350, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    post_id = 0

    for sub in SUBREDDITS:
        for period, (start, end) in [("pre", (PRE_START, PRE_END)),
                                      ("post", (POST_START, POST_END))]:
            key = (sub, period)
            templates = TEMPLATES[key]
            mu, sigma = SENTIMENT_PARAMS[key]
            topic_w = TOPIC_WEIGHTS[key]

            for _ in range(n_per_group):
                text = templates[int(rng.integers(0, len(templates)))]
                # Add minor lexical variation
                if rng.random() < 0.3:
                    text = text + " " + rng.choice([
                        "Thoughts?", "Discussion?", "Source in comments.",
                        "Share widely.", "This is important.",
                        "Please read carefully.", "What do you think?",
                    ])

                # Simulate engagement: score and num_comments
                # Pre-election generally higher engagement
                base_score = 500 if period == "pre" else 420
                if sub == "conservative" and period == "post":
                    base_score = 620
                if sub == "liberal" and period == "post":
                    base_score = 280
                score = max(1, int(rng.normal(base_score, base_score * 0.6)))
                n_comments = max(0, int(rng.normal(score * 0.35, score * 0.20)))

                # Assign dominant topic
                topic_idx = int(rng.choice(len(TOPIC_NAMES), p=topic_w))

                rows.append({
                    "post_id":   post_id,
                    "subreddit": sub,
                    "period":    period,
                    "timestamp": random_timestamp(start, end, rng),
                    "text":      text,
                    "score":     score,
                    "num_comments": n_comments,
                    "topic_true": topic_idx,
                })
                post_id += 1

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = "data/reddit_posts.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} posts -> {out}")
    print(df.groupby(["subreddit", "period"]).size())
