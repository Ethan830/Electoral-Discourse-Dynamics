# Electoral Discourse Dynamics: Analyzing Sentiment Shifts and Topic Evolution

This repository contains the data pipeline and analytical framework for studying partisan Reddit communities before and after the 2024 U.S. Presidential Election. By analyzing subreddits across the ideological spectrum, this project quantifies how electoral outcomes drive emotional shifts and topical engagement in online spaces.

## Project Overview
Using the Reddit PRAW API, we collected and analyzed posts from four key communities: **r/conservative**, **r/liberal**, **r/politics**, and **r/neutralpolitics**. The study covers two symmetric 90-day windows anchored to Election Day (November 5, 2024).

### Key Findings
* **Asymmetric Sentiment:** Post-election, r/conservative showed a large positive sentiment shift ($\Delta=+0.306$), while r/liberal exhibited a significant negative shift ($\Delta=-0.091$).
* **Winner's Boost:** Winning-coalition communities sustained elevated engagement, while losing-coalition communities experienced a marked drop in post scores and comments.
* **Topic Persistence:** Despite emotional volatility, communities maintained core priorities (e.g., Immigration for conservatives; Rights and Healthcare for liberals).
* **Model Validation:** VADER outperformed TextBlob on political slang and social media text, achieving 76.7% accuracy.

## Methodology
The analysis utilizes a multi-stage NLP pipeline:
1. **Data Collection:** Targeted scraping of 2,800 posts via PRAW API.
2. **Preprocessing:** URL removal, stopword filtering, and WordNet lemmatization.
3. **Sentiment Analysis:** Dual-model approach using VADER (rule-based) and TextBlob (pattern-based).
4. **Topic Modeling:** Latent Dirichlet Allocation (LDA) with $K=6$ topics to track issue salience.
5. **Statistics:** Welch's t-tests for sentiment means and $\chi^2$ tests for topic distribution shifts.

## Repository Structure
* `src/collect_reddit.py`: Live Reddit API collection script.
* `src/analysis.py`: Preprocessing, VADER implementation, and LDA modeling.
* `data/`: Sample corpus and hand-labeled validation sets.
* `plots/`: Visualizations of sentiment trends, topic heatmaps, and engagement metrics.

## Limitations
* **Sarcasm:** Lexicon-based models like VADER still struggle with high-irony political discourse.
* **API Limits:** This repository includes synthetic data generated to match original distributions for reproducibility where API access is restricted.
* **Bot Filtering:** While basic score-based filters were used, coordinated inauthentic behavior remains a challenge in political datasets.
