"""
04_sentiment_analysis.py
Sentiment analysis on article titles from FakeNewsNet.
- VADER sentiment scoring
- Compare sentiment distributions between fake and real news
- Correlate sentiment with cascade size
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import stats
import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def analyze_sentiment(df):
    """Run VADER sentiment on article titles."""
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        if not title or title == 'nan':
            sentiments.append({'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1})
            continue
        scores = analyzer.polarity_scores(title)
        sentiments.append(scores)

    sent_df = pd.DataFrame(sentiments)
    df = df.copy()
    df['sentiment_compound'] = sent_df['compound'].values
    df['sentiment_pos'] = sent_df['pos'].values
    df['sentiment_neg'] = sent_df['neg'].values
    df['sentiment_neu'] = sent_df['neu'].values

    # Absolute sentiment (emotional intensity regardless of direction)
    df['sentiment_abs'] = df['sentiment_compound'].abs()

    # Classify sentiment
    df['sentiment_class'] = df['sentiment_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

    return df


def compare_sentiment(df):
    """Compare sentiment between fake and real news."""
    print("\n=== Sentiment Comparison: Fake vs Real ===")

    metrics = ['sentiment_compound', 'sentiment_abs', 'sentiment_pos', 'sentiment_neg']
    comparison = {}

    for m in metrics:
        fake = df[df.label == 'fake'][m].dropna()
        real = df[df.label == 'real'][m].dropna()

        stat, pval = stats.mannwhitneyu(fake, real, alternative='two-sided')

        comparison[m] = {
            'fake_mean': fake.mean(),
            'real_mean': real.mean(),
            'fake_std': fake.std(),
            'real_std': real.std(),
            'p_value': pval,
            'significant': pval < 0.05,
        }
        sig = "*" if pval < 0.05 else ""
        print(f"\n  {m}:")
        print(f"    Fake: mean={fake.mean():.4f} (std={fake.std():.4f})")
        print(f"    Real: mean={real.mean():.4f} (std={real.std():.4f})")
        print(f"    p-value: {pval:.4e} {sig}")

    # Sentiment class distribution
    print("\n  Sentiment class distribution:")
    for label in ['fake', 'real']:
        sub = df[df.label == label]
        counts = sub['sentiment_class'].value_counts(normalize=True)
        print(f"    {label}: " + ", ".join(f"{k}={v:.1%}" for k, v in counts.items()))

    return comparison


def correlate_sentiment_cascade(df):
    """Correlate sentiment with cascade size."""
    print("\n=== Sentiment-Cascade Correlation ===")

    df_valid = df[df['cascade_size'] > 0].copy()
    df_valid['log_cascade'] = np.log1p(df_valid['cascade_size'])

    correlations = {}
    for label in ['fake', 'real', 'all']:
        sub = df_valid if label == 'all' else df_valid[df_valid.label == label]
        if len(sub) < 10:
            continue

        r_compound, p_compound = stats.spearmanr(sub['sentiment_abs'], sub['log_cascade'])
        r_neg, p_neg = stats.spearmanr(sub['sentiment_neg'], sub['log_cascade'])

        correlations[label] = {
            'abs_sentiment_vs_cascade': {'r': r_compound, 'p': p_compound},
            'neg_sentiment_vs_cascade': {'r': r_neg, 'p': p_neg},
            'n': len(sub),
        }
        print(f"\n  {label} (n={len(sub)}):")
        print(f"    |sentiment| vs log(cascade): r={r_compound:.4f}, p={p_compound:.4e}")
        print(f"    neg_sentiment vs log(cascade): r={r_neg:.4f}, p={p_neg:.4e}")

    return correlations


def main():
    # Load cascade features (has cascade_size) merged with processed articles
    cascade_path = os.path.join(RESULTS_DIR, 'cascade_features.csv')
    if os.path.exists(cascade_path):
        df = pd.read_csv(cascade_path)
    else:
        # Fallback: load processed articles
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'processed_articles.csv'))

    print(f"Analyzing sentiment for {len(df)} articles...")

    # Run sentiment analysis
    df = analyze_sentiment(df)

    # Compare fake vs real
    comparison = compare_sentiment(df)

    # Correlate with cascade size
    if 'cascade_size' in df.columns:
        correlations = correlate_sentiment_cascade(df)
    else:
        correlations = {}

    # Save results
    df.to_csv(os.path.join(RESULTS_DIR, 'sentiment_results.csv'), index=False)

    with open(os.path.join(RESULTS_DIR, 'sentiment_comparison.json'), 'w') as f:
        json.dump({'comparison': comparison, 'correlations': correlations},
                  f, indent=2, default=str)

    print(f"\nSaved: sentiment_results.csv, sentiment_comparison.json")


if __name__ == '__main__':
    main()
