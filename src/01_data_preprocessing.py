"""
01_data_preprocessing.py
Load FakeNewsNet CSVs, parse tweet IDs to extract timestamps,
compute per-article features, and save a unified dataset.
"""

import pandas as pd
from datetime import datetime, timezone
from urllib.parse import urlparse
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Twitter snowflake epoch: Nov 4, 2010 01:42:54.657 UTC
TWITTER_EPOCH = 1288834974657


def tweet_id_to_timestamp(tweet_id):
    """Extract UTC timestamp from a Twitter snowflake ID."""
    try:
        tid = int(tweet_id)
        ms = (tid >> 22) + TWITTER_EPOCH
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def parse_tweet_ids(tweet_ids_str):
    """Parse tab-separated tweet IDs string into a list of ints."""
    if pd.isna(tweet_ids_str) or str(tweet_ids_str).strip() == '':
        return []
    return [t.strip() for t in str(tweet_ids_str).split('\t') if t.strip()]


def extract_domain(url):
    """Extract domain from a news URL."""
    try:
        parsed = urlparse(url if url.startswith('http') else f'http://{url}')
        domain = parsed.netloc.replace('www.', '')
        return domain
    except Exception:
        return 'unknown'


def load_and_process():
    files = {
        'politifact_fake': ('politifact', 'fake'),
        'politifact_real': ('politifact', 'real'),
        'gossipcop_fake': ('gossipcop', 'fake'),
        'gossipcop_real': ('gossipcop', 'real'),
    }

    all_rows = []
    for fname, (source, label) in files.items():
        path = os.path.join(DATA_DIR, f'{fname}.csv')
        df = pd.read_csv(path)
        df['source'] = source
        df['label'] = label
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Total articles: {len(df)}")
    print(f"  PolitiFact: {len(df[df.source == 'politifact'])} (fake: {len(df[(df.source=='politifact') & (df.label=='fake')])}, real: {len(df[(df.source=='politifact') & (df.label=='real')])})")
    print(f"  GossipCop:  {len(df[df.source == 'gossipcop'])} (fake: {len(df[(df.source=='gossipcop') & (df.label=='fake')])}, real: {len(df[(df.source=='gossipcop') & (df.label=='real')])})")

    # Parse tweet IDs
    df['tweet_id_list'] = df['tweet_ids'].apply(parse_tweet_ids)
    df['cascade_size'] = df['tweet_id_list'].apply(len)

    # Extract timestamps from first and last tweet IDs (cascade timing)
    def get_cascade_times(tid_list):
        timestamps = []
        for tid in tid_list:
            ts = tweet_id_to_timestamp(tid)
            if ts is not None:
                timestamps.append(ts)
        if not timestamps:
            return None, None, None
        timestamps.sort()
        first = timestamps[0]
        last = timestamps[-1]
        duration_hours = (last - first).total_seconds() / 3600
        return first, last, duration_hours

    cascade_data = df['tweet_id_list'].apply(get_cascade_times)
    df['cascade_start'] = cascade_data.apply(lambda x: x[0])
    df['cascade_end'] = cascade_data.apply(lambda x: x[1])
    df['cascade_duration_hours'] = cascade_data.apply(lambda x: x[2])

    # Extract source domain
    df['domain'] = df['news_url'].apply(extract_domain)

    # Title features
    df['title_length'] = df['title'].fillna('').apply(len)
    df['title_word_count'] = df['title'].fillna('').apply(lambda x: len(x.split()))

    # Summary stats
    print(f"\nCascade size stats:")
    for label in ['fake', 'real']:
        sub = df[df.label == label]
        print(f"  {label}: mean={sub['cascade_size'].mean():.1f}, median={sub['cascade_size'].median():.0f}, max={sub['cascade_size'].max()}")

    print(f"\nCascade duration (hours) stats:")
    for label in ['fake', 'real']:
        sub = df[df.label == label].dropna(subset=['cascade_duration_hours'])
        print(f"  {label}: mean={sub['cascade_duration_hours'].mean():.1f}, median={sub['cascade_duration_hours'].median():.1f}")

    # Save processed dataset
    out_cols = ['id', 'news_url', 'title', 'source', 'label', 'domain',
                'cascade_size', 'cascade_start', 'cascade_end',
                'cascade_duration_hours', 'title_length', 'title_word_count']
    df[out_cols].to_csv(os.path.join(RESULTS_DIR, 'processed_articles.csv'), index=False)
    print(f"\nSaved processed_articles.csv ({len(df)} rows)")

    # Save tweet-level data for network construction
    tweet_rows = []
    for _, row in df.iterrows():
        for tid in row['tweet_id_list']:
            ts = tweet_id_to_timestamp(tid)
            tweet_rows.append({
                'article_id': row['id'],
                'tweet_id': tid,
                'timestamp': ts,
                'label': row['label'],
                'source': row['source'],
            })
    tweet_df = pd.DataFrame(tweet_rows)
    tweet_df.to_csv(os.path.join(RESULTS_DIR, 'tweet_events.csv'), index=False)
    print(f"Saved tweet_events.csv ({len(tweet_df)} rows)")

    return df


if __name__ == '__main__':
    load_and_process()
