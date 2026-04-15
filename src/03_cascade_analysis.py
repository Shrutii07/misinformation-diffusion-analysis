"""
03_cascade_analysis.py
Analyze temporal cascades of misinformation vs factual news.
- Extract timestamps from Twitter snowflake IDs
- Compare cascade size, duration, velocity between fake and real
- Fit SIR-style diffusion curves
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TWITTER_EPOCH = 1288834974657


def tweet_id_to_ms(tweet_id):
    """Convert tweet snowflake ID to Unix milliseconds."""
    try:
        return (int(tweet_id) >> 22) + TWITTER_EPOCH
    except (ValueError, OverflowError):
        return None


def load_data():
    files = {
        'politifact_fake': ('politifact', 'fake'),
        'politifact_real': ('politifact', 'real'),
        'gossipcop_fake': ('gossipcop', 'fake'),
        'gossipcop_real': ('gossipcop', 'real'),
    }
    all_rows = []
    for fname, (source, label) in files.items():
        df = pd.read_csv(os.path.join(DATA_DIR, f'{fname}.csv'))
        df['source'] = source
        df['label'] = label
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)


def build_cascade_features(df):
    """Build cascade-level features from tweet IDs."""
    records = []
    for _, row in df.iterrows():
        if pd.isna(row['tweet_ids']) or str(row['tweet_ids']).strip() == '':
            continue
        tids = [t.strip() for t in str(row['tweet_ids']).split('\t') if t.strip()]
        if len(tids) == 0:
            continue

        # Extract timestamps
        timestamps_ms = []
        for tid in tids:
            ms = tweet_id_to_ms(tid)
            if ms is not None and 1_000_000_000_000 < ms < 2_000_000_000_000:
                timestamps_ms.append(ms)

        if len(timestamps_ms) < 2:
            records.append({
                'article_id': row['id'],
                'label': row['label'],
                'source': row['source'],
                'title': row.get('title', ''),
                'cascade_size': len(tids),
                'valid_timestamps': len(timestamps_ms),
                'duration_hours': 0,
                'velocity_per_hour': 0,
                'first_hour_count': len(tids),
                'first_day_count': len(tids),
            })
            continue

        timestamps_ms.sort()
        t0 = timestamps_ms[0]
        duration_ms = timestamps_ms[-1] - t0
        duration_hours = duration_ms / (1000 * 3600)

        # Velocity: tweets per hour
        velocity = len(timestamps_ms) / max(duration_hours, 0.01)

        # Count in first hour and first day
        first_hour = sum(1 for t in timestamps_ms if (t - t0) <= 3600 * 1000)
        first_day = sum(1 for t in timestamps_ms if (t - t0) <= 86400 * 1000)

        # Time to 50% of cascade
        half_idx = len(timestamps_ms) // 2
        time_to_half_hours = (timestamps_ms[half_idx] - t0) / (1000 * 3600)

        records.append({
            'article_id': row['id'],
            'label': row['label'],
            'source': row['source'],
            'title': row.get('title', ''),
            'cascade_size': len(tids),
            'valid_timestamps': len(timestamps_ms),
            'duration_hours': duration_hours,
            'velocity_per_hour': velocity,
            'first_hour_count': first_hour,
            'first_day_count': first_day,
            'time_to_half_hours': time_to_half_hours,
        })

    return pd.DataFrame(records)


def logistic_growth(t, L, k, t0):
    """Logistic (S-curve) growth model: cumulative adoption."""
    return L / (1 + np.exp(-k * (t - t0)))


def fit_diffusion_curves(df_raw, n_sample=50):
    """Fit logistic diffusion curves to sampled cascades."""
    # Pick cascades with enough data points
    candidates = df_raw[df_raw['tweet_ids'].notna()].copy()

    results = {'fake': [], 'real': []}
    for label in ['fake', 'real']:
        sub = candidates[candidates['label'] == label]
        # Pick articles with many tweets
        sub = sub.copy()
        sub['n_tweets'] = sub['tweet_ids'].apply(
            lambda x: len(str(x).split('\t')) if pd.notna(x) else 0)
        sub = sub.nlargest(n_sample, 'n_tweets')

        for _, row in sub.iterrows():
            tids = [t.strip() for t in str(row['tweet_ids']).split('\t') if t.strip()]
            timestamps = sorted([tweet_id_to_ms(t) for t in tids
                                if tweet_id_to_ms(t) is not None])
            if len(timestamps) < 10:
                continue

            t0_ms = timestamps[0]
            t_hours = np.array([(t - t0_ms) / (1000 * 3600) for t in timestamps])
            cumulative = np.arange(1, len(t_hours) + 1, dtype=float)

            # Fit logistic curve
            try:
                popt, _ = curve_fit(logistic_growth, t_hours, cumulative,
                                    p0=[len(cumulative), 0.1, np.median(t_hours)],
                                    maxfev=5000)
                L, k, t0 = popt
                results[label].append({
                    'article_id': row['id'],
                    'cascade_size': len(tids),
                    'L': L, 'k': k, 't0_hours': t0,
                    'growth_rate': k,
                })
            except (RuntimeError, ValueError):
                continue

    return results


def compare_cascades(cascade_df):
    """Statistical comparison of fake vs real cascades."""
    print("\n=== Cascade Comparison: Fake vs Real ===")

    metrics = ['cascade_size', 'duration_hours', 'velocity_per_hour',
               'first_hour_count', 'first_day_count']

    comparison = {}
    for m in metrics:
        fake = cascade_df[cascade_df.label == 'fake'][m].dropna()
        real = cascade_df[cascade_df.label == 'real'][m].dropna()

        # Mann-Whitney U test (non-parametric)
        if len(fake) > 0 and len(real) > 0:
            stat, pval = stats.mannwhitneyu(fake, real, alternative='two-sided')
        else:
            stat, pval = 0, 1

        comparison[m] = {
            'fake_mean': fake.mean(),
            'fake_median': fake.median(),
            'real_mean': real.mean(),
            'real_median': real.median(),
            'mannwhitney_stat': stat,
            'p_value': pval,
            'significant': pval < 0.05,
        }
        sig = "*" if pval < 0.05 else ""
        print(f"\n  {m}:")
        print(f"    Fake: mean={fake.mean():.2f}, median={fake.median():.1f}")
        print(f"    Real: mean={real.mean():.2f}, median={real.median():.1f}")
        print(f"    p-value: {pval:.4e} {sig}")

    return comparison


def main():
    print("Loading data...")
    df = load_data()

    print("Building cascade features...")
    cascade_df = build_cascade_features(df)
    cascade_df.to_csv(os.path.join(RESULTS_DIR, 'cascade_features.csv'), index=False)
    print(f"  Cascades: {len(cascade_df)} ({len(cascade_df[cascade_df.label=='fake'])} fake, "
          f"{len(cascade_df[cascade_df.label=='real'])} real)")

    # Compare fake vs real
    comparison = compare_cascades(cascade_df)

    # Fit diffusion curves
    print("\nFitting logistic diffusion curves...")
    diffusion = fit_diffusion_curves(df, n_sample=50)
    print(f"  Fitted curves: fake={len(diffusion['fake'])}, real={len(diffusion['real'])}")

    if diffusion['fake'] and diffusion['real']:
        fake_k = [d['growth_rate'] for d in diffusion['fake']]
        real_k = [d['growth_rate'] for d in diffusion['real']]
        print(f"  Growth rate (k): fake mean={np.mean(fake_k):.4f}, real mean={np.mean(real_k):.4f}")

    # Save diffusion results
    diffusion_df = pd.DataFrame(diffusion['fake'] + diffusion['real'])
    if not diffusion_df.empty:
        diffusion_df['label'] = ['fake'] * len(diffusion['fake']) + ['real'] * len(diffusion['real'])
        diffusion_df.to_csv(os.path.join(RESULTS_DIR, 'diffusion_curves.csv'), index=False)

    # Save comparison
    with open(os.path.join(RESULTS_DIR, 'cascade_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print("\nSaved: cascade_features.csv, diffusion_curves.csv, cascade_comparison.json")


if __name__ == '__main__':
    main()
