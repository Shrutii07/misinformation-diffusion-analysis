"""
05_visualization.py
Generate all figures for the misinformation diffusion project.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

FAKE_COLOR = '#e74c3c'
REAL_COLOR = '#2ecc71'


def plot_cascade_size_distribution(cascade_df):
    """Plot cascade size distribution for fake vs real."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, label, color, title in [
        (axes[0], 'fake', FAKE_COLOR, 'Fake News'),
        (axes[1], 'real', REAL_COLOR, 'Real News'),
    ]:
        data = cascade_df[cascade_df.label == label]['cascade_size']
        data = data[data > 0]
        ax.hist(np.log10(data), bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.set_xlabel('log10(Cascade Size)')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Cascade Size Distribution')
        ax.axvline(np.log10(data.median()), color='black', linestyle='--',
                   label=f'Median: {data.median():.0f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cascade_size_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved cascade_size_distribution.png")


def plot_cascade_velocity(cascade_df):
    """Compare cascade velocity between fake and real."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        data = cascade_df[(cascade_df.label == label) & (cascade_df.velocity_per_hour > 0)]
        data = data['velocity_per_hour']
        ax.hist(np.log10(data.clip(lower=0.01)), bins=50, color=color, alpha=0.6,
                label=f'{label.title()} (median: {data.median():.2f}/hr)', edgecolor='white')

    ax.set_xlabel('log10(Tweets per Hour)')
    ax.set_ylabel('Count')
    ax.set_title('Cascade Velocity: Fake vs Real News')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cascade_velocity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved cascade_velocity.png")


def plot_cascade_duration(cascade_df):
    """Compare cascade duration between fake and real."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        data = cascade_df[(cascade_df.label == label) & (cascade_df.duration_hours > 0)]
        data = np.log10(data['duration_hours'].clip(lower=0.01))
        ax.hist(data, bins=50, color=color, alpha=0.6, label=label.title(), edgecolor='white')

    ax.set_xlabel('log10(Duration in Hours)')
    ax.set_ylabel('Count')
    ax.set_title('Cascade Duration: Fake vs Real News')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cascade_duration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved cascade_duration.png")


def plot_sentiment_distribution(sentiment_df):
    """Plot sentiment distributions for fake vs real."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compound sentiment
    ax = axes[0]
    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        data = sentiment_df[sentiment_df.label == label]['sentiment_compound']
        ax.hist(data, bins=50, color=color, alpha=0.6, label=label.title(), edgecolor='white')
    ax.set_xlabel('VADER Compound Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    ax.legend()

    # Absolute sentiment (emotional intensity)
    ax = axes[1]
    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        data = sentiment_df[sentiment_df.label == label]['sentiment_abs']
        ax.hist(data, bins=50, color=color, alpha=0.6, label=label.title(), edgecolor='white')
    ax.set_xlabel('Absolute Sentiment (Emotional Intensity)')
    ax.set_ylabel('Count')
    ax.set_title('Emotional Intensity Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sentiment_distribution.png")


def plot_sentiment_vs_cascade(sentiment_df):
    """Scatter plot of sentiment vs cascade size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = sentiment_df[sentiment_df.cascade_size > 0].copy()
    df['log_cascade'] = np.log10(df['cascade_size'])

    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        sub = df[df.label == label]
        ax.scatter(sub['sentiment_abs'], sub['log_cascade'],
                  color=color, alpha=0.15, s=10, label=label.title())

    ax.set_xlabel('Absolute Sentiment Score')
    ax.set_ylabel('log10(Cascade Size)')
    ax.set_title('Emotional Intensity vs Cascade Size')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_vs_cascade.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sentiment_vs_cascade.png")


def plot_sentiment_class_bar(sentiment_df):
    """Bar chart of sentiment class proportions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = ['negative', 'neutral', 'positive']
    x = np.arange(len(classes))
    width = 0.35

    for i, (label, color) in enumerate([('fake', FAKE_COLOR), ('real', REAL_COLOR)]):
        sub = sentiment_df[sentiment_df.label == label]
        counts = sub['sentiment_class'].value_counts(normalize=True)
        vals = [counts.get(c, 0) for c in classes]
        ax.bar(x + i * width, vals, width, label=label.title(), color=color, edgecolor='white')

    ax.set_xlabel('Sentiment Class')
    ax.set_ylabel('Proportion')
    ax.set_title('Sentiment Class Distribution: Fake vs Real')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_class_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sentiment_class_bar.png")


def plot_network_graph(gexf_path, name, max_nodes=500):
    """Visualize a network from GEXF file."""
    G = nx.read_gexf(gexf_path)

    # Take largest connected component if too big
    if G.number_of_nodes() > max_nodes:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        nodes = list(components[0])[:max_nodes]
        G = G.subgraph(nodes).copy()

    if G.number_of_nodes() == 0:
        print(f"  Skipping {name} network plot (empty)")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by label
    colors = []
    for n in G.nodes():
        label = G.nodes[n].get('label', 'unknown')
        if label == 'fake':
            colors.append(FAKE_COLOR)
        elif label == 'real':
            colors.append(REAL_COLOR)
        else:
            # For domain network, use fake_ratio
            fr = float(G.nodes[n].get('fake_ratio', 0.5))
            colors.append(FAKE_COLOR if fr > 0.5 else REAL_COLOR)

    # Node sizes by degree
    degrees = dict(G.degree())
    sizes = [max(20, min(300, degrees[n] * 5)) for n in G.nodes()]

    pos = nx.spring_layout(G, k=1.5/np.sqrt(max(G.number_of_nodes(), 1)),
                           iterations=50, seed=42)
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.7, ax=ax)

    fake_patch = mpatches.Patch(color=FAKE_COLOR, label='Fake')
    real_patch = mpatches.Patch(color=REAL_COLOR, label='Real')
    ax.legend(handles=[fake_patch, real_patch], loc='upper left')
    ax.set_title(f'{name} Network (n={G.number_of_nodes()}, e={G.number_of_edges()})')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{name.lower().replace(" ", "_")}_network.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name.lower().replace(' ', '_')}_network.png")


def plot_degree_distribution(csv_path, name):
    """Plot degree distribution from saved CSV."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        sub = df[df.label == label]
        degrees = sub['degree']
        if len(degrees) == 0:
            continue
        # Log-log degree distribution
        deg_counts = Counter(degrees)
        degs = sorted(deg_counts.keys())
        counts = [deg_counts[d] for d in degs]
        ax.scatter(degs, counts, color=color, alpha=0.6, s=20, label=label.title())

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree (log)')
    ax.set_ylabel('Count (log)')
    ax.set_title(f'{name} Degree Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{name.lower().replace(" ", "_")}_degree_dist.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name.lower().replace(' ', '_')}_degree_dist.png")


def plot_diffusion_curves(diffusion_df):
    """Plot fitted logistic growth rates comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color in [('fake', FAKE_COLOR), ('real', REAL_COLOR)]:
        sub = diffusion_df[diffusion_df.label == label]
        if len(sub) == 0:
            continue
        ax.hist(sub['growth_rate'].clip(upper=sub['growth_rate'].quantile(0.95)),
                bins=30, color=color, alpha=0.6, label=label.title(), edgecolor='white')

    ax.set_xlabel('Logistic Growth Rate (k)')
    ax.set_ylabel('Count')
    ax.set_title('Diffusion Growth Rate: Fake vs Real News')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'diffusion_growth_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved diffusion_growth_rate.png")


def plot_summary_dashboard(cascade_df, sentiment_df):
    """Create a summary dashboard with key findings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cascade size boxplot
    ax = axes[0, 0]
    fake_sizes = cascade_df[cascade_df.label == 'fake']['cascade_size']
    real_sizes = cascade_df[cascade_df.label == 'real']['cascade_size']
    bp = ax.boxplot([np.log10(fake_sizes.clip(lower=1)), np.log10(real_sizes.clip(lower=1))],
                    labels=['Fake', 'Real'], patch_artist=True)
    bp['boxes'][0].set_facecolor(FAKE_COLOR)
    bp['boxes'][1].set_facecolor(REAL_COLOR)
    ax.set_ylabel('log10(Cascade Size)')
    ax.set_title('Cascade Size Comparison')

    # 2. Sentiment boxplot
    ax = axes[0, 1]
    fake_sent = sentiment_df[sentiment_df.label == 'fake']['sentiment_compound']
    real_sent = sentiment_df[sentiment_df.label == 'real']['sentiment_compound']
    bp = ax.boxplot([fake_sent, real_sent], labels=['Fake', 'Real'], patch_artist=True)
    bp['boxes'][0].set_facecolor(FAKE_COLOR)
    bp['boxes'][1].set_facecolor(REAL_COLOR)
    ax.set_ylabel('VADER Compound Score')
    ax.set_title('Sentiment Comparison')

    # 3. First hour tweets
    ax = axes[1, 0]
    if 'first_hour_count' in cascade_df.columns:
        fake_fh = cascade_df[cascade_df.label == 'fake']['first_hour_count']
        real_fh = cascade_df[cascade_df.label == 'real']['first_hour_count']
        bp = ax.boxplot([np.log10(fake_fh.clip(lower=1)), np.log10(real_fh.clip(lower=1))],
                        labels=['Fake', 'Real'], patch_artist=True)
        bp['boxes'][0].set_facecolor(FAKE_COLOR)
        bp['boxes'][1].set_facecolor(REAL_COLOR)
        ax.set_ylabel('log10(First Hour Tweets)')
        ax.set_title('Early Cascade Activity')

    # 4. Emotional intensity
    ax = axes[1, 1]
    fake_abs = sentiment_df[sentiment_df.label == 'fake']['sentiment_abs']
    real_abs = sentiment_df[sentiment_df.label == 'real']['sentiment_abs']
    bp = ax.boxplot([fake_abs, real_abs], labels=['Fake', 'Real'], patch_artist=True)
    bp['boxes'][0].set_facecolor(FAKE_COLOR)
    bp['boxes'][1].set_facecolor(REAL_COLOR)
    ax.set_ylabel('Absolute Sentiment')
    ax.set_title('Emotional Intensity Comparison')

    plt.suptitle('Misinformation Diffusion: Key Findings Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved summary_dashboard.png")


def main():
    print("Generating visualizations...")

    # Load data
    cascade_df = pd.read_csv(os.path.join(RESULTS_DIR, 'cascade_features.csv'))
    sentiment_df = pd.read_csv(os.path.join(RESULTS_DIR, 'sentiment_results.csv'))

    # Cascade plots
    plot_cascade_size_distribution(cascade_df)
    plot_cascade_velocity(cascade_df)
    plot_cascade_duration(cascade_df)

    # Sentiment plots
    plot_sentiment_distribution(sentiment_df)
    plot_sentiment_vs_cascade(sentiment_df)
    plot_sentiment_class_bar(sentiment_df)

    # Network plots
    article_gexf = os.path.join(RESULTS_DIR, 'article_network.gexf')
    domain_gexf = os.path.join(RESULTS_DIR, 'domain_network.gexf')
    if os.path.exists(article_gexf):
        plot_network_graph(article_gexf, 'Article Co-sharing')
    if os.path.exists(domain_gexf):
        plot_network_graph(domain_gexf, 'Domain')

    # Degree distributions
    article_deg = os.path.join(RESULTS_DIR, 'article_cosharing_degree_dist.csv')
    if os.path.exists(article_deg):
        plot_degree_distribution(article_deg, 'Article Co-sharing')

    # Diffusion curves
    diffusion_path = os.path.join(RESULTS_DIR, 'diffusion_curves.csv')
    if os.path.exists(diffusion_path):
        diffusion_df = pd.read_csv(diffusion_path)
        plot_diffusion_curves(diffusion_df)

    # Summary dashboard
    plot_summary_dashboard(cascade_df, sentiment_df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
