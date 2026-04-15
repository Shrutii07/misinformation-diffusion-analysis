"""
02_network_analysis.py
Build co-sharing networks from FakeNewsNet data and compute network metrics.
- Article co-sharing network: articles connected if shared by overlapping tweet audiences
- Domain-level network: domains connected if they share articles on same topics
- Compute degree distribution, clustering, centrality, small-world properties
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load the raw CSVs and parse tweet ID lists."""
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

    def parse_tids(x):
        if pd.isna(x) or str(x).strip() == '':
            return set()
        return set(t.strip() for t in str(x).split('\t') if t.strip())

    df['tweet_set'] = df['tweet_ids'].apply(parse_tids)
    df['cascade_size'] = df['tweet_set'].apply(len)
    return df


def extract_domain(url):
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url if url.startswith('http') else f'http://{url}')
        return parsed.netloc.replace('www.', '')
    except Exception:
        return 'unknown'


def build_article_cosharing_network(df, min_overlap=2, max_articles=2000):
    """
    Build a network where articles are nodes, connected if they share
    at least `min_overlap` tweet IDs (proxy for shared audience).
    Sample to max_articles for tractability.
    """
    # Sample articles with most tweets for a richer network
    sampled = df.nlargest(max_articles, 'cascade_size').copy()
    print(f"Building co-sharing network with {len(sampled)} articles...")

    articles = sampled[['id', 'label', 'source', 'cascade_size']].copy()
    tweet_sets = dict(zip(sampled['id'], sampled['tweet_set']))

    G = nx.Graph()
    for _, row in articles.iterrows():
        G.add_node(row['id'], label=row['label'], source=row['source'],
                   cascade_size=row['cascade_size'])

    ids = list(tweet_sets.keys())
    edge_count = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            overlap = len(tweet_sets[ids[i]] & tweet_sets[ids[j]])
            if overlap >= min_overlap:
                G.add_edge(ids[i], ids[j], weight=overlap)
                edge_count += 1

    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def build_domain_network(df, min_articles=5):
    """
    Build a domain-level network where domains are nodes, connected
    if they have articles shared by overlapping tweet audiences.
    """
    df['domain'] = df['news_url'].apply(extract_domain)

    # Aggregate tweet sets per domain
    domain_tweets = defaultdict(set)
    domain_labels = defaultdict(lambda: {'fake': 0, 'real': 0})
    for _, row in df.iterrows():
        d = row['domain']
        domain_tweets[d] |= row['tweet_set']
        domain_labels[d][row['label']] += 1

    # Filter to domains with enough articles
    domains = [d for d, ts in domain_tweets.items() if len(ts) >= min_articles]
    print(f"Building domain network with {len(domains)} domains...")

    G = nx.Graph()
    for d in domains:
        labels = domain_labels[d]
        total = labels['fake'] + labels['real']
        fake_ratio = labels['fake'] / total if total > 0 else 0
        G.add_node(d, fake_ratio=fake_ratio, total_articles=total,
                   tweet_count=len(domain_tweets[d]))

    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            overlap = len(domain_tweets[domains[i]] & domain_tweets[domains[j]])
            if overlap >= 2:
                G.add_edge(domains[i], domains[j], weight=overlap)

    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def compute_network_metrics(G, name):
    """Compute and print key network metrics."""
    print(f"\n=== {name} Network Metrics ===")

    # Basic stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    print(f"Nodes: {n_nodes}, Edges: {n_edges}, Density: {density:.6f}")

    if n_nodes == 0:
        return {}

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    print(f"Degree: mean={np.mean(degrees):.2f}, median={np.median(degrees):.0f}, "
          f"max={np.max(degrees)}, std={np.std(degrees):.2f}")

    # Connected components
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    largest_cc = max(components, key=len)
    print(f"Largest component: {len(largest_cc)} nodes ({100*len(largest_cc)/n_nodes:.1f}%)")

    # Clustering coefficient
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")

    # Small-world check on largest component
    G_lcc = G.subgraph(largest_cc).copy()
    if len(G_lcc) > 1:
        avg_path = nx.average_shortest_path_length(G_lcc) if len(G_lcc) < 5000 else "too large"
        print(f"Avg shortest path (LCC): {avg_path}")

    # Centrality (top 10)
    if n_nodes < 5000:
        betweenness = nx.betweenness_centrality(G)
        top_betweenness = sorted(betweenness.items(), key=lambda x: -x[1])[:10]
        print(f"Top 10 betweenness centrality:")
        for node, val in top_betweenness:
            label = G.nodes[node].get('label', G.nodes[node].get('fake_ratio', ''))
            print(f"  {node}: {val:.4f} (label={label})")

    metrics = {
        'nodes': n_nodes, 'edges': n_edges, 'density': density,
        'avg_degree': np.mean(degrees), 'max_degree': int(np.max(degrees)),
        'n_components': len(components),
        'largest_component_pct': 100 * len(largest_cc) / n_nodes,
        'avg_clustering': avg_clustering,
    }
    return metrics


def compare_fake_real_subgraphs(G):
    """Compare network properties of fake vs real article subgraphs."""
    fake_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'fake']
    real_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'real']

    print(f"\n=== Fake vs Real Subgraph Comparison ===")
    results = {}
    for label, nodes in [('fake', fake_nodes), ('real', real_nodes)]:
        sub = G.subgraph(nodes)
        degrees = [d for _, d in sub.degree()]
        if len(degrees) == 0:
            continue
        clustering = nx.average_clustering(sub)
        results[label] = {
            'nodes': len(nodes),
            'edges': sub.number_of_edges(),
            'avg_degree': np.mean(degrees),
            'max_degree': int(np.max(degrees)),
            'avg_clustering': clustering,
            'density': nx.density(sub),
        }
        print(f"\n  {label.upper()} subgraph:")
        for k, v in results[label].items():
            print(f"    {k}: {v}")

    # Cross-edges (fake-real connections)
    cross_edges = sum(1 for u, v in G.edges()
                      if G.nodes[u].get('label') != G.nodes[v].get('label'))
    print(f"\n  Cross-edges (fake-real): {cross_edges}")
    results['cross_edges'] = cross_edges
    return results


def save_degree_distribution(G, name):
    """Save degree distribution data for plotting."""
    degrees = [d for _, d in G.degree()]
    labels = [G.nodes[n].get('label', 'unknown') for n in G.nodes()]
    deg_df = pd.DataFrame({'node': list(G.nodes()), 'degree': degrees, 'label': labels})
    deg_df.to_csv(os.path.join(RESULTS_DIR, f'{name}_degree_dist.csv'), index=False)
    return deg_df


def main():
    df = load_data()

    # Build and analyze article co-sharing network
    G_article = build_article_cosharing_network(df, min_overlap=2, max_articles=2000)
    article_metrics = compute_network_metrics(G_article, "Article Co-sharing")
    comparison = compare_fake_real_subgraphs(G_article)
    save_degree_distribution(G_article, 'article_cosharing')

    # Build and analyze domain network
    G_domain = build_domain_network(df, min_articles=5)
    domain_metrics = compute_network_metrics(G_domain, "Domain")
    save_degree_distribution(G_domain, 'domain')

    # Save networks for visualization
    nx.write_gexf(G_article, os.path.join(RESULTS_DIR, 'article_network.gexf'))
    nx.write_gexf(G_domain, os.path.join(RESULTS_DIR, 'domain_network.gexf'))

    # Save metrics summary
    all_metrics = {
        'article_network': article_metrics,
        'domain_network': domain_metrics,
        'fake_vs_real': comparison,
    }
    with open(os.path.join(RESULTS_DIR, 'network_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print("\nSaved: article_network.gexf, domain_network.gexf, network_metrics.json")


if __name__ == '__main__':
    main()
