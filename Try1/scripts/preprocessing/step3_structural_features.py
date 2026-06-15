import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("Loading Higgs-5000 graph...")

G = nx.read_edgelist(
    "data/higgs_5000.edgelist",
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

# Convert to undirected where necessary
G_und = G.to_undirected()

nodes = sorted(G.nodes())

print("\nComputing Degree...")
degree = dict(G.degree())

print("Computing Approximate Betweenness...")
betweenness = nx.betweenness_centrality(
    G_und,
    k=500,
    seed=42,
    normalized=True
)

print("Computing Closeness...")
closeness = nx.closeness_centrality(G_und)

print("Computing Eigenvector...")
eigenvector = nx.eigenvector_centrality(
    G_und,
    max_iter=1000
)

print("Computing PageRank...")
pagerank = nx.pagerank(
    G,
    alpha=0.85
)

print("Computing Clustering Coefficient...")
clustering = nx.clustering(G_und)

print("\nBuilding Feature Matrix...")

features = pd.DataFrame({
    "node": nodes,
    "degree": [degree[n] for n in nodes],
    "betweenness": [betweenness[n] for n in nodes],
    "closeness": [closeness[n] for n in nodes],
    "eigenvector": [eigenvector[n] for n in nodes],
    "pagerank": [pagerank[n] for n in nodes],
    "clustering": [clustering[n] for n in nodes],
})

print("\nRaw Feature Statistics:")
print(features.describe())

# Normalize
scaler = MinMaxScaler()

feature_cols = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
]

features[feature_cols] = scaler.fit_transform(
    features[feature_cols]
)

print("\nNormalized Feature Statistics:")
print(features[feature_cols].describe())

features.to_csv(
    "data/structural_features.csv",
    index=False
)

print("\nSaved:")
print("data/structural_features.csv")

print("\nFeature Matrix Shape:")
print(features.shape)