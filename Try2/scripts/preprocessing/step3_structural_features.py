import networkx as nx
import pandas as pd

print("Loading Higgs-5000 graph...")

# ==================================================
# Paths (Try2 Structure)
# ==================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "higgs_5000.edgelist"
OUTPUT_PATH = BASE_DIR / "data" / "structural_features.csv"


G = nx.read_edgelist(
    INPUT_PATH,
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

G_und = G.to_undirected()

nodes = sorted(G.nodes())


print("\nComputing In-Degree...")
in_degree = dict(G.in_degree())

print("Computing Out-Degree")
out_degree = dict(G.out_degree())

print("Computing Total Degree")
degree = dict(G.degree())

print("Computing Clustering Coefficient")
clustering = nx.clustering(G_und)


print("\nBuilding Raw Feature Matrix")

features = pd.DataFrame({
    "node": nodes,
    "in_degree": [in_degree[n] for n in nodes],
    "out_degree": [out_degree[n] for n in nodes],
    "degree": [degree[n] for n in nodes],
    "clustering": [clustering[n] for n in nodes],
})


print("\nRaw Feature Statistics:")

print(
    features[
        [
            "in_degree",
            "out_degree",
            "degree",
            "clustering"
        ]
    ].describe()
)


features.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nFeature Matrix Shape:")
print(features.shape)

print("\nFeature Columns:")
print(features.columns.tolist())

print("\nStep 3 completed")
print(" Features are intentionally left UNNORMALIZED.")
