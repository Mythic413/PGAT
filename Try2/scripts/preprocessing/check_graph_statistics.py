import gzip
import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

ORIGINAL_PATH = BASE_DIR / "data" / "higgs-social_network.edgelist.gz"
SAMPLED_PATH = BASE_DIR / "Try2" / "data" / "higgs_5000.edgelist"

print("=" * 60)
print("ORIGINAL HIGGS")
print("=" * 60)

G = nx.DiGraph()

with gzip.open(ORIGINAL_PATH, "rt") as f:
    for line in f:
        u, v = map(int, line.split())
        G.add_edge(u, v)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

avg_degree = (
    sum(dict(G.degree()).values())
    / G.number_of_nodes()
)

print(f"Average Degree: {avg_degree:.2f}")

print(f"Density: {nx.density(G):.8f}")

print("\n")

print("=" * 60)
print("TRY2 SAMPLE")
print("=" * 60)

H = nx.read_edgelist(
    SAMPLED_PATH,
    nodetype=int,
    create_using=nx.DiGraph()
)

print("Nodes:", H.number_of_nodes())
print("Edges:", H.number_of_edges())

avg_degree = (
    sum(dict(H.degree()).values())
    / H.number_of_nodes()
)

print(f"Average Degree: {avg_degree:.2f}")

print(f"Density: {nx.density(H):.8f}")