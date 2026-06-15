from pathlib import Path
import pandas as pd
import networkx as nx

# ============================================================
# PROJECT_HAIL – Try2
# Step 5: Weighted Cascade Edge Probabilities

print("PROJECT_HAIL – Try2")
print("Step 5: Weighted Cascade Edge Probabilities")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

GRAPH_PATH = DATA_DIR / "higgs_5000.edgelist"
OUTPUT_PATH = DATA_DIR / "edge_probabilities.csv"

# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------

print("\nLoading Higgs-5000 graph...")

G = nx.read_edgelist(
    GRAPH_PATH,
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"Nodes : {G.number_of_nodes():,}")
print(f"Edges : {G.number_of_edges():,}")

# ------------------------------------------------------------
# Compute in-degrees
# ------------------------------------------------------------

print("\nComputing destination in-degrees...")

in_degree = dict(G.in_degree())

# ------------------------------------------------------------
# Assign Weighted Cascade probabilities
# ------------------------------------------------------------

print("\nAssigning Weighted Cascade probabilities...")

rows = []

for u, v in G.edges():
    # Weighted Cascade:
    # p(u,v) = 1 / in_degree(v)

    p = 1.0 / max(in_degree[v], 1)

    rows.append([u, v, p])

edge_df = pd.DataFrame(
    rows,
    columns=[
        "src",
        "dst",
        "probability"
    ]
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("\nProbability Statistics:")
print(edge_df["probability"].describe())

print("\nProbability Range:")
print(f"Minimum : {edge_df['probability'].min():.6f}")
print(f"Maximum : {edge_df['probability'].max():.6f}")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

edge_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nOutput Shape:")
print(edge_df.shape)

print("\nStep 5 Complete.")
print("Weighted Cascade edge probabilities finalized.")