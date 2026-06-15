from pathlib import Path
import random

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# PROJECT_HAIL – Try2
# Step 6b: IC Label Generation
# ============================================================
#
# Objective:
# Generate leakage-free IC labels using the canonical
# Independent Cascade model and the finalized
# Weighted Cascade probabilities from Step 5.
#
# Method:
# - Weighted Cascade probabilities from Step 5
# - Standard Independent Cascade diffusion
# - Monte Carlo estimation of expected spread
# - Degree-stratified node selection
#
# Outputs:
# Try2/data/ic_labels.csv
#
# Columns:
# node
# ic_mean
# ic_std
#
# ============================================================

MC_RUNS = 200
NUM_LABELS = 1000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("PROJECT_HAIL – Try2")
print("Step 6b: IC Label Generation")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

GRAPH_PATH = DATA_DIR / "higgs_5000.edgelist"
EDGE_PROB_PATH = DATA_DIR / "edge_probabilities.csv"
OUTPUT_PATH = DATA_DIR / "ic_labels.csv"

# ------------------------------------------------------------
# Load graph
# ------------------------------------------------------------

print("\nLoading graph...")

G = nx.read_edgelist(
    GRAPH_PATH,
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

# ------------------------------------------------------------
# Load edge probabilities
# ------------------------------------------------------------

print("\nLoading edge probabilities...")

edge_df = pd.read_csv(EDGE_PROB_PATH)

print(f"Edge probabilities: {len(edge_df):,}")

# ------------------------------------------------------------
# Attach probabilities to graph
# ------------------------------------------------------------

print("\nAttaching probabilities to graph...")

for row in edge_df.itertuples(index=False):
    G[row.src][row.dst]["p"] = row.probability

# ------------------------------------------------------------
# Degree-stratified node selection
# ------------------------------------------------------------

print("\nBuilding degree quartiles...")

degree = dict(G.degree())

deg_df = pd.DataFrame({
    "node": list(degree.keys()),
    "degree": list(degree.values())
})

deg_df["quartile"] = pd.qcut(
    deg_df["degree"],
    q=4,
    labels=False,
    duplicates="drop"
)

quartiles = sorted(deg_df["quartile"].dropna().unique())

sample_per_quartile = NUM_LABELS // len(quartiles)

selected = []

for q in quartiles:

    nodes_q = deg_df[
        deg_df["quartile"] == q
    ]["node"].tolist()

    k = min(sample_per_quartile, len(nodes_q))

    selected.extend(
        random.sample(nodes_q, k)
    )

selected = sorted(selected)

print(f"Labeled nodes: {len(selected)}")

# ------------------------------------------------------------
# Save shared labeled nodes
# ------------------------------------------------------------

labeled_nodes_path = DATA_DIR / "labeled_nodes.csv"

pd.DataFrame({
    "node": selected
}).to_csv(
    labeled_nodes_path,
    index=False
)

print("\nSaved shared labeled nodes:")
print(labeled_nodes_path)

# ------------------------------------------------------------
# Independent Cascade simulator
# ------------------------------------------------------------

def run_ic(seed_node):

    activated = {seed_node}
    frontier = {seed_node}

    while frontier:

        new_frontier = set()

        for u in frontier:

            for v in G.successors(u):

                if v in activated:
                    continue

                p = G[u][v]["p"]

                # Standard IC coin flip
                if random.random() < p:
                    activated.add(v)
                    new_frontier.add(v)

        frontier = new_frontier

    return len(activated)

# ------------------------------------------------------------
# Generate labels
# ------------------------------------------------------------

print("\nGenerating IC labels...")

rows = []

for node in tqdm(selected):

    spreads = []

    for _ in range(MC_RUNS):
        spreads.append(
            run_ic(node)
        )

    mean_spread = np.mean(spreads)
    std_spread = np.std(spreads)

    rows.append([
        node,
        mean_spread,
        std_spread
    ])

labels = pd.DataFrame(
    rows,
    columns=[
        "node",
        "ic_mean",
        "ic_std"
    ]
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("\nIC Mean Statistics:")
print(labels["ic_mean"].describe())

print("\nIC Std Statistics:")
print(labels["ic_std"].describe())

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

labels.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nOutput Shape:")
print(labels.shape)

print("\nStep 6aComplete.")
print("IC labels finalized.")