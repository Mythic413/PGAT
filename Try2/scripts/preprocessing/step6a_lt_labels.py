from pathlib import Path
import random

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# PROJECT_HAIL – Try2
# Step 6a: LT Label Generation
# ============================================================

MC_RUNS = 200
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

GRAPH_PATH = DATA_DIR / "higgs_5000.edgelist"
EDGE_PATH = DATA_DIR / "edge_probabilities.csv"
NODES_PATH = DATA_DIR / "labeled_nodes.csv"
OUTPUT_PATH = DATA_DIR / "lt_labels.csv"

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
# Load WC weights
# ------------------------------------------------------------

print("\nLoading edge probabilities...")

edge_df = pd.read_csv(EDGE_PATH)

for row in edge_df.itertuples(index=False):
    G[row.src][row.dst]["w"] = row.probability

print(f"Loaded weights: {len(edge_df):,}")

# ------------------------------------------------------------
# Load shared labeled nodes
# ------------------------------------------------------------

print("\nLoading labeled nodes...")

selected = pd.read_csv(NODES_PATH)["node"].tolist()

print(f"Labeled nodes: {len(selected)}")

# ------------------------------------------------------------
# LT Simulation (Frontier Accumulator)
# ------------------------------------------------------------

def run_lt(seed_node):

    thresholds = {
        node: random.random()
        for node in G.nodes()
    }

    influence = {}

    active = {seed_node}
    frontier = {seed_node}

    while frontier:

        new_frontier = set()

        for u in frontier:

            for v in G.successors(u):

                if v in active:
                    continue

                influence[v] = (
                    influence.get(v, 0.0)
                    + G[u][v]["w"]
                )

                if influence[v] >= thresholds[v]:
                    active.add(v)
                    new_frontier.add(v)

        frontier = new_frontier

    return len(active)

# ------------------------------------------------------------
# Generate LT labels
# ------------------------------------------------------------

print("\nGenerating LT labels...")

rows = []

for node in tqdm(selected):

    spreads = []

    for _ in range(MC_RUNS):
        spreads.append(
            run_lt(node)
        )

    mean_spread = np.mean(spreads)

    std_spread = np.std(
        spreads,
        ddof=1
    )

    rows.append([
        node,
        mean_spread,
        std_spread
    ])

labels = pd.DataFrame(
    rows,
    columns=[
        "node",
        "lt_mean",
        "lt_std"
    ]
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("\nLT Mean Statistics:")
print(labels["lt_mean"].describe())

print("\nLT Std Statistics:")
print(labels["lt_std"].describe())

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

labels.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nOutput Shape:")
print(labels.shape)

print("\nStep 6a Complete.")
print("LT labels finalized.")