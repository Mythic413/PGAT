import pandas as pd
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

MC_RUNS = 200
NUM_LABELS = 1000
SEED = 42
LAMBDA = 0.1

random.seed(SEED)
np.random.seed(SEED)

print("Loading graph...")

G = nx.read_edgelist(
    "data/higgs_5000.edgelist",
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

print("\nLoading edge probabilities...")

edge_df = pd.read_csv("data/edge_probabilities.csv")

# --------------------------------------------------
# Build LT weights
# --------------------------------------------------

print("Building LT weights...")

incoming_sum = {}

for node in G.nodes():
    incoming_sum[node] = 0.0

for _, row in edge_df.iterrows():
    incoming_sum[row.dst] += row.probability

lt_weight = {}

for _, row in edge_df.iterrows():

    u = row.src
    v = row.dst

    lt_weight[(u, v)] = (
        row.probability /
        incoming_sum[v]
    )

print("LT weights built.")

# --------------------------------------------------
# Select same 1000 nodes
# --------------------------------------------------

degree = dict(G.degree())

deg_df = pd.DataFrame({
    "node": list(degree.keys()),
    "degree": list(degree.values())
})

deg_df["quartile"] = pd.qcut(
    deg_df["degree"],
    q=4,
    labels=False
)

selected = []

for q in range(4):

    nodes_q = deg_df[
        deg_df["quartile"] == q
    ]["node"].tolist()

    selected.extend(
        random.sample(nodes_q, 250)
    )

print(f"Labeled nodes: {len(selected)}")

# --------------------------------------------------
# LT Simulation
# --------------------------------------------------

def run_lt(seed_node):

    thresholds = {
        node: random.random()
        for node in G.nodes()
    }

    active = {seed_node}

    changed = True

    while changed:

        changed = False

        new_active = set()

        for v in G.nodes():

            if v in active:
                continue

            influence = 0.0

            for u in G.predecessors(v):

                if u in active:

                    influence += lt_weight[(u, v)]

            if influence >= thresholds[v]:

                new_active.add(v)

        if new_active:

            active.update(new_active)
            changed = True

    return len(active)

# --------------------------------------------------
# Generate LT labels
# --------------------------------------------------

print("\nGenerating LT labels...")

rows = []

for node in tqdm(selected):

    spreads = []

    for _ in range(MC_RUNS):

        spreads.append(
            run_lt(node)
        )

    mean_spread = np.mean(spreads)
    std_spread = np.std(spreads)

    robust_spread = (
        mean_spread
        - LAMBDA * std_spread
    )

    rows.append([
        node,
        mean_spread,
        std_spread,
        robust_spread
    ])

labels = pd.DataFrame(
    rows,
    columns=[
        "node",
        "lt_mean",
        "lt_std",
        "lt_robust"
    ]
)

print("\nLT Mean Statistics:")
print(labels["lt_mean"].describe())

print("\nLT Robust Statistics:")
print(labels["lt_robust"].describe())

labels.to_csv(
    "data/lt_labels.csv",
    index=False
)

print("\nSaved:")
print("data/lt_labels.csv")