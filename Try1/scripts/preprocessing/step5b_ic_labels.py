import pandas as pd
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

MC_RUNS = 200
NUM_LABELS = 1000
SEED = 42

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

edge_prob = {
    (row.src, row.dst): row.probability
    for _, row in edge_df.iterrows()
}

print("Building degree quartiles...")

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

SCALES = [0.15, 0.20, 0.25, 0.30]

def run_ic(seed_node):

    activated = {seed_node}
    frontier = {seed_node}

    IC_SCALE = random.choice(SCALES)

    while frontier:

        new_frontier = set()

        for u in frontier:

            for v in G.successors(u):

                if v in activated:
                    continue

                p = edge_prob[(u, v)] * IC_SCALE

                p = max(0.001, min(0.25, p))

                if random.random() < p:
                    activated.add(v)
                    new_frontier.add(v)

        frontier = new_frontier

    return len(activated)

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

    robust_spread = mean_spread - 0.1* std_spread

    rows.append([
    node,
    mean_spread,
    std_spread,
    robust_spread])

labels = pd.DataFrame(
    rows,
    columns=[
        "node",
        "ic_mean",
        "ic_std",
        "ic_robust"
    ]
)
print("\nIC Mean Statistics:")
print(labels["ic_mean"].describe())

print("\nIC Robust Statistics:")
print(labels["ic_robust"].describe())

print("\nRobustness Penalty:")
print((labels["ic_mean"] - labels["ic_robust"]).describe())

print("\nIC Statistics:")
print(labels["ic_mean"].describe())

labels.to_csv(
    "data/ic_labels.csv",
    index=False
)

print("\nSaved:")
print("data/ic_labels.csv")