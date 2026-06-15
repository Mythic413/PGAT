import gzip
import networkx as nx
import numpy as np
from collections import deque

TARGET_NODES = 5000

# Canonical Forest Fire parameters
FORWARD_BURN_PROB = 0.60
BACKWARD_BURN_PROB = 0.15

SEED = 42
rng = np.random.default_rng(SEED)

print("Loading Higgs social network...")

G = nx.DiGraph()

with gzip.open("data/higgs-social_network.edgelist.gz", "rt") as f:
    for line in f:
        u, v = map(int, line.strip().split())
        G.add_edge(u, v)

print("\nOriginal Graph:")
print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

# --------------------------------------------------
# Largest Weakly Connected Component
# --------------------------------------------------

print("\nFinding largest connected component...")

G_und = G.to_undirected()

largest_cc = max(nx.connected_components(G_und), key=len)

G_cc = G.subgraph(largest_cc).copy()

print(f"Largest CC Nodes: {G_cc.number_of_nodes():,}")
print(f"Largest CC Edges: {G_cc.number_of_edges():,}")

# --------------------------------------------------
# Canonical Forest Fire Sampling
# --------------------------------------------------

print(f"\nExtracting {TARGET_NODES} nodes using Canonical Forest Fire...")

all_nodes = np.array(list(G_cc.nodes()))

# Initial ignition
seed_node = rng.choice(all_nodes)

visited = set([seed_node])
fire_queue = deque([seed_node])

print("Initial ignition node:", seed_node)

restart_count = 0

while len(visited) < TARGET_NODES:

    # Restart if fire dies out
    if not fire_queue:

        remaining = np.array(
            list(set(all_nodes) - visited)
        )

        if len(remaining) == 0:
            break

        new_seed = rng.choice(remaining)

        visited.add(new_seed)
        fire_queue.append(new_seed)

        restart_count += 1

        print(
            f"Restart #{restart_count}: "
            f"igniting node {new_seed} "
            f"(sample={len(visited)})"
        )

    u = fire_queue.popleft()

    successors = list(G_cc.successors(u))
    predecessors = list(G_cc.predecessors(u))

    # Only unvisited nodes
    unvisited_successors = [
        v for v in successors
        if v not in visited
    ]

    unvisited_predecessors = [
        v for v in predecessors
        if v not in visited
    ]

    # --------------------------------------------------
    # Geometric Burning
    # --------------------------------------------------

    forward_count = (
        rng.geometric(1.0 - FORWARD_BURN_PROB) - 1
    )

    backward_count = (
        rng.geometric(1.0 - BACKWARD_BURN_PROB) - 1
    )

    # --------------------------------------------------
    # Sample exact neighbors to burn
    # --------------------------------------------------

    if len(unvisited_successors) > 0 and forward_count > 0:

        forward_size = min(
            forward_count,
            len(unvisited_successors)
        )

        to_burn_forward = rng.choice(
            unvisited_successors,
            size=forward_size,
            replace=False
        )

    else:
        to_burn_forward = []

    if len(unvisited_predecessors) > 0 and backward_count > 0:

        backward_size = min(
            backward_count,
            len(unvisited_predecessors)
        )

        to_burn_backward = rng.choice(
            unvisited_predecessors,
            size=backward_size,
            replace=False
        )

    else:
        to_burn_backward = []

    # --------------------------------------------------
    # Merge and ignite
    # --------------------------------------------------

    to_burn = set(to_burn_forward) | set(to_burn_backward)

    for v in to_burn:

        if len(visited) >= TARGET_NODES:
            break

        visited.add(v)
        fire_queue.append(v)

# --------------------------------------------------
# Build induced subgraph
# --------------------------------------------------

subgraph = G_cc.subgraph(visited).copy()

print("\nFinal Subgraph Statistics:")

print(f"Nodes: {subgraph.number_of_nodes():,}")
print(f"Edges: {subgraph.number_of_edges():,}")

avg_degree = (
    sum(dict(subgraph.degree()).values())
    / subgraph.number_of_nodes()
)

print(f"Average Degree: {avg_degree:.2f}")

density = nx.density(subgraph)

print(f"Density: {density:.6f}")

clustering = nx.average_clustering(
    subgraph.to_undirected()
)

print(
    f"Average Clustering Coefficient: "
    f"{clustering:.4f}"
)

print(f"Forest Fire Restarts: {restart_count}")

# --------------------------------------------------
# Save
# --------------------------------------------------

nx.write_edgelist(
    subgraph,
    "Try2/data/higgs_5000.edgelist",
    data=False
)

print("\nSaved:")
print("data/higgs_5000.edgelist")