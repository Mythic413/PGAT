import gzip
import networkx as nx
from collections import deque

TARGET_NODES = 5000

print("Loading Higgs social network...")

G = nx.DiGraph()

with gzip.open("data/higgs-social_network.edgelist.gz", "rt") as f:
    for line in f:
        u, v = map(int, line.strip().split())
        G.add_edge(u, v)

print(f"Original Graph:")
print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

# Convert to undirected for component extraction
G_und = G.to_undirected()

print("\nFinding largest connected component...")

largest_cc = max(nx.connected_components(G_und), key=len)

G_cc = G.subgraph(largest_cc).copy()

print(f"Largest CC Nodes: {G_cc.number_of_nodes():,}")
print(f"Largest CC Edges: {G_cc.number_of_edges():,}")

# BFS sampling
print(f"\nExtracting {TARGET_NODES} nodes using BFS...")

deg = dict(G_cc.degree())

start = max(deg, key=deg.get)

print("Starting BFS from hub:", start)
print("Hub degree:", deg[start])

visited = set([start])
queue = deque([start])

while queue and len(visited) < TARGET_NODES:
    u = queue.popleft()

    neighbors = list(G_cc.successors(u)) + list(G_cc.predecessors(u))

    for v in neighbors:
        if v not in visited:
            visited.add(v)
            queue.append(v)

            if len(visited) >= TARGET_NODES:
                break

subgraph = G_cc.subgraph(visited).copy()

print("\nFinal Subgraph Statistics:")
print(f"Nodes: {subgraph.number_of_nodes():,}")
print(f"Edges: {subgraph.number_of_edges():,}")

avg_degree = sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes()

print(f"Average Degree: {avg_degree:.2f}")

density = nx.density(subgraph)

print(f"Density: {density:.6f}")

nx.write_edgelist(
    subgraph,
    "data/higgs_5000.edgelist",
    data=False
)

print("\nSaved:")
print("data/higgs_5000.edgelist")