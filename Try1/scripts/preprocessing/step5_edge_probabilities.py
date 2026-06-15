import gzip
import pandas as pd
import networkx as nx

print("Loading Higgs-5000...")

G = nx.read_edgelist(
    "data/higgs_5000.edgelist",
    nodetype=int,
    create_using=nx.DiGraph()
)

print(f"Nodes: {G.number_of_nodes():,}")
print(f"Edges: {G.number_of_edges():,}")

print("\nLoading features...")

struct = pd.read_csv("data/structural_features.csv")
behav = pd.read_csv("data/behavioral_features.csv")

struct = struct.set_index("node")
behav = behav.set_index("node")

# ---------------------------------------------------
# Build retweet edge set
# ---------------------------------------------------

print("\nScanning retweet network...")

node_set = set(G.nodes())

retweet_edges = set()
matched_events = 0

with gzip.open("data/higgs-retweet_network.edgelist.gz", "rt") as f:

    for line in f:

        u, v, w = line.strip().split()

        u = int(u)
        v = int(v)

        if u in node_set and v in node_set:

            retweet_edges.add((u, v))
            matched_events += 1

print(f"Matched retweet events: {matched_events:,}")

# ---------------------------------------------------
# Assign probabilities
# ---------------------------------------------------

print("\nAssigning probabilities...")

rows = []

boosted = 0
non_boosted = 0

for u, v in G.edges():

    # Structural susceptibility
    P_struct = struct.loc[v, "degree"]

    # Behavioral susceptibility
    P_behav = (
        0.5 * behav.loc[v, "activity_count"]
        + 0.5 * behav.loc[v, "retweet_ratio"]
    )

    # Base probability
    p = 0.5 * P_struct + 0.5 * P_behav

    # Retweet boost
    if (u, v) in retweet_edges:
        p *= 1.5
        boosted += 1
    else:
        non_boosted += 1

    # IC clipping
    p = max(0.01, min(0.5, p))

    rows.append([u, v, p])

edge_df = pd.DataFrame(
    rows,
    columns=["src", "dst", "probability"]
)

print("\nProbability Statistics:")
print(edge_df["probability"].describe())

print("\nRetweet Boosting:")
print(f"Boosted edges     : {boosted:,}")
print(f"Non-boosted edges : {non_boosted:,}")

edge_df.to_csv(
    "data/edge_probabilities.csv",
    index=False
)

print("\nSaved:")
print("data/edge_probabilities.csv")

print("\nShape:")
print(edge_df.shape)