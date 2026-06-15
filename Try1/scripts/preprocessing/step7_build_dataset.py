import torch
import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.data import Data

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", device)

# ====================================================
# Load graph
# ====================================================

print("\nLoading graph...")

G = nx.read_edgelist(
    "data/higgs_5000.edgelist",
    nodetype=int,
    create_using=nx.DiGraph()
)

nodes = sorted(G.nodes())

node_to_idx = {
    node: idx
    for idx, node in enumerate(nodes)
}

print("Nodes:", len(nodes))
print("Edges:", G.number_of_edges())

# ====================================================
# Load features
# ====================================================

print("\nLoading features...")

struct = pd.read_csv("data/structural_features.csv")
behav = pd.read_csv("data/behavioral_features.csv")

struct = struct.sort_values("node")
behav = behav.sort_values("node")

X_struct = struct[
    [
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
        "pagerank",
        "clustering",
    ]
].values

X_behav = behav[
    [
        "activity_count",
        "retweet_ratio",
        "mention_ratio",
        "reply_ratio",
        "activity_span",
    ]
].values

X = np.concatenate(
    [X_struct, X_behav],
    axis=1
)

print("Feature shape:", X.shape)

# ====================================================
# Edge Index
# ====================================================

print("\nBuilding edge index...")

src = []
dst = []

for u, v in G.edges():

    src.append(node_to_idx[u])
    dst.append(node_to_idx[v])

edge_index = torch.tensor(
    [src, dst],
    dtype=torch.long
)

print("Edge index shape:", edge_index.shape)

# ====================================================
# Edge Attributes
# ====================================================

print("\nLoading edge probabilities...")

edge_prob = pd.read_csv(
    "data/edge_probabilities.csv"
)

edge_attr = torch.tensor(
    edge_prob["probability"].values,
    dtype=torch.float
).view(-1, 1)

print("Edge attr shape:", edge_attr.shape)

# ====================================================
# Labels
# ====================================================

print("\nLoading labels...")

ic = pd.read_csv("data/ic_labels.csv")
lt = pd.read_csv("data/lt_labels.csv")

y_ic = torch.full(
    (len(nodes),),
    float("nan")
)

y_lt = torch.full(
    (len(nodes),),
    float("nan")
)

for _, row in ic.iterrows():

    idx = node_to_idx[row["node"]]

    y_ic[idx] = row["ic_mean"]

for _, row in lt.iterrows():

    idx = node_to_idx[row["node"]]

    y_lt[idx] = row["lt_mean"]

# ====================================================
# Masks
# ====================================================

print("\nBuilding masks...")

labeled = torch.where(
    ~torch.isnan(y_ic)
)[0]

perm = labeled[
    torch.randperm(len(labeled))
]

n = len(perm)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_idx = perm[:train_end]
val_idx = perm[train_end:val_end]
test_idx = perm[val_end:]

train_mask = torch.zeros(
    len(nodes),
    dtype=torch.bool
)

val_mask = torch.zeros(
    len(nodes),
    dtype=torch.bool
)

test_mask = torch.zeros(
    len(nodes),
    dtype=torch.bool
)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

print("Train:", train_mask.sum().item())
print("Val  :", val_mask.sum().item())
print("Test :", test_mask.sum().item())

# ====================================================
# Data Object
# ====================================================

data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    edge_attr=edge_attr,
)

data.y_ic = y_ic.float()
data.y_lt = y_lt.float()

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

torch.save(
    data,
    "data/higgs_pyg.pt"
)

print("\nSaved:")
print("data/higgs_pyg.pt")

print("\nSummary")
print(data)

torch.save(
    {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    },
    "data/splits.pt"
)