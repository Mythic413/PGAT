from pathlib import Path

import torch
import pandas as pd
import networkx as nx
import numpy as np

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

GRAPH_PATH = DATA_DIR / "higgs_5000.edgelist"
STRUCT_PATH = DATA_DIR / "structural_features.csv"
BEHAV_PATH = DATA_DIR / "behavioral_features.csv"

EDGE_PATH = DATA_DIR / "edge_probabilities.csv"

IC_PATH = DATA_DIR / "ic_labels.csv"
LT_PATH = DATA_DIR / "lt_labels.csv"

OUTPUT_PATH = DATA_DIR / "higgs_pyg.pt"
SPLIT_PATH = DATA_DIR / "splits.pt"

# Load Graph

print("\nLoading graph...")

G = nx.read_edgelist(
    GRAPH_PATH,
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

# Load Features

print("\nLoading features...")

struct = pd.read_csv(STRUCT_PATH)
behav = pd.read_csv(BEHAV_PATH)

struct = struct.sort_values("node").reset_index(drop=True)
behav = behav.sort_values("node").reset_index(drop=True)

# Alignment Checks

assert np.array_equal(
    struct["node"].values,
    behav["node"].values
), "Structural and behavioral feature nodes do not match."

assert np.array_equal(
    struct["node"].values,
    np.array(nodes)
), "Feature nodes and graph nodes do not match."

print("Feature alignment checks passed.")

# Structural Features (Step 3 FINAL)

X_struct = struct[
    [
        "in_degree",
        "out_degree",
        "degree",
        "clustering",
    ]
].values

# Behavioral Features (Step 4 FINAL)

X_behav = behav[
    [
        "activity_count",
        "retweet_ratio",
        "mention_ratio",
        "reply_ratio",
        "activity_span",
    ]
].values

# Combine Features

X_raw = np.concatenate(
    [X_struct, X_behav],
    axis=1
)

print("Raw Feature Shape:", X_raw.shape)

print("\nFeature Summary:")

feature_names = [
    "in_degree",
    "out_degree",
    "degree",
    "clustering",
    "activity_count",
    "retweet_ratio",
    "mention_ratio",
    "reply_ratio",
    "activity_span",
]

for idx, name in enumerate(feature_names):
    print(
        f"{name:18s} | "
        f"mean={X_raw[:, idx].mean():12.4f} | "
        f"std={X_raw[:, idx].std():12.4f}"
    )

# Edge Index & Edge Attributes

print("\nLoading edge probabilities...")

edge_prob = pd.read_csv(EDGE_PATH)

# Build edge_index and edge_attr together

src = []
dst = []
weights = []

for row in edge_prob.itertuples(index=False):

    u = row.src
    v = row.dst
    p = row.probability

    src.append(node_to_idx[u])
    dst.append(node_to_idx[v])
    weights.append(p)

edge_index = torch.tensor(
    [src, dst],
    dtype=torch.long
)

edge_attr = torch.tensor(
    weights,
    dtype=torch.float
).view(-1, 1)

print("Edge index shape :", edge_index.shape)
print("Edge attr shape  :", edge_attr.shape)

assert (
    edge_index.shape[1]
    ==
    edge_attr.shape[0]
), "Mismatch between edge_index and edge_attr"

print("Edge synchronization checks passed.")

# Labels

print("\nLoading labels...")

ic = pd.read_csv(IC_PATH)
lt = pd.read_csv(LT_PATH)

print("IC labels :", len(ic))
print("LT labels :", len(lt))

# Initialize labels

y_ic = torch.full(
    (len(nodes),),
    float("nan"),
    dtype=torch.float
)

y_lt = torch.full(
    (len(nodes),),
    float("nan"),
    dtype=torch.float
)

# Populate IC labels

for row in ic.itertuples(index=False):

    node = row.node

    y_ic[
        node_to_idx[node]
    ] = row.ic_mean

# Populate LT labels

for row in lt.itertuples(index=False):

    node = row.node

    y_lt[
        node_to_idx[node]
    ] = row.lt_mean

# Label Consistency Checks

ic_mask = ~torch.isnan(y_ic)
lt_mask = ~torch.isnan(y_lt)

assert torch.equal(
    ic_mask,
    lt_mask
), (
    "IC and LT labeled nodes do not match."
)

labeled = torch.where(ic_mask)[0]

print("Labeled nodes:", len(labeled))

assert len(labeled) == 1000, (
    f"Expected 1000 labeled nodes, "
    f"found {len(labeled)}."
)

print(" IC/LT consistency checks passed.")

# Label Statistics

print("\nIC Label Statistics:")

print(
    pd.Series(
        y_ic[ic_mask].numpy()
    ).describe()
)

print("\nLT Label Statistics:")

print(
    pd.Series(
        y_lt[lt_mask].numpy()
    ).describe()
)

# Train / Validation / Test Splits

print("\nBuilding degree-stratified splits")

# Degree quartiles on LABELED nodes only

labeled_nodes_original = [
    nodes[idx]
    for idx in labeled.numpy()
]

labeled_degrees = [
    G.degree(node)
    for node in labeled_nodes_original
]

split_df = pd.DataFrame({
    "idx": labeled.numpy(),
    "node": labeled_nodes_original,
    "degree": labeled_degrees,
})

split_df["quartile"] = pd.qcut(
    split_df["degree"],
    q=4,
    labels=False,
    duplicates="drop"
)

train_idx_list = []
val_idx_list = []
test_idx_list = []

# 70 / 15 / 15 split within each quartile

for q in sorted(split_df["quartile"].unique()):

    subset = split_df[
        split_df["quartile"] == q
    ]

    idxs = subset["idx"].values.copy()

    np.random.shuffle(idxs)

    n = len(idxs)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_idx_list.extend(
        idxs[:train_end]
    )

    val_idx_list.extend(
        idxs[train_end:val_end]
    )

    test_idx_list.extend(
        idxs[val_end:]
    )

train_idx = torch.tensor(
    train_idx_list,
    dtype=torch.long
)

val_idx = torch.tensor(
    val_idx_list,
    dtype=torch.long
)

test_idx = torch.tensor(
    test_idx_list,
    dtype=torch.long
)

# Masks

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

# Sanity Checks

assert (
    train_mask.sum()
    + val_mask.sum()
    + test_mask.sum()
    == len(labeled)
), "Masks do not cover all labeled nodes."

assert not torch.any(
    train_mask & val_mask
), "Train and validation overlap."

assert not torch.any(
    train_mask & test_mask
), "Train and test overlap."

assert not torch.any(
    val_mask & test_mask
), "Validation and test overlap."

print("Split integrity checks passed.")

# Leakage-Free Feature Scaling

print("\nScaling features...")

scaler = StandardScaler()

# Fit ONLY on training nodes

scaler.fit(
    X_raw[
        train_idx.numpy()
    ]
)

# Transform ALL nodes

X_scaled = scaler.transform(
    X_raw
)

print("Scaled Feature Shape:", X_scaled.shape)

# Scaling Sanity Check

train_scaled = X_scaled[
    train_idx.numpy()
]

print("\nTraining Feature Statistics")

print(
    "Mean:",
    np.round(
        train_scaled.mean(axis=0),
        4
    )
)

print(
    "Std :",
    np.round(
        train_scaled.std(axis=0),
        4
    )
)

print(
    "\nFeatures scaled using "
    "train nodes only."
)

# Build PyG Data Object

print("\nBuilding PyG Data object...")

data = Data(
    x=torch.tensor(
        X_scaled,
        dtype=torch.float
    ),
    edge_index=edge_index,
    edge_attr=edge_attr,
)

# Labels

data.y_ic = y_ic.float()
data.y_lt = y_lt.float()

# Masks

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Original Higgs Node IDs

data.node_ids = torch.tensor(
    nodes,
    dtype=torch.long
)

# Final Dataset Sanity Checks

print("\nRunning dataset sanity checks")

# Number of nodes
assert data.num_nodes == len(nodes), (
    "Node count mismatch."
)

# Edge consistency
assert (
    data.edge_index.shape[1]
    ==
    data.edge_attr.shape[0]
), (
    "Edge attribute mismatch."
)

# Feature dimensions
assert data.x.shape[0] == len(nodes), (
    "Feature row mismatch."
)

assert data.x.shape[1] == 9, (
    f"Expected 9 features, found {data.x.shape[1]}."
)

# Label consistency
assert (
    (~torch.isnan(data.y_ic)).sum()
    ==
    (~torch.isnan(data.y_lt)).sum()
), (
    "IC/LT label mismatch."
)

# Split consistency
assert (
    data.train_mask.sum()
    +
    data.val_mask.sum()
    +
    data.test_mask.sum()
    ==
    (~torch.isnan(data.y_ic)).sum()
), (
    "Split coverage mismatch."
)

print("Dataset sanity checks passed.")

# Save Dataset

print("\nSaving dataset...")

torch.save(
    data,
    OUTPUT_PATH
)

print("Saved:")
print(OUTPUT_PATH)

# Save Splits

torch.save(
    {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    },
    SPLIT_PATH
)

print("\nSaved:")
print(SPLIT_PATH)

# Final Summary


print("FINAL DATASET:")

print("Nodes           :", data.num_nodes)
print("Edges           :", data.edge_index.shape[1])

print("\nFeatures")
print("Shape           :", tuple(data.x.shape))

print("\nLabels")
print(
    "IC labeled      :",
    (~torch.isnan(data.y_ic)).sum().item()
)

print(
    "LT labeled      :",
    (~torch.isnan(data.y_lt)).sum().item()
)

print("\nSplits")
print(
    "Train           :",
    data.train_mask.sum().item()
)

print(
    "Validation      :",
    data.val_mask.sum().item()
)

print(
    "Test            :",
    data.test_mask.sum().item()
)

print("\nEdge Attributes")
print(
    "Shape           :",
    tuple(data.edge_attr.shape)
)

print("\nNode IDs")
print(
    "Stored          :",
    len(data.node_ids)
)

print("\nPyG Object")
print(data)
print(" Dataset saved successfully.")