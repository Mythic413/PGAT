import pandas as pd
from scipy.stats import spearmanr

print("Loading data...")

# Load structural features
struct = pd.read_csv("data/structural_features.csv")

# Load labels
ic = pd.read_csv("data/ic_labels.csv")
lt = pd.read_csv("data/lt_labels.csv")

# Merge on node
ic_df = struct.merge(
    ic[["node", "ic_mean"]],
    on="node"
)

lt_df = struct.merge(
    lt[["node", "lt_mean"]],
    on="node"
)

features = [
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering"
]

print("\n==========================")
print("IC SPEARMAN CORRELATIONS")
print("==========================")

ic_results = []

for feat in features:

    rho, p = spearmanr(
        ic_df[feat],
        ic_df["ic_mean"]
    )

    ic_results.append((feat, rho))

    print(
        f"{feat:15s} "
        f"rho = {rho:.4f}"
    )

print("\n==========================")
print("LT SPEARMAN CORRELATIONS")
print("==========================")

lt_results = []

for feat in features:

    rho, p = spearmanr(
        lt_df[feat],
        lt_df["lt_mean"]
    )

    lt_results.append((feat, rho))

    print(
        f"{feat:15s} "
        f"rho = {rho:.4f}"
    )

# Sort results
print("\n==========================")
print("IC FEATURE RANKING")
print("==========================")

for feat, rho in sorted(
    ic_results,
    key=lambda x: abs(x[1]),
    reverse=True
):
    print(
        f"{feat:15s} "
        f"{rho:.4f}"
    )

print("\n==========================")
print("LT FEATURE RANKING")
print("==========================")

for feat, rho in sorted(
    lt_results,
    key=lambda x: abs(x[1]),
    reverse=True
):
    print(
        f"{feat:15s} "
        f"{rho:.4f}"
    )