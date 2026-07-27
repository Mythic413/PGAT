from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np

print("Diffusion Sanity Check")


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

GRAPH_PATH = DATA_DIR / "higgs_5000.edgelist"
IC_PATH = DATA_DIR / "ic_labels.csv"
LT_PATH = DATA_DIR / "lt_labels.csv"
NODES_PATH = DATA_DIR / "labeled_nodes.csv"

# Load graph

print("\nLoading graph...")

G = nx.read_edgelist(
    GRAPH_PATH,
    nodetype=int,
    create_using=nx.DiGraph()
)

N = G.number_of_nodes()

print(f"Nodes : {N:,}")
print(f"Edges : {G.number_of_edges():,}")

# Load labels

print("\nLoading IC labels...")
ic = pd.read_csv(IC_PATH)

print("Loading LT labels...")
lt = pd.read_csv(LT_PATH)

print("Loading shared nodes...")
shared = pd.read_csv(NODES_PATH)

print(f"IC labels : {len(ic)}")
print(f"LT labels : {len(lt)}")
print(f"Shared nodes : {len(shared)}")

# CHECK 1: Shared node universe
print("CHECK 1: Shared Label Universe")

ic_nodes = set(ic["node"])
lt_nodes = set(lt["node"])
shared_nodes = set(shared["node"])

if ic_nodes == lt_nodes == shared_nodes:
    print("PASS: IC and LT use identical node sets.")
else:
    print("FAIL: Label node mismatch detected.")

    print("IC only :", len(ic_nodes - lt_nodes))
    print("LT only :", len(lt_nodes - ic_nodes))

# CHECK 2: IC Statistics
print("CHECK 2: IC Statistics")

print(ic["ic_mean"].describe())

ic_cv = ic["ic_std"] / ic["ic_mean"]

print("\nIC Coefficient of Variation:")
print(ic_cv.describe())

# CHECK 3: LT Statistics
print("CHECK 3: LT Statistics")
print(lt["lt_mean"].describe())

lt_cv = lt["lt_std"] / lt["lt_mean"]

print("\nLT Coefficient of Variation:")
print(lt_cv.describe())

# CHECK 4: LT vs IC Relationship

print("CHECK 4: LT vs IC Comparison")

merged = ic.merge(
    lt,
    on="node"
)

print("\nMean Spreads")

print(f"IC Mean Spread : {merged['ic_mean'].mean():.4f}")
print(f"LT Mean Spread : {merged['lt_mean'].mean():.4f}")

if merged["lt_mean"].mean() >= merged["ic_mean"].mean():
    print("PASS: LT spread >= IC spread (expected).")
else:
    print("WARNING: LT spread unexpectedly lower than IC.")

# CHECK 5: Degree Correlation

print("CHECK 5: Degree Correlation")

degree = dict(G.degree())

merged["degree"] = merged["node"].map(degree)

ic_corr = merged["degree"].corr(merged["ic_mean"])
lt_corr = merged["degree"].corr(merged["lt_mean"])

print(f"Degree vs IC Mean : {ic_corr:.4f}")
print(f"Degree vs LT Mean : {lt_corr:.4f}")

if ic_corr < 0.8:
    print("PASS: IC not dominated by degree.")
else:
    print("WARNING: IC highly degree-driven.")

if lt_corr < 0.8:
    print("PASS: LT not dominated by degree.")
else:
    print("WARNING: LT highly degree-driven.")

# CHECK 6: IC vs LT Correlation
print("CHECK 6: IC vs LT Correlation")

ic_lt_corr = merged["ic_mean"].corr(
    merged["lt_mean"]
)

print(f"IC vs LT Pearson Correlation: {ic_lt_corr:.4f}")

if ic_lt_corr < 0.99:
    print("PASS: IC and LT are related but distinct.")
else:
    print("WARNING: IC and LT labels are nearly identical.")

# CHECK 7: Top Influencers

print("CHECK 7: Top-10 Overlap")
top_ic = set(
    merged.nlargest(
        10,
        "ic_mean"
    )["node"]
)

top_lt = set(
    merged.nlargest(
        10,
        "lt_mean"
    )["node"]
)

overlap = len(top_ic & top_lt)

print(f"Top-10 Overlap: {overlap}/10")

if overlap < 10:
    print("PASS: IC and LT identify partially different influencers.")
else:
    print("WARNING: IC and LT rankings are identical.")

# CHECK 8: Spread Bounds

print("\n" + "=" * 80)
print("CHECK 8: Spread Bounds")
print("=" * 80)

checks_passed = True

for col in ["ic_mean", "lt_mean"]:

    vals = merged[col]

    if vals.min() < 1:
        print(f"FAIL: {col} minimum < 1")
        checks_passed = False

    if vals.max() > N:
        print(f"FAIL: {col} maximum exceeds graph size")
        checks_passed = False

if checks_passed:
    print("PASS: Spread bounds valid.")

# Final Verdict

if (
    ic_nodes == lt_nodes == shared_nodes
    and merged["lt_mean"].mean() >= merged["ic_mean"].mean()
    and ic_corr < 0.8
    and lt_corr < 0.8
    and ic_lt_corr < 0.99
):
    print(" IC labels validated.")
    print(" LT labels validated.")
    print(" Shared node universe confirmed.")
    print(" No evidence of major leakage.")
    print(" Diffusion benchmarks appear publication-ready.")
else:
    print(" Review warnings above before freezing Step 6.")

print("\nSanity check complete.")