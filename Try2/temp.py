import os
import shutil

print("=" * 70)
print("PROJECT HAIL DIRECTORY REORGANIZER")
print("=" * 70)

ROOT = os.getcwd()

# ==========================================================
# Create Folder Structure
# ==========================================================

folders = [
    "checkpoints",

    "results",
    "results/ic",
    "results/lt",

    "scripts",
    "scripts/preprocessing",
    "scripts/training",
]

print("\nCreating folders...")

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"[✓] {folder}")

# ==========================================================
# Files to Move
# ==========================================================

move_map = {
    # -----------------------------
    # Checkpoints
    # -----------------------------
    "best_gcn_ic.pt":
        "checkpoints/best_gcn_ic.pt",

    "best_pgat_ic.pt":
        "checkpoints/best_pgat_ic.pt",

    "best_pgatv2_mt.pt":
        "checkpoints/best_pgatv2_mt.pt",

    # -----------------------------
    # IC Results
    # -----------------------------
    "gcn_ic_results.txt":
        "results/ic/gcn_ic_results.txt",

    "pgat_ic_results.txt":
        "results/ic/pgat_ic_results.txt",

    "pgatv2_mt_results.txt":
        "results/ic/pgatv2_mt_results.txt",

    # -----------------------------
    # Preprocessing Scripts
    # -----------------------------
    "step2_extract_subgraph.py":
        "scripts/preprocessing/step2_extract_subgraph.py",

    "step3_structural_features.py":
        "scripts/preprocessing/step3_structural_features.py",

    "step4_behavioral_features.py":
        "scripts/preprocessing/step4_behavioral_features.py",

    "step5_edge_probabilities.py":
        "scripts/preprocessing/step5_edge_probabilities.py",

    "step5b_ic_labels.py":
        "scripts/preprocessing/step5b_ic_labels.py",

    "step6_lt_labels.py":
        "scripts/preprocessing/step6_lt_labels.py",

    "step7_build_dataset.py":
        "scripts/preprocessing/step7_build_dataset.py",

    "step7b_spearman_analysis.py":
        "scripts/preprocessing/step7b_spearman_analysis.py",

    # -----------------------------
    # Training Scripts
    # -----------------------------
    "train_gcn_ic.py":
        "scripts/training/train_gcn_ic.py",

    "train_pgatV2_ic.py":
        "scripts/training/train_pgatV2_ic.py",
}

# ==========================================================
# Move Files
# ==========================================================

print("\nMoving files...\n")

moved = 0
already = 0
missing = 0

for src, dst in move_map.items():

    if os.path.exists(src):

        if os.path.exists(dst):
            print(f"[SKIP] Destination exists: {dst}")
            already += 1

        else:
            shutil.move(src, dst)
            print(f"[MOVE] {src} -> {dst}")
            moved += 1

    else:
        print(f"[MISS] {src}")
        missing += 1

# ==========================================================
# Create Placeholder Files
# ==========================================================

print("\nCreating future result files...")

placeholders = [
    "results/ic/graphsage_results.txt",
    "results/ic/gat_results.txt",
    "results/ic/gin_results.txt",

    "results/lt/gcn_results.txt",
    "results/lt/graphsage_results.txt",
    "results/lt/gat_results.txt",
    "results/lt/gin_results.txt",
    "results/lt/pgatv2_mt_results.txt",
]

for file in placeholders:

    if not os.path.exists(file):
        open(file, "w").close()
        print(f"[CREATE] {file}")

# ==========================================================
# README
# ==========================================================

readme = "README.md"

if not os.path.exists(readme):

    with open(readme, "w") as f:
        f.write(
"""# PROJECT HAIL

Diffusion-Aware Influence Estimation using GNNs.

Completed:
- GCN (IC)
- PGAT Ablations
- PGAT-v2-MT

Pending:
- GraphSAGE (IC)
- GAT (IC)
- GIN (IC)

Then:
- LT experiments.
"""
        )

    print(f"[CREATE] {readme}")

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 70)
print("REORGANIZATION COMPLETE")
print("=" * 70)

print(f"Moved Files        : {moved}")
print(f"Already Organized  : {already}")
print(f"Missing Files      : {missing}")

print("\nFinal Structure:")

print("""
PROJECT_HAIL/
│
├── checkpoints/
├── data/
├── results/
│   ├── ic/
│   └── lt/
│
├── scripts/
│   ├── preprocessing/
│   └── training/
│
├── temp.py
└── README.md
""")

print("Done.")
print("=" * 70)