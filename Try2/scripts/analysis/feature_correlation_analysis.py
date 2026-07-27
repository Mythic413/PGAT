from pathlib import Path
import pandas as pd
import numpy as np
import torch

from scipy.stats import spearmanr
from scipy.stats import false_discovery_control

# Configuration

DATA_DIR = Path("Try2/data")
RESULTS_DIR = Path("Try2/results/feature_analysis")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ALPHA = 0.05

STRUCTURAL_FEATURES = [
    "in_degree",
    "out_degree",
    "degree",
    "clustering"
]

BEHAVIORAL_FEATURES = [
    "activity_count",
    "retweet_ratio",
    "mention_ratio",
    "reply_ratio",
    "activity_span"
]

FEATURES = STRUCTURAL_FEATURES + BEHAVIORAL_FEATURES

# Load Frozen Artifacts
def load_data():

    print("Loading frozen artifacts...")

    struct = pd.read_csv(
        DATA_DIR / "structural_features.csv"
    )

    behavioral = pd.read_csv(
        DATA_DIR / "behavioral_features.csv"
    )

    ic = pd.read_csv(
        DATA_DIR / "ic_labels.csv"
    )

    lt = pd.read_csv(
        DATA_DIR / "lt_labels.csv"
    )

    splits = torch.load(
        DATA_DIR / "splits.pt",
        weights_only=False
    )

    print("Structural shape :", struct.shape)
    print("Behavioral shape :", behavioral.shape)
    print("IC labels shape  :", ic.shape)
    print("LT labels shape  :", lt.shape)

    # Merge structural + behavioral

    features = struct.merge(
        behavioral,
        on="node",
        how="inner"
    )

    # Merge labels

    merged = (
        features
        .merge(
            ic[["node", "ic_mean"]],
            on="node",
            how="inner"
        )
        .merge(
            lt[["node", "lt_mean"]],
            on="node",
            how="inner"
        )
    )

    print("\nMerged labeled shape:", merged.shape)

    # Sanity checks

    expected_keys = {
        "train_idx",
        "val_idx",
        "test_idx"
    }

    if not expected_keys.issubset(splits.keys()):
        raise KeyError(
            "splits.pt is missing required keys.\n"
            f"Expected: {expected_keys}\n"
            f"Found: {set(splits.keys())}"
        )

    if len(merged) != 1000:
        raise ValueError(
            "Expected 1000 shared labeled nodes, "
            f"found {len(merged)}."
        )

    # Determine train nodes using frozen PyG object

    pyg_data = torch.load(
    DATA_DIR / "higgs_pyg.pt",
    weights_only=False)

    required_attrs = [
    "train_mask",
    "node_ids"]

    for attr in required_attrs:
        if not hasattr(pyg_data, attr):
            raise AttributeError(
                f"{attr} not found in higgs_pyg.pt")

    train_mask = pyg_data.train_mask.cpu()

    node_ids = pyg_data.node_ids.cpu().numpy()

    if len(train_mask) != len(node_ids):
        raise ValueError(
        "train_mask and node_ids have "
        "different lengths.")

# Node IDs belonging to the frozen training split
    train_nodes = set(
    node_ids[
        train_mask.numpy()])

    print("Total graph train nodes:", len(train_nodes))

# Restrict to labeled nodes only

    labeled_train_nodes = set(
        merged[
            merged["node"].isin(train_nodes)]["node"])

    print(
    "Labeled train nodes:",
    len(labeled_train_nodes))

    if len(labeled_train_nodes) != 700:
        raise ValueError(
        "Expected 700 labeled training nodes, "
        f"found {len(labeled_train_nodes)}.")

    return merged, labeled_train_nodes
# Dataset Modes

def get_dataset(
    merged_df,
    train_nodes,
    mode="all"
):

    if mode == "all":

        df = merged_df.copy()

        print(
            "\nUsing ALL labeled nodes "
            f"(N={len(df)})"
        )

        return df

    elif mode == "train":

        df = merged_df[
            merged_df["node"].isin(train_nodes)
        ].copy()

        print(
            "\nUsing TRAIN nodes only "
            f"(N={len(df)})"
        )

        return df

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

# Validation Checks

def validate_dataframe(df):
    """Ensures all frozen features exist."""

    missing = [
        feat
        for feat in FEATURES
        if feat not in df.columns
    ]

    if len(missing) > 0:

        raise KeyError(
            "Missing features detected:\n"
            f"{missing}"
        )

    print("\nFeature validation passed.")

    print(
        f"Features analyzed ({len(FEATURES)}):"
    )

    for feat in FEATURES:
        print(" -", feat)

# Spearman Correlation Analysis

def compute_correlations(
    df,
    label_column
):

    results = []

    print(
        f"\nComputing correlations "
        f"against {label_column}..."
    )

    for feat in FEATURES:

        rho, p = spearmanr(
            df[feat],
            df[label_column]
        )

        # scipy returns nan for constant vectors
        if pd.isna(rho):
            rho = 0.0

        if pd.isna(p):
            p = 1.0

        results.append(
            {
                "feature": feat,
                "rho": float(rho),
                "p_value": float(p)
            }
        )

    results_df = pd.DataFrame(results)

    return results_df

# Multiple Testing Correction

def apply_fdr_correction(
    results_df
):

    corrected = results_df.copy()

    p_adj = false_discovery_control(
    corrected["p_value"].values,
    method="bh")

    reject = p_adj <= ALPHA

    corrected["p_fdr"] = p_adj
    corrected["significant"] = reject

    return corrected

# Feature Rankings

def rank_features(
    results_df
):

    ranked = (
        results_df
        .copy()
        .sort_values(
            by="rho",
            key=lambda x: np.abs(x),
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranked["rank"] = (
        np.arange(len(ranked)) + 1
    )

    cols = [
        "rank",
        "feature",
        "rho",
        "p_value",
        "p_fdr",
        "significant"
    ]

    return ranked[cols]

# Console Reporting

def print_ranking(
    ranked_df,
    title
):

    print(title)
    for _, row in ranked_df.iterrows():

        sig = "YES" if row["significant"] else "NO"

        print(
            f"#{int(row['rank']):<2d} "
            f"{row['feature']:<18s} "
            f"rho={row['rho']:+.4f}   "
            f"p={row['p_value']:.4e}   "
            f"FDR={row['p_fdr']:.4e}   "
            f"sig={sig}"
        )

# Export Results

def save_results(
    ranked_df,
    filename
):
    # Save ranking tables

    output_path = RESULTS_DIR / filename

    ranked_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved: {output_path}"
    )

# Complete Pipeline For One Label

def analyze_label(
    df,
    label_column,
    output_filename,
    title
):

    results = compute_correlations(
        df,
        label_column
    )

    corrected = apply_fdr_correction(
        results
    )

    ranked = rank_features(
        corrected
    )

    print_ranking(
        ranked,
        title
    )

    save_results(
        ranked,
        output_filename
    )

    return ranked

# Ic–Lt Feature Rank Stability

def compute_rank_stability(
    ic_ranked,
    lt_ranked
):

    ic_order = (
        ic_ranked
        .set_index("feature")["rank"]
        .sort_index()
    )

    lt_order = (
        lt_ranked
        .set_index("feature")["rank"]
        .sort_index()
    )

    rho, p = spearmanr(
        ic_order.values,
        lt_order.values
    )

    return float(rho), float(p)

# Summary Report

def append_summary(
    lines,
    section_title,
    ic_ranked,
    lt_ranked,
    stability_rho,
    stability_p
):

    lines.append(section_title)

    lines.append("")

    lines.append("IC Top Features:")

    for _, row in ic_ranked.head(5).iterrows():

        lines.append(
            f"  #{int(row['rank'])} "
            f"{row['feature']} "
            f"(rho={row['rho']:+.4f})"
        )

    lines.append("")

    lines.append("LT Top Features:")

    for _, row in lt_ranked.head(5).iterrows():

        lines.append(
            f"  #{int(row['rank'])} "
            f"{row['feature']} "
            f"(rho={row['rho']:+.4f})"
        )

    lines.append("")

    lines.append(
        "IC-LT Rank Stability:"
    )

    lines.append(
        f"  rho = {stability_rho:.4f}"
    )

    lines.append(
        f"  p   = {stability_p:.4e}"
    )

    lines.append("")
    lines.append("")

# Main Analysis Pipeline

def run_analysis(
    merged,
    train_nodes,
    mode,
    summary_lines
):

    df = get_dataset(
        merged,
        train_nodes,
        mode=mode
    )

    validate_dataframe(df)

    # Ic

    ic_ranked = analyze_label(
        df=df,
        label_column="ic_mean",
        output_filename=f"{mode}_nodes_ic.csv",
        title=(
            f"{mode.upper()} "
            "NODES — IC FEATURE RANKING"
        )
    )

    # Lt

    lt_ranked = analyze_label(
        df=df,
        label_column="lt_mean",
        output_filename=f"{mode}_nodes_lt.csv",
        title=(
            f"{mode.upper()} "
            "NODES — LT FEATURE RANKING"
        )
    )

    # Rank Stability

    rho, p = compute_rank_stability(
        ic_ranked,
        lt_ranked
    )

    print(
        f"{mode.upper()} IC-LT "
        "RANK STABILITY"
    )

    print(
        f"Spearman rho = {rho:.4f}"
    )

    print(
        f"p-value      = {p:.4e}"
    )

    append_summary(
        lines=summary_lines,
        section_title=(
            f"{mode.upper()} NODES "
            f"(N={len(df)})"
        ),
        ic_ranked=ic_ranked,
        lt_ranked=lt_ranked,
        stability_rho=rho,
        stability_p=p
    )

# Entry Point

def main():

    print(
        "FEATURE-INFLUENCE "
        "CORRELATION ANALYSIS"
    )
    merged, train_nodes = load_data()

    summary_lines = []

    # All labeled nodes (paper)

    run_analysis(
        merged=merged,
        train_nodes=train_nodes,
        mode="all",
        summary_lines=summary_lines
    )

    # Train only (appendix)

    run_analysis(
        merged=merged,
        train_nodes=train_nodes,
        mode="train",
        summary_lines=summary_lines
    )

    # Save summary

    summary_path = (
        RESULTS_DIR /
        "summary_report.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(summary_lines)
        )

    print("\nSaved:", summary_path)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()