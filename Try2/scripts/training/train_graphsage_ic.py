# ============================================================
# train_graphsage_ic.py
# Try2 Vanilla Benchmark
# Chunk 1: Imports, Reproducibility, Configuration,
#           Dataset Loading
# ============================================================

import os
import copy
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, kendalltau

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    ndcg_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PyG Imports
# ============================================================

from torch_geometric.nn import SAGEConv




# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "Try2/data/higgs_pyg.pt"

CHECKPOINT_PATH = (
    "Try2/checkpoints/"
    "best_graphsage_ic_vanilla.pt"
)

RESULTS_PATH = (
    "Try2/results/ic/"
    "graphsage_ic_vanilla_results.txt"
)

PREDICTIONS_PATH = (
    "Try2/results/ic/"
    "graphsage_ic_vanilla_predictions.csv"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Frozen Vanilla Benchmark Hyperparameters
# ============================================================

HIDDEN_DIM = 32
DROPOUT = 0.1

LR = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 500
PATIENCE = 100

GRAD_CLIP = 1.0


# ============================================================
# Create Output Directories
# ============================================================

os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(PREDICTIONS_PATH),
    exist_ok=True,
)


# ============================================================
# Dataset Loading
# ============================================================

print("=" * 60)
print("GraphSAGE-IC Vanilla Benchmark")
print("=" * 60)

print("\nLoading Frozen Dataset...")
print("-" * 60)

data = torch.load(
    DATA_PATH,
    weights_only=False,
)

print(data)



def precision_at_k(
    y_true,
    y_pred,
    k=10,
):
    """
    Precision@K based on Top-K overlap.
    """

    k = min(k, len(y_true))

    if np.var(y_pred) < 1e-12:
        return 0.0

    true_topk = np.argsort(y_true)[-k:]
    pred_topk = np.argsort(y_pred)[-k:]

    overlap = len(
        set(true_topk).intersection(
            set(pred_topk)
        )
    )

    return overlap / k


def recall_at_k(
    y_true,
    y_pred,
    k=10,
):
    """
    Recall@K.
    """

    k = min(k, len(y_true))

    if np.var(y_pred) < 1e-12:
        return 0.0

    true_topk = np.argsort(y_true)[-k:]
    pred_topk = np.argsort(y_pred)[-k:]

    overlap = len(
        set(true_topk).intersection(
            set(pred_topk)
        )
    )

    return overlap / k


def ndcg_at_k(
    y_true,
    y_pred,
    k=10,
):
    """
    NDCG@K.
    """

    k = min(k, len(y_true))

    if np.var(y_pred) < 1e-12:
        return 0.0

    return ndcg_score(
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        k=k,
    )


def topk_overlap(
    y_true,
    y_pred,
    k=10,
):
    """
    Absolute Top-K overlap count.
    """

    k = min(k, len(y_true))

    if np.var(y_pred) < 1e-12:
        return 0

    true_topk = np.argsort(y_true)[-k:]
    pred_topk = np.argsort(y_pred)[-k:]

    return len(
        set(true_topk).intersection(
            set(pred_topk)
        )
    )


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
):
    """
    Publication-grade metrics.

    Regression:
        MAE
        RMSE
        R²

    Rank Correlation:
        Spearman
        Kendall Tau

    Retrieval:
        Precision@10
        Precision@20
        Recall@10
        Recall@20
        NDCG@10
        NDCG@20

    Influence Recovery:
        Top10Overlap
        Top20Overlap
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    spearman_corr, _ = spearmanr(
        y_true,
        y_pred,
    )

    kendall_corr, _ = kendalltau(
        y_true,
        y_pred,
    )

    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    if np.isnan(kendall_corr):
        kendall_corr = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": spearman_corr,
        "KendallTau": kendall_corr,

        "Precision@10": precision_at_k(
            y_true,
            y_pred,
            10,
        ),

        "Precision@20": precision_at_k(
            y_true,
            y_pred,
            20,
        ),

        "Recall@10": recall_at_k(
            y_true,
            y_pred,
            10,
        ),

        "Recall@20": recall_at_k(
            y_true,
            y_pred,
            20,
        ),

        "NDCG@10": ndcg_at_k(
            y_true,
            y_pred,
            10,
        ),

        "NDCG@20": ndcg_at_k(
            y_true,
            y_pred,
            20,
        ),

        "Top10Overlap": topk_overlap(
            y_true,
            y_pred,
            10,
        ),

        "Top20Overlap": topk_overlap(
            y_true,
            y_pred,
            20,
        ),
    }

def elite_diagnostics(
    y_true,
    y_pred,
):
    """
    Elite influencer diagnostics.

    Returns
    -------
    dict
    """

    diagnostics = {}

    # ----------------------------
    # Top-1 MAE
    # ----------------------------

    idx = np.argsort(y_true)[-1:]

    diagnostics["Top1_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    # ----------------------------
    # Top-5 MAE
    # ----------------------------

    idx = np.argsort(y_true)[-5:]

    diagnostics["Top5_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    # ----------------------------
    # Top-10 MAE
    # ----------------------------

    idx = np.argsort(y_true)[::-1][:10]

    diagnostics["Top10_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    # ----------------------------
    # Top-1 MSE
    # ----------------------------

    idx = np.argsort(y_true)[-1:]

    diagnostics["Top1_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    # ----------------------------
    # Top-5 MSE
    # ----------------------------

    idx = np.argsort(y_true)[-5:]

    diagnostics["Top5_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    # ----------------------------
    # Top-10 MSE
    # ----------------------------

    idx = np.argsort(y_true)[::-1][:10]

    diagnostics["Top10_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    # ----------------------------
    # 95th Percentile MAE
    # ----------------------------

    threshold = np.percentile(
        y_true,
        95,
    )

    elite_mask = (
        y_true >= threshold
    )

    diagnostics["P95_MAE"] = np.mean(
        np.abs(
            y_true[elite_mask]
            - y_pred[elite_mask]
        )
    )

    # ----------------------------
    # 95th Percentile MSE
    # ----------------------------

    diagnostics["P95_MSE"] = np.mean(
        (
            y_true[elite_mask]
            - y_pred[elite_mask]
        ) ** 2
    )

    return diagnostics

# ============================================================
# Dataset Summary
# ============================================================

print("\nDataset Summary")
print("-" * 40)

print(
    f"Nodes          : {data.num_nodes:,}"
)

print(
    f"Edges          : "
    f"{data.edge_index.shape[1]:,}"
)

print(
    f"Features       : "
    f"{data.x.shape[1]}"
)

print(
    f"Edge Attr Dim  : "
    f"{data.edge_attr.shape[1]}"
)


# ============================================================
# Frozen Split Verification
# ============================================================

train_count = int(
    data.train_mask.sum().item()
)

val_count = int(
    data.val_mask.sum().item()
)

test_count = int(
    data.test_mask.sum().item()
)

print("\nFrozen Split Verification")
print("-" * 40)

print(f"Train Nodes : {train_count}")
print(f"Val Nodes   : {val_count}")
print(f"Test Nodes  : {test_count}")


assert train_count == 700, (
    f"Expected 700 train nodes, "
    f"found {train_count}"
)

assert val_count == 148, (
    f"Expected 148 val nodes, "
    f"found {val_count}"
)

assert test_count == 152, (
    f"Expected 152 test nodes, "
    f"found {test_count}"
)

print("\n✓ Frozen splits verified.")


# ============================================================
# Feature Verification
# ============================================================

print("\nFeature Verification")
print("-" * 40)

print(
    f"x shape : {tuple(data.x.shape)}"
)

assert data.x.shape == (5000, 9), (
    f"Expected x=(5000,9), "
    f"found {tuple(data.x.shape)}"
)

print("✓ Frozen feature matrix verified.")


# ============================================================
# IC Label Statistics
# ============================================================

print("\nIC Label Statistics")
print("-" * 40)

num_nan = torch.isnan(
    data.y_ic
).sum().item()

print(f"NaN Labels     : {num_nan}")

print(
    f"Labeled Nodes  : "
    f"{data.num_nodes - num_nan}"
)

print(
    f"Mean Spread    : "
    f"{data.y_ic.mean().item():.4f}"
)

print(
    f"Max Spread     : "
    f"{data.y_ic.max().item():.4f}"
)


# ============================================================
# Move Dataset to Device
# ============================================================

data = data.to(DEVICE)

print("\nUsing Device :", DEVICE)

print("\nDataset Ready.")
print("=" * 60)

# ============================================================
# Chunk 2: GraphSAGE Model, Loss, Optimizer,
#           Scheduler, Training Setup
# ============================================================

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# Vanilla GraphSAGE Model Definition
# ============================================================

class GraphSAGE(nn.Module):
    """
    Vanilla GraphSAGE for IC Influence Estimation

    Frozen Vanilla Architecture:

        Input (9)
            ↓
        SAGEConv(9 → 32)
            ↓
        LayerNorm(32)
            ↓
        ReLU
            ↓
        Dropout(0.1)
            ↓
        SAGEConv(32 → 1)
            ↓
        Output

    Notes
    -----
    • Authentic GraphSAGE preserved.
    • No decoder.
    • No bypass.
    • No residual projection.
    • No concatenation.
    • LayerNorm treated as shared stabilization.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Hidden GraphSAGE Layer
        # ----------------------------------------------------

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        self.conv2 = SAGEConv(
            hidden_channels,
            1,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):
        """
        Parameters
        ----------
        x : Tensor
            Node features [N,9]

        edge_index : Tensor
            Graph connectivity

        Returns
        -------
        Tensor
            Predictions [N]
        """

        # ----------------------------------------------------
        # Hidden Layer
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index,
        )

        x = self.norm1(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        x = self.conv2(
            x,
            edge_index,
        )

        return x.squeeze(-1)


# ============================================================
# Initialize Model
# ============================================================

model = GraphSAGE(
    in_channels=data.x.shape[1],
    hidden_channels=HIDDEN_DIM,
    dropout=DROPOUT,
).to(DEVICE)


# ============================================================
# Model Summary
# ============================================================

print("\nModel Architecture")
print("-" * 40)

print(model)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"\nTotal Parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable Parameters : "
    f"{trainable_params:,}"
)


# ============================================================
# Frozen Vanilla Loss Function
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# Frozen Vanilla Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Frozen Vanilla Scheduler
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=20,
    min_lr=1e-5,
)


# ============================================================
# Early Stopping Variables
# ============================================================

best_val_mae = float("inf")

best_epoch = -1

best_state_dict = None

patience_counter = 0


# ============================================================
# Training History
# ============================================================

history = {
    "train_loss": [],
    "val_mae": [],
    "learning_rate": [],
}


# ============================================================
# Frozen Benchmark Configuration Summary
# ============================================================

print("\nTraining Configuration")
print("-" * 40)

print("Loss              : MSELoss")

print("Optimizer         : AdamW")

print(f"Learning Rate     : {LR}")

print(
    f"Weight Decay      : "
    f"{WEIGHT_DECAY}"
)

print(
    "Scheduler         : "
    "ReduceLROnPlateau"
)

print("  Factor          : 0.5")

print("  Patience        : 20")

print("  Min LR          : 1e-5")

print(
    f"Gradient Clip     : "
    f"{GRAD_CLIP}"
)

print(
    f"Epochs            : "
    f"{EPOCHS}"
)

print(
    f"Early Stopping    : "
    f"{PATIENCE}"
)

print(
    f"Hidden Dimension  : "
    f"{HIDDEN_DIM}"
)

print(
    f"Dropout           : "
    f"{DROPOUT}"
)

print("\nLeakage Policy")
print("-" * 40)

print("✓ Best checkpoint selected using validation MAE")

print("✓ Test set never used during training")

print("✓ Frozen train/val/test splits")

print("✓ Final evaluation performed after restoring")
print("  the best validation checkpoint")

print("\nModel Setup Complete.")
print("=" * 60)

# ============================================================
# Chunk 3: Training Loop, Validation,
#           Scheduler, Early Stopping
# ============================================================

print("\n" + "=" * 60)
print("Starting Vanilla GraphSAGE Training (IC)")
print("=" * 60)


# ============================================================
# Training Function
# ============================================================

def train_one_epoch():
    """
    One full-batch training epoch.

    Returns
    -------
    float
        Training MSE loss.
    """

    model.train()

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------

    predictions = model(
        data.x,
        data.edge_index,
    )

    # --------------------------------------------------------
    # Train Loss
    # (Frozen train split only)
    # --------------------------------------------------------

    train_pred = predictions[
        data.train_mask
    ]

    train_true = data.y_ic[
        data.train_mask
    ]

    loss = criterion(
        train_pred,
        train_true,
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Frozen Gradient Clipping
    # --------------------------------------------------------

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    optimizer.step()

    return loss.item()


# ============================================================
# Validation Function
# ============================================================

@torch.no_grad()
def validate():
    """
    Compute validation MAE.

    Returns
    -------
    float
        Validation MAE.
    """

    model.eval()

    predictions = model(
        data.x,
        data.edge_index,
    )

    val_pred = predictions[
        data.val_mask
    ]

    val_true = data.y_ic[
        data.val_mask
    ]

    val_mae = mean_absolute_error(
        val_true.cpu().numpy(),
        val_pred.cpu().numpy(),
    )

    return val_mae


# ============================================================
# Main Training Loop
# ============================================================

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss = train_one_epoch()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_mae = validate()

    # --------------------------------------------------------
    # Scheduler Update
    # --------------------------------------------------------

    scheduler.step(val_mae)

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # Save History
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_mae"].append(
        val_mae
    )

    history["learning_rate"].append(
        current_lr
    )

    # --------------------------------------------------------
    # Best Checkpoint Selection
    #
    # FROZEN RULE:
    # Selection based ONLY on validation MAE
    # --------------------------------------------------------

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch

        best_state_dict = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state_dict,
            CHECKPOINT_PATH,
        )

        patience_counter = 0

    else:

        patience_counter += 1

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Best Val: {best_val_mae:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    # --------------------------------------------------------
    # Early Stopping
    #
    # FROZEN RULE:
    # Patience = 100
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        print(
            f"No validation improvement "
            f"for {PATIENCE} epochs."
        )

        break


# ============================================================
# Training Complete
# ============================================================

print("\n" + "=" * 60)
print("Training Complete")
print("=" * 60)

print(
    f"Best Epoch           : "
    f"{best_epoch}"
)

print(
    f"Best Validation MAE  : "
    f"{best_val_mae:.4f}"
)

print(
    f"Final Learning Rate  : "
    f"{optimizer.param_groups[0]['lr']:.6f}"
)

print(
    f"\nCheckpoint Saved To:\n"
    f"{CHECKPOINT_PATH}"
)


# ============================================================
# Restore Best Checkpoint
# ============================================================

print("\nRestoring Best Validation Checkpoint...")

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

print("✓ Best checkpoint restored.")

print("=" * 60)


# ============================================================
# Leakage Audit
# ============================================================

print("\nLeakage Audit")
print("-" * 40)

print(
    "✓ Train loss computed only on train_mask"
)

print(
    "✓ Validation MAE computed only on val_mask"
)

print(
    "✓ Scheduler updated using validation MAE"
)

print(
    "✓ Best checkpoint selected using validation MAE"
)

print(
    "✓ Test labels never accessed during training"
)

print(
    "✓ Final test evaluation will use restored "
    "best checkpoint"
)

print("=" * 60)

# ============================================================
# Chunk 4: Final Evaluation, Elite Diagnostics,
#           Prediction Export, Results Summary
# ============================================================

print("\n" + "=" * 60)
print("Final Evaluation on Test Set")
print("=" * 60)


# ============================================================
# Test Evaluation
# ============================================================

@torch.no_grad()
def evaluate_test():
    """
    Evaluate restored best checkpoint on the
    frozen test split.

    Returns
    -------
    metrics : dict
    elite_metrics : dict
    y_true : ndarray
    y_pred : ndarray
    node_ids : ndarray
    """

    model.eval()

    predictions = model(
        data.x,
        data.edge_index,
    )

    test_pred = predictions[
        data.test_mask
    ]

    test_true = data.y_ic[
        data.test_mask
    ]

    test_node_ids = data.node_ids[
        data.test_mask
    ]

    y_true = (
        test_true.detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        test_pred.detach()
        .cpu()
        .numpy()
    )

    node_ids = (
        test_node_ids.detach()
        .cpu()
        .numpy()
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    elite_metrics = elite_diagnostics(
    y_true,
    y_pred,
)

    return (
        metrics,
        elite_metrics,
        y_true,
        y_pred,
        node_ids,
    )


# ============================================================
# Run Evaluation
# ============================================================

(
    test_metrics,
    elite_metrics,
    y_true_test,
    y_pred_test,
    test_node_ids,
) = evaluate_test()


# ============================================================
# Print Regression Metrics
# ============================================================

print("\nRegression Metrics")
print("-" * 40)

regression_keys = [
    "MAE",
    "RMSE",
    "R2",
    "Spearman",
    "KendallTau",
]

for key in regression_keys:

    if key in test_metrics:

        print(
            f"{key:<15}: "
            f"{test_metrics[key]:.4f}"
        )


# ============================================================
# Print Ranking Metrics
# ============================================================

print("\nRanking Metrics")
print("-" * 40)

ranking_keys = [
    "Precision@10",
    "Precision@20",
    "Recall@10",
    "Recall@20",
    "NDCG@10",
    "NDCG@20",
    "Top10Overlap",
    "Top20Overlap",
]

for key in ranking_keys:

    if key in test_metrics:

        print(
            f"{key:<15}: "
            f"{test_metrics[key]:.4f}"
        )


# ============================================================
# Elite Diagnostics
# ============================================================

print("\nElite Diagnostics")
print("-" * 40)

elite_keys = [
    "Top1_MAE",
    "Top5_MAE",
    "Top10_MAE",
    "Top1_MSE",
    "Top5_MSE",
    "Top10_MSE",
    "P95_MAE",
    "P95_MSE",
]

for key in elite_keys:

    if key in elite_metrics:

        print(
            f"{key:<15}: "
            f"{elite_metrics[key]:.4f}"
        )


# ============================================================
# Top-10 Analysis
# ============================================================

print("\nTop-10 Analysis")
print("-" * 40)

true_top10 = np.argsort(
    -y_true_test
)[:10]

pred_top10 = np.argsort(
    -y_pred_test
)[:10]

overlap = set(
    true_top10
).intersection(
    set(pred_top10)
)

print(
    f"Top-10 Overlap : "
    f"{len(overlap)}/10"
)

print("\nGround Truth Top-10")

for rank, idx in enumerate(
    true_top10,
    start=1,
):

    print(
        f"{rank:2d}. "
        f"Node={test_node_ids[idx]} | "
        f"Spread={y_true_test[idx]:.4f}"
    )


print("\nPredicted Top-10")

for rank, idx in enumerate(
    pred_top10,
    start=1,
):

    print(
        f"{rank:2d}. "
        f"Node={test_node_ids[idx]} | "
        f"Pred={y_pred_test[idx]:.4f}"
    )


missed = sorted(
    set(true_top10) - set(pred_top10)
)

recovered = sorted(
    set(pred_top10) - set(true_top10)
)

print(
    f"\nMissed Elites   : "
    f"{len(missed)}"
)

print(
    f"Recovered Elite : "
    f"{len(recovered)}"
)


# ============================================================
# Prediction Export
# ============================================================

print("\nSaving Predictions...")


prediction_df = pd.DataFrame({
    "node_id": test_node_ids,
    "y_true": y_true_test,
    "y_pred": y_pred_test,
    "split": "test",
    "true_rank": (-y_true_test).argsort().argsort() + 1,
    "pred_rank": (-y_pred_test).argsort().argsort() + 1,
})

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print(
    f"Predictions saved to:\n"
    f"{PREDICTIONS_PATH}"
)


# ============================================================
# Experiment Summary
# ============================================================

print("\n" + "=" * 60)
print("VANILLA GRAPHSAGE IC SUMMARY")
print("=" * 60)

print("Model             : GraphSAGE")
print("Task              : IC")
print("Benchmark Track   : Vanilla")
print("Target            : y_ic")

print(
    f"Hidden Dimension  : "
    f"{HIDDEN_DIM}"
)

print(
    f"Dropout           : "
    f"{DROPOUT}"
)

print("Loss              : MSELoss")

print("Optimizer         : AdamW")

print(
    f"Learning Rate     : "
    f"{LR}"
)

print(
    f"Weight Decay      : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Gradient Clip     : "
    f"{GRAD_CLIP}"
)

print(
    f"Epochs            : "
    f"{EPOCHS}"
)

print(
    f"Patience          : "
    f"{PATIENCE}"
)

print(
    f"Best Epoch        : "
    f"{best_epoch}"
)

print(
    f"Best Val MAE      : "
    f"{best_val_mae:.4f}"
)

print("\nTest Performance")
print("-" * 40)

for key in regression_keys:

    if key in test_metrics:

        print(
            f"{key:<15}: "
            f"{test_metrics[key]:.4f}"
        )


# ============================================================
# Save Results File
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "VANILLA GRAPHSAGE IC RESULTS\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best Validation MAE: "
        f"{best_val_mae:.6f}\n\n"
    )

    f.write(
        "Regression Metrics\n"
    )

    for key in regression_keys:

        if key in test_metrics:

            f.write(
                f"{key}: "
                f"{test_metrics[key]:.6f}\n"
            )

    f.write("\nRanking Metrics\n")

    for key in ranking_keys:

        if key in test_metrics:

            f.write(
                f"{key}: "
                f"{test_metrics[key]:.6f}\n"
            )

    f.write("\nElite Diagnostics\n")

    for key in elite_keys:

        if key in elite_metrics:

            f.write(
                f"{key}: "
                f"{elite_metrics[key]:.6f}\n"
            )


print(
    f"\nResults saved to:\n"
    f"{RESULTS_PATH}"
)


print("\n" + "=" * 60)
print("Experiment Finished Successfully")
print("=" * 60)

