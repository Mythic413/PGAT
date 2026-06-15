import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score,
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import GINConv

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

SEED = 42

DATA_PATH = "Try2/data/higgs_pyg.pt"

CHECKPOINT_PATH = "Try2/checkpoints/best_gin_ic.pt"

RESULTS_PATH = "Try2/results/gin_ic_results.txt"

PREDICTIONS_PATH = "Try2/results/gin_ic_predictions.csv"

TARGET = "y_ic"

HIDDEN_DIM = 32
DROPOUT = 0.1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 500
PATIENCE = 100

GRAD_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# ============================================================
# Create directories
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
# Benchmark Header
# ============================================================

print("=" * 60)
print("GIN-IC Publication Benchmark")
print("=" * 60)

print(f"Seed              : {SEED}")
print(f"Device            : {DEVICE}")

print("\nArchitecture")
print("-" * 40)
print(f"Hidden Dimension  : {HIDDEN_DIM}")
print(f"Dropout           : {DROPOUT}")

print("\nOptimization")
print("-" * 40)
print("Loss              : MSELoss")
print("Optimizer         : AdamW")
print("Scheduler         : ReduceLROnPlateau")
print(f"Learning Rate     : {LEARNING_RATE}")
print(f"Weight Decay      : {WEIGHT_DECAY}")
print(f"Gradient Clip     : {GRAD_CLIP}")

print("\nTraining")
print("-" * 40)
print(f"Epochs            : {EPOCHS}")
print(f"Patience          : {PATIENCE}")

print("=" * 60)

# ============================================================
# Load Dataset
# ============================================================

print("\n" + "=" * 60)
print("Loading PyG Dataset...")
print("=" * 60)

data = torch.load(
    DATA_PATH,
    map_location="cpu",
    weights_only=False,
)

print(data)

print("\nDataset Summary")
print("-" * 40)

print(f"Nodes        : {data.num_nodes}")
print(f"Edges        : {data.edge_index.size(1)}")
print(f"Features     : {data.num_node_features}")

print(
    f"Train Nodes  : {int(data.train_mask.sum())}"
)

print(
    f"Val Nodes    : {int(data.val_mask.sum())}"
)

print(
    f"Test Nodes   : {int(data.test_mask.sum())}"
)

# ============================================================
# Label Statistics
# ============================================================

y = getattr(data, TARGET)

labeled_mask = ~torch.isnan(y)

print("\nIC Label Statistics")
print("-" * 40)

print(
    f"NaN Labels       : {int((~labeled_mask).sum())}"
)

print(
    f"Labeled Nodes    : {int(labeled_mask.sum())}"
)

print(
    f"IC Min Spread    : "
    f"{y[labeled_mask].min().item():.4f}"
)

print(
    f"IC Mean Spread   : "
    f"{y[labeled_mask].mean().item():.4f}"
)

print(
    f"IC Max Spread    : "
    f"{y[labeled_mask].max().item():.4f}"
)

data = data.to(DEVICE)

print(f"\nUsing Device: {DEVICE}")

print("=" * 60)

# ============================================================
# Metric Utilities
# ============================================================

def precision_at_k(
    y_true,
    y_pred,
    k,
):
    true_idx = np.argsort(y_true)[-k:]

    pred_idx = np.argsort(y_pred)[-k:]

    return len(
        set(true_idx) & set(pred_idx)
    ) / k


def recall_at_k(
    y_true,
    y_pred,
    k,
):
    true_idx = np.argsort(y_true)[-k:]

    pred_idx = np.argsort(y_pred)[-k:]

    return len(
        set(true_idx) & set(pred_idx)
    ) / k


def topk_overlap(
    y_true,
    y_pred,
    k,
):
    true_idx = np.argsort(y_true)[-k:]

    pred_idx = np.argsort(y_pred)[-k:]

    return len(
        set(true_idx) & set(pred_idx)
    )


def elite_metrics(
    y_true,
    y_pred,
):
    metrics = {}

    descending = np.argsort(y_true)[::-1]

    for k in [1, 5, 10]:
        idx = descending[:k]

        metrics[f"Top{k}_MAE"] = mean_absolute_error(
            y_true[idx],
            y_pred[idx],
        )

        metrics[f"Top{k}_MSE"] = mean_squared_error(
            y_true[idx],
            y_pred[idx],
        )

    threshold = np.percentile(
        y_true,
        95,
    )

    idx = y_true >= threshold

    metrics["P95_MAE"] = mean_absolute_error(
        y_true[idx],
        y_pred[idx],
    )

    metrics["P95_MSE"] = mean_squared_error(
        y_true[idx],
        y_pred[idx],
    )

    return metrics

# ============================================================
# GIN Model
# ============================================================

class GIN(nn.Module):
    """
    Vanilla GIN for IC Influence Estimation

    Architecture
    ------------
    Input Features
    ↓
    GINConv(
        Linear → ReLU → Linear
    )
    ↓
    LayerNorm(32)
    ↓
    ReLU
    ↓
    Dropout
    ↓
    GINConv(
        Linear → ReLU → Linear
    )
    ↓
    Node Influence Prediction
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout=0.1,
    ):
        super().__init__()

        self.dropout = dropout

        # ----------------------------------------------------
        # First GIN layer
        # ----------------------------------------------------
        mlp1 = nn.Sequential(
            nn.Linear(
                in_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
        )

        self.conv1 = GINConv(
            mlp1,
            train_eps=True,
        )

        # Shared stabilization
        self.norm1 = nn.LayerNorm(
            hidden_channels,
        )

        # ----------------------------------------------------
        # Second GIN layer
        # ----------------------------------------------------
        mlp2 = nn.Sequential(
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                1,
            ),
        )

        self.conv2 = GINConv(
            mlp2,
            train_eps=True,
        )

        self.reset_parameters()

    def reset_parameters(self):

        self.conv1.reset_parameters()

        self.conv2.reset_parameters()

        self.norm1.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
    ):

        # --------------------------------------------
        # GIN Block 1
        # --------------------------------------------
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

        # --------------------------------------------
        # GIN Block 2
        # --------------------------------------------
        x = self.conv2(
            x,
            edge_index,
        )

        return x.squeeze(-1)


# ============================================================
# Model Initialization
# ============================================================

model = GIN(
    in_channels=data.x.shape[1],
    hidden_channels=HIDDEN_DIM,
    dropout=DROPOUT,
).to(DEVICE)

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
    f"Total Parameters     : "
    f"{total_params}"
)

print(
    f"Trainable Parameters : "
    f"{trainable_params}"
)

print("=" * 60)
print("Dataset + Model Ready")
print("=" * 60)


# ============================================================
# Optimization Components
# ============================================================

loss_fn = nn.MSELoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=20,
    min_lr=1e-5,
)

print("\nOptimization Components")
print("-" * 40)

print("Loss Function : MSELoss")
print("Optimizer     : AdamW")
print("Scheduler     : ReduceLROnPlateau")

print("Factor         : 0.5")
print("Patience       : 20")
print("Minimum LR     : 1e-05")

print("\nTraining Configuration")
print("-" * 40)

print(
    f"Learning Rate   : "
    f"{LEARNING_RATE}"
)

print(
    f"Weight Decay    : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Epoch Budget    : "
    f"{EPOCHS}"
)

print(
    f"Early Stopping  : "
    f"{PATIENCE}"
)

print(
    f"Gradient Clip   : "
    f"{GRAD_CLIP}"
)

print(
    f"Checkpoint Path : "
    f"{CHECKPOINT_PATH}"
)

print("=" * 60)
print("Optimization Setup Complete")
print("=" * 60)


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 60)
print("Starting GIN Training on IC Labels...")
print("=" * 60)

best_val_mae = float("inf")
best_epoch = 0
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_mae": [],
}

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    model.train()

    optimizer.zero_grad()

    predictions = model(
        data.x,
        data.edge_index,
    )

    train_predictions = predictions[
        data.train_mask
    ]

    train_targets = data.y_ic[
        data.train_mask
    ]

    loss = loss_fn(
        train_predictions,
        train_targets,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRAD_CLIP,
    )

    optimizer.step()

    train_mse = loss.item()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    model.eval()

    with torch.no_grad():

        val_predictions_all = model(
            data.x,
            data.edge_index,
        )

        val_predictions = val_predictions_all[
            data.val_mask
        ]

        val_targets = data.y_ic[
            data.val_mask
        ]

        val_mae = mean_absolute_error(
            val_targets.cpu().numpy(),
            val_predictions.cpu().numpy(),
        )

        val_spearman, _ = spearmanr(
            val_targets.cpu().numpy(),
            val_predictions.cpu().numpy(),
        )

        val_precision10 = precision_at_k(
            val_targets.cpu().numpy(),
            val_predictions.cpu().numpy(),
            10,
        )

        val_ndcg10 = ndcg_score(
            [val_targets.cpu().numpy()],
            [val_predictions.cpu().numpy()],
            k=10,
        )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------
    scheduler.step(val_mae)

    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------
    history["train_loss"].append(
        train_mse
    )

    history["val_mae"].append(
        val_mae
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------
    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "val_mae": val_mae,
            },
            CHECKPOINT_PATH,
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Best Val MAE: {best_val_mae:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        print(
            f"    Spearman: "
            f"{val_spearman:.4f} | "
            f"Precision@10: "
            f"{val_precision10:.4f} | "
            f"NDCG@10: "
            f"{val_ndcg10:.4f}"
        )

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------
    if epochs_without_improvement >= PATIENCE:

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
    f"Best Epoch         : "
    f"{best_epoch}"
)

print(
    f"Best Validation MAE: "
    f"{best_val_mae:.4f}"
)

print(
    f"Final Learning Rate: "
    f"{optimizer.param_groups[0]['lr']:.6f}"
)

print("\nBest checkpoint saved to:")

print(CHECKPOINT_PATH)


# ============================================================
# Restore Best Checkpoint
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Restored epoch      : "
    f"{checkpoint['epoch']}"
)

print(
    f"Restored Val MAE    : "
    f"{checkpoint['val_mae']:.4f}"
)

print("Checkpoint restored.")

print("=" * 60)

# ============================================================
# Final Evaluation on Test Set
# ============================================================

print("\n" + "=" * 60)
print("Final Evaluation on Test Set")
print("=" * 60)

model.eval()

with torch.no_grad():

    predictions = model(
        data.x,
        data.edge_index,
    )

test_predictions = predictions[
    data.test_mask
]

test_targets = data.y_ic[
    data.test_mask
]

y_true_test = (
    test_targets.cpu().numpy()
)

y_pred_test = (
    test_predictions.cpu().numpy()
)

test_node_ids = (
    data.node_ids[
        data.test_mask
    ]
    .cpu()
    .numpy()
)

# ============================================================
# Regression Metrics
# ============================================================

mae = mean_absolute_error(
    y_true_test,
    y_pred_test,
)

rmse = np.sqrt(
    mean_squared_error(
        y_true_test,
        y_pred_test,
    )
)

r2 = r2_score(
    y_true_test,
    y_pred_test,
)

spearman, _ = spearmanr(
    y_true_test,
    y_pred_test,
)

kendall, _ = kendalltau(
    y_true_test,
    y_pred_test,
)

# ============================================================
# Ranking Metrics
# ============================================================

precision10 = precision_at_k(
    y_true_test,
    y_pred_test,
    10,
)

precision20 = precision_at_k(
    y_true_test,
    y_pred_test,
    20,
)

recall10 = recall_at_k(
    y_true_test,
    y_pred_test,
    10,
)

recall20 = recall_at_k(
    y_true_test,
    y_pred_test,
    20,
)

ndcg10 = ndcg_score(
    [y_true_test],
    [y_pred_test],
    k=10,
)

ndcg20 = ndcg_score(
    [y_true_test],
    [y_pred_test],
    k=20,
)

top10_overlap = topk_overlap(
    y_true_test,
    y_pred_test,
    10,
)

top20_overlap = topk_overlap(
    y_true_test,
    y_pred_test,
    20,
)

# ============================================================
# Elite Diagnostics
# ============================================================

elite = elite_metrics(
    y_true_test,
    y_pred_test,
)

# ============================================================
# Print Metrics
# ============================================================

print("\nTest Metrics")
print("-" * 40)

print(f"MAE                 : {mae:.4f}")
print(f"RMSE                : {rmse:.4f}")
print(f"R2                  : {r2:.4f}")
print(f"Spearman            : {spearman:.4f}")
print(f"KendallTau          : {kendall:.4f}")

print(f"Precision@10        : {precision10:.4f}")
print(f"Precision@20        : {precision20:.4f}")

print(f"Recall@10           : {recall10:.4f}")
print(f"Recall@20           : {recall20:.4f}")

print(f"NDCG@10             : {ndcg10:.4f}")
print(f"NDCG@20             : {ndcg20:.4f}")

print(f"Top10Overlap        : {top10_overlap:.4f}")
print(f"Top20Overlap        : {top20_overlap:.4f}")

print("\nElite Influencer Diagnostics")
print("-" * 40)

for metric, value in elite.items():

    print(
        f"{metric:<20}: {value:.4f}"
    )

# ============================================================
# Top-10 Ranking Analysis
# ============================================================

print("\nTop-10 Ranking Analysis")
print("-" * 40)

true_top10 = np.argsort(
    y_true_test
)[::-1][:10]

pred_top10 = np.argsort(
    y_pred_test
)[::-1][:10]

print("\nGround Truth Top-10:")

for rank, idx in enumerate(
    true_top10,
    start=1,
):

    print(
        f"{rank:2d}. "
        f"Node {test_node_ids[idx]:6d} | "
        f"Influence = "
        f"{y_true_test[idx]:8.4f}"
    )

print("\nPredicted Top-10:")

for rank, idx in enumerate(
    pred_top10,
    start=1,
):

    print(
        f"{rank:2d}. "
        f"Node {test_node_ids[idx]:6d} | "
        f"Prediction = "
        f"{y_pred_test[idx]:8.4f}"
    )

# ============================================================
# Prediction Export
# ============================================================

prediction_df = pd.DataFrame({

    "node_id":
        test_node_ids,

    "y_true":
        y_true_test,

    "y_pred":
        y_pred_test,

    "split":
        "test",

    "true_rank":
        (-y_true_test).argsort().argsort() + 1,

    "pred_rank":
        (-y_pred_test).argsort().argsort() + 1,
})

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print("\nPredictions saved to:")

print(PREDICTIONS_PATH)

# ============================================================
# Experiment Summary
# ============================================================

summary = f"""
============================================================
GIN IC EXPERIMENT SUMMARY
============================================================
Model               : GIN
Task                : IC Influence Estimation
Target              : y_ic
Hidden Dimension    : {HIDDEN_DIM}
Dropout             : {DROPOUT}
Optimizer           : AdamW
Scheduler           : ReduceLROnPlateau
Learning Rate       : {LEARNING_RATE}
Weight Decay        : {WEIGHT_DECAY}
Gradient Clip       : {GRAD_CLIP}
Loss                : MSELoss
Max Epochs          : {EPOCHS}
Early Stopping      : {PATIENCE}
Best Epoch          : {best_epoch}
Best Validation MAE : {best_val_mae:.4f}

Test Performance
----------------------------------------
MAE                 : {mae:.4f}
RMSE                : {rmse:.4f}
R2                  : {r2:.4f}
Spearman            : {spearman:.4f}
KendallTau          : {kendall:.4f}
Precision@10        : {precision10:.4f}
Precision@20        : {precision20:.4f}
Recall@10           : {recall10:.4f}
Recall@20           : {recall20:.4f}
NDCG@10             : {ndcg10:.4f}
NDCG@20             : {ndcg20:.4f}
Top10Overlap        : {top10_overlap:.4f}
Top20Overlap        : {top20_overlap:.4f}

Elite Diagnostics
----------------------------------------
Top1_MAE            : {elite['Top1_MAE']:.4f}
Top5_MAE            : {elite['Top5_MAE']:.4f}
Top10_MAE           : {elite['Top10_MAE']:.4f}
Top1_MSE            : {elite['Top1_MSE']:.4f}
Top5_MSE            : {elite['Top5_MSE']:.4f}
Top10_MSE           : {elite['Top10_MSE']:.4f}
P95_MAE             : {elite['P95_MAE']:.4f}
P95_MSE             : {elite['P95_MSE']:.4f}
============================================================
"""

print("\n" + summary)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(summary)

print("Experiment Finished Successfully")

print("=" * 60)

print("\nResults saved to:")

print(RESULTS_PATH)

print("\nGIN-IC Benchmark Complete.")