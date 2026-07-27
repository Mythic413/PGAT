
import random
import warnings
from pathlib import Path

import numpy as np
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

from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

# Reproducibility

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration

DATA_PATH = Path("Try2/data/higgs_pyg.pt")

CHECKPOINT_PATH = Path(
    "Try2/checkpoints/best_gcn_lt.pt"
)

RESULTS_PATH = Path(
    "Try2/results/gcn_lt_results.txt"
)

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Architecture

HIDDEN_DIM = 32
DROPOUT = 0.10

# Optimization

LR = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 500
PATIENCE = 100

GRAD_CLIP = 1.0

# Scheduler

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 20
MIN_LR = 1e-5

# Ranking Metrics

def precision_at_k(
    y_true,
    y_pred,
    k=10
):
    """Compute Precision@K."""

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
    k=10
):
    """Compute Recall@K."""

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
    k=10
):
    """Compute NDCG@K."""

    k = min(k, len(y_true))

    if np.var(y_pred) < 1e-12:
        return 0.0

    return ndcg_score(
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        k=k
    )

def topk_overlap(
    y_true,
    y_pred,
    k=10
):
    """Compute Top-K overlap."""

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

# Evaluation Metrics

def compute_metrics(
    y_true,
    y_pred
):
    """Compute regression and ranking metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    spearman_corr, _ = spearmanr(
        y_true,
        y_pred
    )

    kendall_corr, _ = kendalltau(
        y_true,
        y_pred
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
            10
        ),

        "Precision@20": precision_at_k(
            y_true,
            y_pred,
            20
        ),

        "Recall@10": recall_at_k(
            y_true,
            y_pred,
            10
        ),

        "Recall@20": recall_at_k(
            y_true,
            y_pred,
            20
        ),

        "NDCG@10": ndcg_at_k(
            y_true,
            y_pred,
            10
        ),

        "NDCG@20": ndcg_at_k(
            y_true,
            y_pred,
            20
        ),

        "Top10Overlap": topk_overlap(
            y_true,
            y_pred,
            10
        ),

        "Top20Overlap": topk_overlap(
            y_true,
            y_pred,
            20
        ),
    }

print("=" * 60)
print("GCN-LT Publication Baseline")
print("=" * 60)

print("Seed              :", SEED)
print("Device            :", DEVICE)

print("\nArchitecture")
print("-" * 40)
print("Hidden Dimension  :", HIDDEN_DIM)
print("Dropout           :", DROPOUT)

print("\nOptimization")
print("-" * 40)
print("Loss              : MSELoss")
print("Optimizer         : AdamW")
print("Scheduler         : ReduceLROnPlateau")
print("Learning Rate     :", LR)
print("Weight Decay      :", WEIGHT_DECAY)
print("Gradient Clip     :", GRAD_CLIP)

print("\nTraining")
print("-" * 40)
print("Epochs            :", EPOCHS)
print("Patience          :", PATIENCE)

print("=" * 60)

print("\n" + "=" * 60)
print("Loading PyG Dataset...")
print("=" * 60)

data = torch.load(
    DATA_PATH,
    weights_only=False
)

print(data)

print("\nDataset Summary")
print("-" * 40)

print("Nodes        :", data.num_nodes)
print("Edges        :", data.edge_index.shape[1])
print("Features     :", data.x.shape[1])

print(
    "Train Nodes  :",
    data.train_mask.sum().item()
)

print(
    "Val Nodes    :",
    data.val_mask.sum().item()
)

print(
    "Test Nodes   :",
    data.test_mask.sum().item()
)

# LT Label Diagnostics

print("\nLT Label Statistics")
print("-" * 40)

num_nan = torch.isnan(
    data.y_lt
).sum().item()

valid_lt = data.y_lt[
    ~torch.isnan(data.y_lt)
]

print("NaN Labels       :", num_nan)

print(
    "Labeled Nodes    :",
    valid_lt.numel()
)

print(
    "LT Min Spread    :",
    f"{valid_lt.min().item():.4f}"
)

print(
    "LT Mean Spread   :",
    f"{valid_lt.mean().item():.4f}"
)

print(
    "LT Max Spread    :",
    f"{valid_lt.max().item():.4f}"
)

# Move to Device

data = data.to(DEVICE)

print("\nUsing Device:", DEVICE)

print("=" * 60)

# Publication-Grade  GCN

class GCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout=0.1
    ):
        super().__init__()

        # Pure GCN Layers

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
           hidden_channels,
            1)

        # Stabilization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.dropout = dropout

        # Initialization

        self.reset_parameters()

    def reset_parameters(self):

        self.conv1.reset_parameters()

        self.conv2.reset_parameters()

    def forward(self, x, edge_index):

        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)

        h = F.dropout(
        h,
        p=self.dropout,
        training=self.training,
    )

        h = self.conv2(h, edge_index)

        return h.squeeze(-1)

# Model Initialization

model = GCN(
    in_channels=data.x.shape[1],
    hidden_channels=HIDDEN_DIM,
    dropout=DROPOUT
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
    f"{total_params:,}"
)

print(
    f"Trainable Parameters : "
    f"{trainable_params:,}"
)

print("=" * 60)
print("Dataset + Model Ready")
print("=" * 60)

# Loss Function

loss_fn = nn.MSELoss()

print("\nOptimization Components")
print("-" * 40)

print("Loss Function : MSELoss")

# Optimizer

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

print("Optimizer     : AdamW")

# Scheduler

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
    min_lr=MIN_LR,
)

print("Scheduler     : ReduceLROnPlateau")

print(
    f"Factor         : {SCHEDULER_FACTOR}"
)

print(
    f"Patience       : {SCHEDULER_PATIENCE}"
)

print(
    f"Minimum LR     : {MIN_LR}"
)

# Masked MSE Loss

def masked_mse_loss(
    predictions,
    targets,
    mask
):
    """Compute masked MSE Loss."""

    valid_mask = (
        mask &
        (~torch.isnan(targets))
    )

    pred = predictions[valid_mask]

    true = targets[valid_mask]

    return loss_fn(
        pred,
        true
    )

# Prediction Extraction

@torch.no_grad()
def get_mask_predictions(
    predictions,
    targets,
    mask
):
    """Get predictions on masked nodes."""

    valid_mask = (
        mask &
        (~torch.isnan(targets))
    )

    y_true = (
        targets[valid_mask]
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions[valid_mask]
        .detach()
        .cpu()
        .numpy()
    )

    return y_true, y_pred

# Checkpoint Utilities

def save_checkpoint(
    epoch,
    best_val_mae
):
    """Save full training state."""

    checkpoint = {
        "epoch":
            epoch,

        "best_val_mae":
            best_val_mae,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

def load_checkpoint():
    """Restore best checkpoint."""

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    return checkpoint

# Early Stopping State

best_val_mae = float("inf")

best_epoch = -1

patience_counter = 0

history = {
    "train_loss": [],
    "val_mae": [],
    "learning_rate": [],
}

# Training Summary

print("\nTraining Configuration")
print("-" * 40)

print(
    f"Learning Rate   : {LR}"
)

print(
    f"Weight Decay    : {WEIGHT_DECAY}"
)

print(
    f"Epoch Budget    : {EPOCHS}"
)

print(
    f"Early Stopping  : {PATIENCE}"
)

print(
    f"Gradient Clip   : {GRAD_CLIP}"
)

print(
    f"Checkpoint Path : {CHECKPOINT_PATH}"
)

print("=" * 60)
print("Optimization Setup Complete")
print("=" * 60)

print("\n" + "=" * 60)
print("Starting GCN Training on LT Labels...")
print("=" * 60)

# Training Function

def train_one_epoch():
    """Train one epoch."""

    model.train()

    optimizer.zero_grad()

    # Forward pass

    predictions = model(
        data.x,
        data.edge_index
    )

    # Loss on training nodes only

    loss = masked_mse_loss(
        predictions,
        data.y_lt,
        data.train_mask
    )

    # Backpropagation

    loss.backward()

    # Gradient clipping

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRAD_CLIP
    )

    optimizer.step()

    return loss.item()

# Validation Function

@torch.no_grad()
def validate():
    """Compute validation metrics."""

    model.eval()

    predictions = model(
        data.x,
        data.edge_index
    )

    y_true, y_pred = get_mask_predictions(
        predictions,
        data.y_lt,
        data.val_mask
    )

    val_metrics = compute_metrics(
        y_true,
        y_pred
    )

    val_mae = val_metrics["MAE"]

    return val_mae, val_metrics

# Main Training Loop

for epoch in range(1, EPOCHS + 1):

    # Train

    train_loss = train_one_epoch()

    # Validate

    val_mae, val_metrics = validate()

    # Scheduler Step

    scheduler.step(val_mae)

    current_lr = optimizer.param_groups[0]["lr"]

    # History

    history["train_loss"].append(
        train_loss
    )

    history["val_mae"].append(
        val_mae
    )

    history["learning_rate"].append(
        current_lr
    )

    # Save Best Model

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch

        patience_counter = 0

        save_checkpoint(
            epoch=epoch,
            best_val_mae=best_val_mae
        )

    else:

        patience_counter += 1

    # Logging

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Best Val MAE: {best_val_mae:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    # Early Stopping

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        print(
            f"No validation improvement "
            f"for {PATIENCE} epochs."
        )

        break

# Training Summary

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

print(
    f"\nBest checkpoint saved to:\n"
    f"{CHECKPOINT_PATH}"
)

# Restore Best Checkpoint

print("\nLoading best checkpoint...")

checkpoint = load_checkpoint()

print(
    f"Restored epoch      : "
    f"{checkpoint['epoch']}"
)

print(
    f"Restored Val MAE    : "
    f"{checkpoint['best_val_mae']:.4f}"
)

model.eval()

print("Checkpoint restored.")

print("=" * 60)

print("\n" + "=" * 60)
print("Final Evaluation on Test Set")
print("=" * 60)

# Test Evaluation
def elite_diagnostics(
    y_true,
    y_pred,
):
    """Compute elite diagnostics."""

    diagnostics = {}

    # Top-1 MAE

    idx = np.argsort(y_true)[-1:]

    diagnostics["Top1_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    # Top-5 MAE

    idx = np.argsort(y_true)[-5:]

    diagnostics["Top5_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    # Top-10 MAE

    idx = np.argsort(y_true)[-10:]

    diagnostics["Top10_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    # Top-1 MSE

    idx = np.argsort(y_true)[-1:]

    diagnostics["Top1_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    # Top-5 MSE

    idx = np.argsort(y_true)[-5:]

    diagnostics["Top5_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    # Top-10 MSE

    idx = np.argsort(y_true)[-10:]

    diagnostics["Top10_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    # 95th Percentile MAE

    threshold = np.percentile(
        y_true,
        95
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

    # 95th Percentile MSE

    diagnostics["P95_MSE"] = np.mean(
        (
            y_true[elite_mask]
            - y_pred[elite_mask]
        ) ** 2
    )

    return diagnostics

@torch.no_grad()
def evaluate_test():
    """Evaluate model on test dataset."""

    model.eval()

    predictions = model(
        data.x,
        data.edge_index
    )

    valid_mask = (
        data.test_mask &
        (~torch.isnan(data.y_lt))
    )

    y_true = (
        data.y_lt[valid_mask]
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions[valid_mask]
        .detach()
        .cpu()
        .numpy()
    )

    metrics = compute_metrics(
        y_true,
        y_pred
    )

    elite_metrics = elite_diagnostics(
        y_true,
        y_pred
    )

    node_ids = (
        data.node_ids[valid_mask]
        .detach()
        .cpu()
        .numpy()
    )

    return (
    metrics,
    y_true,
    y_pred,
    node_ids,
    elite_metrics,
)

# Run Test Evaluation

(
    test_metrics,
    y_true_test,
    y_pred_test,
    test_node_ids,
    elite_metrics,
) = evaluate_test()

# Print Metrics

print("\nTest Metrics")
print("-" * 40)

for metric, value in test_metrics.items():

    if isinstance(value, float):

        print(
            f"{metric:<20}: "
            f"{value:.4f}"
        )

    else:

        print(
            f"{metric:<20}: "
            f"{value}"
        )

print("\nElite Diagnostics")
print("-" * 40)

for metric, value in elite_metrics.items():

    print(
        f"{metric:<20}: "
        f"{value:.4f}"
    )

# Top-10 Influence Analysis

print("\nTop-10 Ranking Analysis")
print("-" * 40)

true_top10 = np.argsort(
    y_true_test
)[-10:][::-1]

pred_top10 = np.argsort(
    y_pred_test
)[-10:][::-1]

print("\nGround Truth Top-10:")

for rank, idx in enumerate(
    true_top10,
    start=1
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
    start=1
):

    print(
        f"{rank:2d}. "
        f"Node {test_node_ids[idx]:6d} | "
        f"Prediction = "
        f"{y_pred_test[idx]:8.4f}"
    )

# Prediction Export

prediction_path = (
    RESULTS_PATH.parent /
    "gcn_lt_predictions.csv"
)

import pandas as pd

true_rank = (
    len(y_true_test)
    - y_true_test.argsort().argsort()
)

pred_rank = (
    len(y_pred_test)
    - y_pred_test.argsort().argsort()
)

prediction_df = pd.DataFrame({
    "node_id": test_node_ids,
    "y_true": y_true_test,
    "y_pred": y_pred_test,
    "split": "test",
    "true_rank": true_rank,
    "pred_rank": pred_rank,
})

prediction_df.to_csv(
    prediction_path,
    index=False
)

print(
    f"\nPredictions saved to:\n"
    f"{prediction_path}"
)

# Experiment Summary

print("\n" + "=" * 60)
print("GCN LT EXPERIMENT SUMMARY")
print("=" * 60)

print("Model               :  GCN")

print(
    "Task                : "
    "LT Influence Estimation"
)

print(
    "Target              : "
    "y_lt"
)

print(
    "Hidden Dimension    : "
    f"{HIDDEN_DIM}"
)

print(
    "Dropout             : "
    f"{DROPOUT}"
)

print(
    "Optimizer           : "
    "AdamW"
)

print(
    "Scheduler           : "
    "ReduceLROnPlateau"
)

print(
    "Learning Rate       : "
    f"{LR}"
)

print(
    "Weight Decay        : "
    f"{WEIGHT_DECAY}"
)

print(
    "Gradient Clip       : "
    f"{GRAD_CLIP}"
)

print(
    "Loss                : "
    "MSELoss"
)

print(
    "Max Epochs          : "
    f"{EPOCHS}"
)

print(
    "Early Stopping      : "
    f"{PATIENCE}"
)

print(
    "Best Epoch          : "
    f"{best_epoch}"
)

print(
    "Best Validation MAE: "
    f"{best_val_mae:.4f}"
)

print("\nTest Performance")
print("-" * 40)

for metric, value in test_metrics.items():

    if isinstance(value, float):

        print(
            f"{metric:<20}: "
            f"{value:.4f}"
        )

    else:

        print(
            f"{metric:<20}: "
            f"{value}"
        )

print("=" * 60)
print("Experiment Finished Successfully")
print("=" * 60)

# Save Summary Results

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GCN LT EXPERIMENT RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best Validation MAE: "
        f"{best_val_mae:.6f}\n\n"
    )

    f.write(
        "Test Metrics\n"
    )

    f.write(
        "-" * 40 + "\n"
    )

    for metric, value in test_metrics.items():

        f.write(
            f"{metric}: "
            f"{value}\n"
        )

    f.write("\n")

    f.write(
    "Elite Diagnostics\n"
)

    f.write(
    "-" * 40 + "\n"
)

    for metric, value in elite_metrics.items():

        f.write(
            f"{metric}: "
            f"{value}\n"
    )

print(
    f"\nResults saved to:\n"
    f"{RESULTS_PATH}"
)

print("\nGCN- LT Benchmark Complete.")