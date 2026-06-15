# ============================================================
# train_gat_ic.py
# Chunk 1: Imports, Configuration, Dataset Loading
# ============================================================

import os
import copy
import random
import warnings
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error

from torch_geometric.nn import GATConv

from Try1.scripts.training.utils_metrics import (
    compute_metrics,
    masked_huber_loss,
    get_mask_predictions,
    save_predictions,
)

warnings.filterwarnings("ignore")


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

DATA_PATH = "data/higgs_pyg.pt"

CHECKPOINT_PATH = (
    "checkpoints/best_gat_ic.pt"
)

RESULTS_PATH = (
    "results/ic/gat_results.txt"
)

PREDICTIONS_PATH = (
    "results/ic/gat_predictions.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

HIDDEN_DIM = 32
DROPOUT = 0.1

LR = 1e-3
WEIGHT_DECAY = 1e-4

# Match actual GCN setup
EPOCHS = 1000
PATIENCE = 100

HUBER_DELTA = 0.1


# ============================================================
# Dataset Loading
# ============================================================

print("=" * 60)
print("Loading PyG Dataset...")
print("=" * 60)

data = torch.load(
    DATA_PATH,
    weights_only=False
)

print(data)

print("\nDataset Summary")
print("-" * 40)

print("Nodes       :", data.num_nodes)
print("Edges       :", data.edge_index.shape[1])
print("Features    :", data.x.shape[1])

print(
    "Train Nodes :",
    data.train_mask.sum().item()
)

print(
    "Val Nodes   :",
    data.val_mask.sum().item()
)

print(
    "Test Nodes  :",
    data.test_mask.sum().item()
)


print("\nIC Label Statistics")
print("-" * 40)

num_nan = torch.isnan(
    data.y_ic
).sum().item()

print("NaN Labels      :", num_nan)

print(
    "Labeled Nodes   :",
    data.num_nodes - num_nan
)


# ============================================================
# Move Dataset to Device
# ============================================================

data = data.to(DEVICE)

print("\nUsing Device :", DEVICE)

print("=" * 60)
print("Dataset Ready")
print("=" * 60)

# ============================================================
# Chunk 2: GAT Model, Loss, Optimizer Helpers
# ============================================================

import torch.nn.functional as F


# ============================================================
# GAT Model Definition
# ============================================================

class GAT(nn.Module):
    """
    2-layer GAT for IC Influence Estimation.

    Fair baseline design:
        Input
          ↓
        GATConv
          ↓
        ELU
          ↓
        Dropout
          ↓
        GATConv
          ↓
        Output

    Uses single attention head to keep parameter
    count comparable with GCN and GraphSAGE.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout=0.1,
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.conv2 = GATConv(
            hidden_channels,
            1,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):
        """
        Returns
        -------
        Tensor of shape [num_nodes]
        """

        # First GAT layer
        x = self.conv1(
            x,
            edge_index,
        )

        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Output layer
        x = self.conv2(
            x,
            edge_index,
        )

        return x.squeeze(-1)


# ============================================================
# Model Initialization
# ============================================================

model = GAT(
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
    f"{total_params:,}"
)

print(
    f"Trainable Parameters : "
    f"{trainable_params:,}"
)


# ============================================================
# Loss Function
# ============================================================

loss_fn = nn.HuberLoss(
    delta=HUBER_DELTA
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Early Stopping Variables
# ============================================================

best_val_mae = float("inf")

best_epoch = -1

best_state_dict = None

patience_counter = 0


history = {
    "train_loss": [],
    "val_mae": [],
}


# ============================================================
# Training Configuration Summary
# ============================================================

print("\nTraining Configuration")
print("-" * 40)

print(
    f"Loss            : "
    f"HuberLoss(delta={HUBER_DELTA})"
)

print("Optimizer       : AdamW")

print(
    f"Learning Rate   : {LR}"
)

print(
    f"Weight Decay    : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Epochs          : {EPOCHS}"
)

print(
    f"Patience        : {PATIENCE}"
)

print(
    f"Dropout         : {DROPOUT}"
)

print(
    f"Hidden Dim      : "
    f"{HIDDEN_DIM}"
)

print("=" * 60)
print("Model Setup Complete")
print("=" * 60)

# ============================================================
# Chunk 3: Training Loop with Early Stopping
# ============================================================

print("\n" + "=" * 60)
print("Starting GAT Training on IC Labels...")
print("=" * 60)


# ============================================================
# Training Function
# ============================================================

def train_one_epoch():
    """
    Perform one full-batch training epoch.

    Returns
    -------
    float
        Training loss
    """

    model.train()

    optimizer.zero_grad()

    # Forward pass on all 5000 nodes
    predictions = model(
        data.x,
        data.edge_index
    )

    # Loss only on labeled training nodes
    loss = masked_huber_loss(
        predictions=predictions,
        targets=data.y_ic,
        mask=data.train_mask,
        loss_fn=loss_fn,
    )

    loss.backward()

    optimizer.step()

    return loss.item()


# ============================================================
# Validation Function
# ============================================================

@torch.no_grad()
def validate():
    """
    Compute validation MAE.
    """

    model.eval()

    predictions = model(
        data.x,
        data.edge_index
    )

    y_true, y_pred = get_mask_predictions(
        predictions=predictions,
        targets=data.y_ic,
        mask=data.val_mask,
    )

    val_mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return val_mae


# ============================================================
# Main Training Loop
# ============================================================

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss = train_one_epoch()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    val_mae = validate()

    history["train_loss"].append(
        train_loss
    )

    history["val_mae"].append(
        val_mae
    )

    # --------------------------------------------------------
    # Save Best Model
    # --------------------------------------------------------

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch

        best_state_dict = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state_dict,
            CHECKPOINT_PATH
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
            f"Epoch {epoch:04d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Best Val MAE: {best_val_mae:.4f}"
        )

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        print(
            f"No improvement for "
            f"{PATIENCE} consecutive epochs."
        )

        break


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
    f"\nBest model saved to:"
    f" {CHECKPOINT_PATH}"
)


# ============================================================
# Restore Best Checkpoint
# ============================================================

print("\nLoading best checkpoint...")

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

print("Best checkpoint restored.")

print("=" * 60)

# ============================================================
# Chunk 4: Final Evaluation and Thesis Summary
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
    Evaluate the best checkpoint on test nodes.
    """

    model.eval()

    predictions = model(
        data.x,
        data.edge_index
    )

    y_true, y_pred = get_mask_predictions(
        predictions=predictions,
        targets=data.y_ic,
        mask=data.test_mask,
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    return (
        metrics,
        y_true,
        y_pred,
    )


# ============================================================
# Run Evaluation
# ============================================================

test_metrics, y_true_test, y_pred_test = evaluate_test()


# ============================================================
# Print Test Metrics
# ============================================================

print("\nTest Metrics")
print("-" * 40)

print(
    f"Test MAE          : "
    f"{test_metrics['MAE']:.4f}"
)

print(
    f"Test RMSE         : "
    f"{test_metrics['RMSE']:.4f}"
)

print(
    f"Test R²           : "
    f"{test_metrics['R2']:.4f}"
)

print(
    f"Spearman          : "
    f"{test_metrics['Spearman']:.4f}"
)

print(
    f"NDCG@10           : "
    f"{test_metrics['NDCG@10']:.4f}"
)

print(
    f"Precision@10      : "
    f"{test_metrics['Precision@10']:.4f}"
)


# ============================================================
# Top-10 Influence Ranking Analysis
# ============================================================

print("\nTop-10 Ranking Analysis")
print("-" * 40)

true_top10_idx = (
    np.argsort(y_true_test)[-10:][::-1]
)

pred_top10_idx = (
    np.argsort(y_pred_test)[-10:][::-1]
)

print("\nGround Truth Top-10 IC Influence:")

for rank, idx in enumerate(
    true_top10_idx,
    start=1,
):
    print(
        f"{rank:2d}. "
        f"Influence = "
        f"{y_true_test[idx]:8.4f}"
    )


print("\nPredicted Top-10 IC Influence:")

for rank, idx in enumerate(
    pred_top10_idx,
    start=1,
):
    print(
        f"{rank:2d}. "
        f"Influence = "
        f"{y_pred_test[idx]:8.4f}"
    )


# ============================================================
# Save Predictions
# ============================================================

save_predictions(
    PREDICTIONS_PATH,
    y_true_test,
    y_pred_test,
)


# ============================================================
# Experiment Summary
# ============================================================

print("\n" + "=" * 60)
print("GAT IC EXPERIMENT SUMMARY")
print("=" * 60)

print("Model                 : GAT")
print("Task                  : IC Influence Estimation")
print("Target                : y_ic (IC-v1)")
print(f"Hidden Dimension      : {HIDDEN_DIM}")
print(f"Dropout               : {DROPOUT}")
print("Optimizer             : AdamW")
print(f"Learning Rate         : {LR}")
print(f"Weight Decay          : {WEIGHT_DECAY}")
print(
    f"Loss                  : "
    f"HuberLoss(delta={HUBER_DELTA})"
)
print(f"Max Epochs            : {EPOCHS}")
print(f"Early Stopping        : {PATIENCE}")
print(f"Best Epoch            : {best_epoch}")

print("\nValidation Performance")
print("-" * 40)

print(
    f"Best Validation MAE   : "
    f"{best_val_mae:.4f}"
)

print("\nTest Performance")
print("-" * 40)

print(
    f"MAE                   : "
    f"{test_metrics['MAE']:.4f}"
)

print(
    f"RMSE                  : "
    f"{test_metrics['RMSE']:.4f}"
)

print(
    f"R²                    : "
    f"{test_metrics['R2']:.4f}"
)

print(
    f"Spearman              : "
    f"{test_metrics['Spearman']:.4f}"
)

print(
    f"NDCG@10               : "
    f"{test_metrics['NDCG@10']:.4f}"
)

print(
    f"Precision@10          : "
    f"{test_metrics['Precision@10']:.4f}"
)

print("=" * 60)
print("Experiment Finished Successfully")
print("=" * 60)


# ============================================================
# Save Results to Text File
# ============================================================

with open(RESULTS_PATH, "w") as f:

    f.write(
        "GAT IC EXPERIMENT RESULTS\n"
    )

    f.write("=" * 50 + "\n")

    f.write(
        f"Best Epoch: "
        f"{best_epoch}\n"
    )

    f.write(
        f"Best Validation MAE: "
        f"{best_val_mae:.6f}\n\n"
    )

    f.write("Test Metrics\n")

    f.write("-" * 20 + "\n")

    for metric, value in test_metrics.items():

        f.write(
            f"{metric}: "
            f"{value:.6f}\n"
        )


print(
    f"\nResults saved to: "
    f"{RESULTS_PATH}"
)

print(
    f"Predictions saved to: "
    f"{PREDICTIONS_PATH}"
)


# ============================================================
# End of Script
# ============================================================