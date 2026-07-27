# Training
# Configuration

import os
import copy
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.metrics import ndcg_score

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

DATA_PATH = "data/higgs_pyg.pt"
CHECKPOINT_PATH = "best_gcn_ic.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HIDDEN_DIM = 32
DROPOUT = 0.1

LR = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 1000
PATIENCE = 100

HUBER_DELTA = 0.1

def precision_at_k(y_true, y_pred, k=10):
    """Compute Precision@K."""

    k = min(k, len(y_true))

    true_topk = np.argsort(y_true)[-k:]
    pred_topk = np.argsort(y_pred)[-k:]

    overlap = len(set(true_topk).intersection(set(pred_topk)))

    return overlap / k

def ndcg_at_k(y_true, y_pred, k=10):
    """Compute NDCG@K."""

    k = min(k, len(y_true))

    return ndcg_score(
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        k=k
    )

def compute_metrics(y_true, y_pred):
    """Compute regression and ranking metrics."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    spearman_corr, _ = spearmanr(
        y_true,
        y_pred
    )

    ndcg10 = ndcg_at_k(
        y_true,
        y_pred,
        k=10
    )

    precision10 = precision_at_k(
        y_true,
        y_pred,
        k=10
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Spearman": spearman_corr,
        "NDCG@10": ndcg10,
        "Precision@10": precision10,
    }

# Dataset

print("Loading PyG Dataset...")

data = torch.load(
    DATA_PATH,
    weights_only=False
)

print(data)

print("\nDataset Summary")

print("Nodes:", data.num_nodes)
print("Edges:", data.edge_index.shape[1])
print("Features:", data.x.shape[1])

print("Train:", data.train_mask.sum().item())
print("Validation:", data.val_mask.sum().item())
print("Test:", data.test_mask.sum().item())

print("\nIC Label Statistics")
print("-" * 40)

num_nan = torch.isnan(data.y_ic).sum().item()

print("NaN Labels:", num_nan)
print("Labeled Nodes:", data.num_nodes - num_nan)

data = data.to(DEVICE)

print("\nUsing Device:", DEVICE)

# Model

class GCN(nn.Module):
    """2-layer GCN model."""

    def __init__(self, in_channels, hidden_channels, dropout=0.1):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            1
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        """Returns: shape = [num_nodes]"""

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(x, edge_index)

        return x.squeeze(-1)

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

print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")

loss_fn = nn.HuberLoss(
    delta=HUBER_DELTA
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

def masked_huber_loss(
    predictions,
    targets,
    mask
):
    """Compute masked Huber Loss."""

    valid_mask = (
        mask &
        (~torch.isnan(targets))
    )

    pred = predictions[valid_mask]
    true = targets[valid_mask]

    return loss_fn(pred, true)

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

best_val_mae = float("inf")

best_epoch = -1

best_state_dict = None

patience_counter = 0

history = {
    "train_loss": [],
    "val_mae": []
}

print("\nTraining Configuration")
print(f"Loss            : HuberLoss(delta={HUBER_DELTA})")
print("Optimizer       : AdamW")
print(f"Learning Rate   : {LR}")
print(f"Weight Decay    : {WEIGHT_DECAY}")
print(f"Epochs          : {EPOCHS}")
print(f"Patience        : {PATIENCE}")
print(f"Dropout         : {DROPOUT}")

print("Setup Complete")

print("Starting GCN Training on IC Labels...")

def train_one_epoch():
    """Train one epoch."""

    model.train()

    optimizer.zero_grad()

    predictions = model(
        data.x,
        data.edge_index
    )

    loss = masked_huber_loss(
        predictions,
        data.y_ic,
        data.train_mask
    )

    loss.backward()

    optimizer.step()

    return loss.item()

# Validation

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
        data.y_ic,
        data.val_mask
    )

    val_mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return val_mae

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()

    val_mae = validate()

    history["train_loss"].append(train_loss)
    history["val_mae"].append(val_mae)

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

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Best Val MAE: {best_val_mae:.4f}"
        )

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")
        print(
            f"No improvement for "
            f"{PATIENCE} consecutive epochs."
        )

        break

print("Training Complete")

print(f"Best Epoch        : {best_epoch}")
print(f"Best Validation MAE: {best_val_mae:.4f}")

print(
    f"\nBest model saved to:"
    f" {CHECKPOINT_PATH}"
)

print("\nLoading best checkpoint...")

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Best checkpoint restored.")
print("Final Evaluation on Test Set")
# Evaluation

@torch.no_grad()
def evaluate_test():
    """Evaluate model on test dataset."""

    model.eval()

    predictions = model(
        data.x,
        data.edge_index
    )

    y_true, y_pred = get_mask_predictions(
        predictions,
        data.y_ic,
        data.test_mask
    )

    metrics = compute_metrics(
        y_true,
        y_pred
    )

    return metrics, y_true, y_pred

test_metrics, y_true_test, y_pred_test = evaluate_test()

print("\nTest Metrics")

print(f"Test MAE          : {test_metrics['MAE']:.4f}")
print(f"Test RMSE         : {test_metrics['RMSE']:.4f}")
print(f"Test R²           : {test_metrics['R2']:.4f}")
print(f"Spearman          : {test_metrics['Spearman']:.4f}")
print(f"NDCG@10           : {test_metrics['NDCG@10']:.4f}")
print(f"Precision@10      : {test_metrics['Precision@10']:.4f}")

print("\nTop-10 Ranking Analysis")

true_top10_idx = np.argsort(y_true_test)[-10:][::-1]
pred_top10_idx = np.argsort(y_pred_test)[-10:][::-1]

print("\nGround Truth Top-10 IC Influence:")
for rank, idx in enumerate(true_top10_idx, start=1):
    print(
        f"{rank:2d}. "
        f"Influence = {y_true_test[idx]:8.4f}"
    )

print("\nPredicted Top-10 IC Influence:")
for rank, idx in enumerate(pred_top10_idx, start=1):
    print(
        f"{rank:2d}. "
        f"Influence = {y_pred_test[idx]:8.4f}"
    )

print("GCN IC EXPERIMENT SUMMARY")

print(f"Model                 : GCN")
print(f"Task                  : IC Influence Estimation")
print(f"Target                : y_ic (IC-v1)")
print(f"Hidden Dimension      : {HIDDEN_DIM}")
print(f"Dropout               : {DROPOUT}")
print(f"Optimizer             : AdamW")
print(f"Learning Rate         : {LR}")
print(f"Weight Decay          : {WEIGHT_DECAY}")
print(f"Loss                  : HuberLoss(delta={HUBER_DELTA})")
print(f"Max Epochs            : {EPOCHS}")
print(f"Early Stopping        : {PATIENCE}")
print(f"Best Epoch            : {best_epoch}")

print("\nValidation Performance")
print(f"Best Validation MAE   : {best_val_mae:.4f}")

print("\nTest Performance")
print(f"MAE                   : {test_metrics['MAE']:.4f}")
print(f"RMSE                  : {test_metrics['RMSE']:.4f}")
print(f"R²                    : {test_metrics['R2']:.4f}")
print(f"Spearman              : {test_metrics['Spearman']:.4f}")
print(f"NDCG@10               : {test_metrics['NDCG@10']:.4f}")
print(f"Precision@10          : {test_metrics['Precision@10']:.4f}")

print("Experiment Finished")

results_path = "gcn_ic_results.txt"

with open(results_path, "w") as f:
    f.write("GCN IC EXPERIMENT RESULTS\n")

    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Validation MAE: {best_val_mae:.6f}\n\n")

    f.write("Test Metrics\n")

    for metric, value in test_metrics.items():
        f.write(f"{metric}: {value:.6f}\n")

print(f"\nResults saved to: {results_path}")
