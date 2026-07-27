# Try2 Multitask PGAT Ablation
# Imports
# Reproducibility
# Configuration
# Dataset Loading
# Dataset Verification
# Label Statistics
# Metrics Utilities
# Elite Diagnostics

import os
import copy
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Metrics Imports

from scipy.stats import (
    spearmanr,
    kendalltau,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score,
)

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

DATA_PATH = "Try2/data/higgs_pyg.pt"

CHECKPOINT_PATH = (
    "Try2/checkpoints/"
    "best_pgat_im_vanilla.pt"
)

IC_RESULTS_PATH = (
    "Try2/results/im/"
    "pgat_im_ic_results.txt"
)

LT_RESULTS_PATH = (
    "Try2/results/im/"
    "pgat_im_lt_results.txt"
)

IC_PREDICTIONS_PATH = (
    "Try2/results/im/"
    "pgat_im_ic_predictions.csv"
)

LT_PREDICTIONS_PATH = (
    "Try2/results/im/"
    "pgat_im_lt_predictions.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Frozen Vanilla Hyperparameters

HIDDEN_DIM = 32

HEADS = 4

NUM_LAYERS = 2

EMBED_DIM = 32

DROPOUT = 0.1

LR = 1e-3

WEIGHT_DECAY = 1e-4

EPOCHS = 500

PATIENCE = 100

GRAD_CLIP = 1.0

# Create Output Directories

os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(IC_RESULTS_PATH),
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(LT_RESULTS_PATH),
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(IC_PREDICTIONS_PATH),
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(LT_PREDICTIONS_PATH),
    exist_ok=True,
)

# Dataset Loading

print("=" * 60)
print("PGAT-IM Multitask Ablation")
print("=" * 60)

print("\nLoading Frozen Dataset...")
print("-" * 60)

data = torch.load(
    DATA_PATH,
    weights_only=False,
)

print(data)

# Dataset Summary

print("\nDataset Summary")
print("-" * 40)

print(
    f"Nodes          : "
    f"{data.num_nodes:,}"
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

# Frozen Split Verification

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

print(
    f"Train Nodes : {train_count}"
)

print(
    f"Val Nodes   : {val_count}"
)

print(
    f"Test Nodes  : {test_count}"
)

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

# Feature Verification

print("\nFeature Verification")
print("-" * 40)

print(
    f"x shape : "
    f"{tuple(data.x.shape)}"
)

assert data.x.shape == (5000, 9), (
    f"Expected x=(5000,9), "
    f"found {tuple(data.x.shape)}"
)

print(
    "✓ Frozen feature matrix verified."
)

# IC Label Statistics

print("\nIC Label Statistics")
print("-" * 40)

ic_valid = ~torch.isnan(
    data.y_ic
)

ic_nan = torch.isnan(
    data.y_ic
).sum().item()

print(
    f"NaN Labels     : {ic_nan}"
)

print(
    f"Labeled Nodes  : "
    f"{ic_valid.sum().item()}"
)

print(
    f"Mean Spread    : "
    f"{data.y_ic[ic_valid].mean().item():.4f}"
)

print(
    f"Max Spread     : "
    f"{data.y_ic[ic_valid].max().item():.4f}"
)

# LT Label Statistics

print("\nLT Label Statistics")
print("-" * 40)

lt_valid = ~torch.isnan(
    data.y_lt
)

lt_nan = torch.isnan(
    data.y_lt
).sum().item()

print(
    f"NaN Labels     : {lt_nan}"
)

print(
    f"Labeled Nodes  : "
    f"{lt_valid.sum().item()}"
)

print(
    f"Mean Spread    : "
    f"{data.y_lt[lt_valid].mean().item():.4f}"
)

print(
    f"Max Spread     : "
    f"{data.y_lt[lt_valid].max().item():.4f}"
)

# Metric Utilities

def precision_at_k(
    y_true,
    y_pred,
    k,
):

    true_topk = set(
        np.argsort(-y_true)[:k]
    )

    pred_topk = set(
        np.argsort(-y_pred)[:k]
    )

    return (
        len(
            true_topk.intersection(
                pred_topk
            )
        )
        / k
    )

def recall_at_k(
    y_true,
    y_pred,
    k,
):

    true_topk = set(
        np.argsort(-y_true)[:k]
    )

    pred_topk = set(
        np.argsort(-y_pred)[:k]
    )

    return (
        len(
            true_topk.intersection(
                pred_topk
            )
        )
        / len(true_topk)
    )

def ndcg_at_k(
    y_true,
    y_pred,
    k,
):

    return ndcg_score(
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        k=k,
    )

def topk_overlap(
    y_true,
    y_pred,
    k,
):

    true_topk = set(
        np.argsort(-y_true)[:k]
    )

    pred_topk = set(
        np.argsort(-y_pred)[:k]
    )

    return len(
        true_topk.intersection(
            pred_topk
        )
    )

# Publication Metrics

def compute_metrics(
    y_true,
    y_pred,
):

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

        "Precision@10":
            precision_at_k(
                y_true,
                y_pred,
                10,
            ),

        "Precision@20":
            precision_at_k(
                y_true,
                y_pred,
                20,
            ),

        "Recall@10":
            recall_at_k(
                y_true,
                y_pred,
                10,
            ),

        "Recall@20":
            recall_at_k(
                y_true,
                y_pred,
                20,
            ),

        "NDCG@10":
            ndcg_at_k(
                y_true,
                y_pred,
                10,
            ),

        "NDCG@20":
            ndcg_at_k(
                y_true,
                y_pred,
                20,
            ),

        "Top10Overlap":
            topk_overlap(
                y_true,
                y_pred,
                10,
            ),

        "Top20Overlap":
            topk_overlap(
                y_true,
                y_pred,
                20,
            ),
    }

# Elite Diagnostics

def elite_diagnostics(
    y_true,
    y_pred,
):

    diagnostics = {}

    idx = np.argsort(y_true)[-1:]

    diagnostics["Top1_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    idx = np.argsort(y_true)[-5:]

    diagnostics["Top5_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    idx = np.argsort(y_true)[-10:]

    diagnostics["Top10_MAE"] = np.mean(
        np.abs(
            y_true[idx]
            - y_pred[idx]
        )
    )

    idx = np.argsort(y_true)[-1:]

    diagnostics["Top1_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    idx = np.argsort(y_true)[-5:]

    diagnostics["Top5_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

    idx = np.argsort(y_true)[-10:]

    diagnostics["Top10_MSE"] = np.mean(
        (
            y_true[idx]
            - y_pred[idx]
        ) ** 2
    )

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

    diagnostics["P95_MSE"] = np.mean(
        (
            y_true[elite_mask]
            - y_pred[elite_mask]
        ) ** 2
    )

    return diagnostics

# Move Dataset to Device

data = data.to(DEVICE)

print(
    f"\nUsing Device : {DEVICE}"
)

print("\nDataset Ready.")
print("=" * 60)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Scatter Utilities

def scatter_softmax_temp(
    logits,
    dst_idx,
    N,
    degree,
):
    """Degree-temperature scaled softmax."""

    H = logits.size(1)

    tau = (
        degree[dst_idx]
        .float()
        .clamp(min=1.0)
        .sqrt()
        .unsqueeze(-1)
    )

    scaled = logits / tau

    idx = dst_idx.unsqueeze(-1).expand(
        -1,
        H,
    )

    maxv = torch.full(
        (N, H),
        float("-inf"),
        device=logits.device,
    )

    maxv.scatter_reduce_(
        0,
        idx,
        scaled,
        reduce="amax",
        include_self=True,
    )

    ex = torch.exp(
        torch.clamp(
            scaled - maxv[dst_idx],
            -50,
            50,
        )
    )

    sumx = torch.zeros(
        N,
        H,
        device=logits.device,
    )

    sumx.scatter_add_(
        0,
        idx,
        ex,
    )

    return ex / (
        sumx[dst_idx] + 1e-10
    )

def scatter_add_heads(
    src,
    dst_idx,
    N,
):
    """Multi-head aggregation."""

    out = torch.zeros(
        N,
        src.size(1),
        src.size(2),
        device=src.device,
    )

    out.scatter_add_(
        0,
        dst_idx.view(-1, 1, 1).expand_as(src),
        src,
    )

    return out

# PGAT Convolution

class PGATConv(nn.Module):
    """PGAT convolution layer."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.heads = heads
        self.out_channels = hidden_channels

        D = heads * hidden_channels

        # Source / Destination projections

        self.W_src = nn.Linear(
            in_channels,
            D,
            bias=False,
        )

        self.W_dst = nn.Linear(
            in_channels,
            D,
            bias=False,
        )

        # Attention vectors

        self.a_src = nn.Parameter(
            torch.empty(
                heads,
                hidden_channels,
            )
        )

        self.a_dst = nn.Parameter(
            torch.empty(
                heads,
                hidden_channels,
            )
        )

        # Probability gate

        self.w_g = nn.Parameter(
            torch.tensor(1.0)
        )

        self.b_g = nn.Parameter(
            torch.tensor(0.0)
        )

        # Output projection

        self.W_out = nn.Linear(
            D,
            D,
        )

        self.norm = nn.LayerNorm(D)

        self.drop = nn.Dropout(
            dropout,
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.W_src.weight
        )

        nn.init.xavier_uniform_(
            self.W_dst.weight
        )

        nn.init.xavier_uniform_(
            self.a_src
        )

        nn.init.xavier_uniform_(
            self.a_dst
        )

        nn.init.xavier_uniform_(
            self.W_out.weight
        )

        nn.init.constant_(
            self.W_out.bias,
            0.0,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        degree,
    ):

        N = x.size(0)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        p = edge_attr.squeeze(-1)

        # Node projections

        h_src = self.W_src(x).view(
            N,
            self.heads,
            self.out_channels,
        )

        h_dst = self.W_dst(x).view(
            N,
            self.heads,
            self.out_channels,
        )

        # Attention

        a_src = self.a_src.unsqueeze(0)

        a_dst = self.a_dst.unsqueeze(0)

        scores = (
            (
                h_src[src_idx]
                * a_src
            ).sum(-1)
            +
            (
                h_dst[dst_idx]
                * a_dst
            ).sum(-1)
        )

        # Probability gate

        gate = (
            1.0
            +
            torch.tanh(
                self.w_g * p
                +
                self.b_g
            )
        )

        scores = F.leaky_relu(
            gate.unsqueeze(-1)
            * scores,
            negative_slope=0.2,
        )

        # Degree temperature

        alpha = scatter_softmax_temp(
            scores,
            dst_idx,
            N,
            degree,
        )

        alpha = self.drop(alpha)

        # Normalize probabilities

        p_sum = torch.zeros(
            N,
            device=x.device,
        )

        p_sum.scatter_add_(
            0,
            src_idx,
            p,
        )

        p_norm = p / (
            p_sum[src_idx]
            + 1e-10
        )

        # Double probability propagation

        messages = (
            h_src[src_idx]
            *
            (
                alpha
                *
                p_norm.unsqueeze(-1)
            ).unsqueeze(-1)
        )

        z = scatter_add_heads(
            messages,
            dst_idx,
            N,
        )

        z = z.view(N, -1)

        # Residual update

        return self.norm(
            x
            +
            F.elu(
                self.W_out(z)
            )
        )

# Pgat-Im

class PGAT_IM(nn.Module):
    """Multitask PGAT."""

    def __init__(
        self,
        in_channels=9,
        hidden_channels=32,
        embed_dim=32,
        heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        D = heads * hidden_channels

        # Input projection

        self.input_proj = nn.Sequential(

            nn.Linear(
                in_channels,
                D,
            ),

            nn.GELU(),

            nn.LayerNorm(D),

            nn.Dropout(
                dropout,
            ),
        )

        # Shared PGAT encoder

        self.convs = nn.ModuleList([
            PGATConv(
                D,
                hidden_channels,
                heads,
                dropout,
            )
            for _ in range(num_layers)
        ])

        # Shared embedding

        self.output_proj = nn.Sequential(

            nn.Linear(
                D + in_channels,
                64,
            ),

            nn.GELU(),

            nn.BatchNorm1d(
                64,
            ),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                64,
                embed_dim,
            ),
        )

        # IC Head

        self.ic_head = nn.Sequential(

            nn.Linear(
                embed_dim,
                16,
            ),

            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

        # LT Head

        self.lt_head = nn.Sequential(

            nn.Linear(
                embed_dim,
                16,
            ),

            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.constant_(
                        module.bias,
                        0.0,
                    )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
    ):

        N = x.size(0)

        # Degree computation

        degree = torch.zeros(
            N,
            dtype=torch.long,
            device=x.device,
        )

        degree.scatter_add_(
            0,
            edge_index[1],
            torch.ones(
                edge_index.size(1),
                dtype=torch.long,
                device=x.device,
            ),
        )

        degree.clamp_(min=1)

        # Input projection

        h = self.input_proj(x)

        # Shared encoder

        for conv in self.convs:

            h = conv(
                h,
                edge_index,
                edge_attr,
                degree,
            )

        # Shared embedding

        embed = self.output_proj(
            torch.cat(
                [h, x],
                dim=-1,
            )
        )

        # Task heads

        ic_pred = self.ic_head(
            embed
        ).squeeze(-1)

        lt_pred = self.lt_head(
            embed
        ).squeeze(-1)

        return (
            ic_pred,
            lt_pred,
        )

# Model Initialization

model = PGAT_IM(
    in_channels=9,
    hidden_channels=HIDDEN_DIM,
    embed_dim=EMBED_DIM,
    heads=HEADS,
    num_layers=NUM_LAYERS,
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
    f"\nTotal Parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable Parameters : "
    f"{trainable_params:,}"
)

# Frozen Vanilla Optimization

criterion_ic = nn.MSELoss()

criterion_lt = nn.MSELoss()

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=20,
    min_lr=1e-5,
)

# Early Stopping State

best_val_score = float("inf")

best_epoch = -1

patience_counter = 0

history = {
    "train_loss": [],
    "val_ic": [],
    "val_lt": [],
    "val_total": [],
    "learning_rate": [],
}

print("\nPGAT-IM Setup Complete.")
print("=" * 60)

# Scheduler, Early Stopping

print("\n" + "=" * 60)
print("Starting PGAT-IM Multitask Training")
print("=" * 60)

# Training Function

def train_one_epoch():
    """Train one epoch."""

    model.train()

    optimizer.zero_grad()

    # Forward

    ic_pred, lt_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

    # IC Loss

    train_ic_pred = ic_pred[
        data.train_mask
    ]

    train_ic_true = data.y_ic[
        data.train_mask
    ]

    loss_ic = criterion_ic(
        train_ic_pred,
        train_ic_true,
    )

    # LT Loss

    train_lt_pred = lt_pred[
        data.train_mask
    ]

    train_lt_true = data.y_lt[
        data.train_mask
    ]

    loss_lt = criterion_lt(
        train_lt_pred,
        train_lt_true,
    )

    # Joint Loss
    # No lambda:
    # Equal task importance

    total_loss = (
        loss_ic
        +
        loss_lt
    )

    # Backpropagation

    total_loss.backward()

    # Frozen Gradient Clipping

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    optimizer.step()

    return (
        total_loss.item(),
        loss_ic.item(),
        loss_lt.item(),
    )

# Validation Function

@torch.no_grad()
def validate():
    """Compute validation metrics."""

    model.eval()

    ic_pred, lt_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

    # IC Validation

    val_ic_pred = ic_pred[
        data.val_mask
    ]

    val_ic_true = data.y_ic[
        data.val_mask
    ]

    val_ic_mae = mean_absolute_error(
        val_ic_true.cpu().numpy(),
        val_ic_pred.cpu().numpy(),
    )

    # LT Validation

    val_lt_pred = lt_pred[
        data.val_mask
    ]

    val_lt_true = data.y_lt[
        data.val_mask
    ]

    val_lt_mae = mean_absolute_error(
        val_lt_true.cpu().numpy(),
        val_lt_pred.cpu().numpy(),
    )

    # Combined Validation Score
    # Used for:
    # Scheduler
    # Checkpoint
    # Early stopping

    combined_score = (
        val_ic_mae
        +
        val_lt_mae
    )

    return (
        val_ic_mae,
        val_lt_mae,
        combined_score,
    )

# Main Training Loop

for epoch in range(1, EPOCHS + 1):

    (
        train_total,
        train_ic,
        train_lt,
    ) = train_one_epoch()

    (
        val_ic_mae,
        val_lt_mae,
        val_score,
    ) = validate()

    # Scheduler Update

    scheduler.step(
        val_score
    )

    current_lr = optimizer.param_groups[0]["lr"]

    # Save History

    history["train_loss"].append(
        train_total
    )

    history["val_ic"].append(
        val_ic_mae
    )

    history["val_lt"].append(
        val_lt_mae
    )

    history["val_total"].append(
        val_score
    )

    history["learning_rate"].append(
        current_lr
    )

    # Best Checkpoint

    if val_score < best_val_score:

        best_val_score = val_score

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH,
        )

        patience_counter = 0

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
            f"Train: {train_total:.4f} "
            f"(IC={train_ic:.4f}, "
            f"LT={train_lt:.4f}) | "
            f"Val IC: {val_ic_mae:.4f} | "
            f"Val LT: {val_lt_mae:.4f} | "
            f"Val Total: {val_score:.4f} | "
            f"Best: {best_val_score:.4f} | "
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

# Training Complete

print("\n" + "=" * 60)
print("Training Complete")
print("=" * 60)

print(
    f"Best Epoch          : "
    f"{best_epoch}"
)

print(
    f"Best Validation "
    f"Score (IC+LT): "
    f"{best_val_score:.4f}"
)

print(
    f"Final Learning Rate : "
    f"{optimizer.param_groups[0]['lr']:.6f}"
)

print(
    f"\nCheckpoint Saved To:\n"
    f"{CHECKPOINT_PATH}"
)

# Restore Best Checkpoint

print(
    "\nRestoring Best "
    "Validation Checkpoint..."
)

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

# Leakage Audit

print("\nLeakage Audit")
print("-" * 40)

print(
    "✓ IC loss uses only train_mask"
)

print(
    "✓ LT loss uses only train_mask"
)

print(
    "✓ Validation uses only val_mask"
)

print(
    "✓ Scheduler uses "
    "IC+LT validation score"
)

print(
    "✓ Checkpoint selected using "
    "IC+LT validation score"
)

print(
    "✓ Test labels never accessed "
    "during training"
)

print(
    "✓ Final evaluation uses restored "
    "best checkpoint"
)

print("=" * 60)

# Results Saving, PGAT vs PGAT-IM

print("\n" + "=" * 60)
print("Final Evaluation")
print("=" * 60)

# Generic Evaluation Function

@torch.no_grad()
def evaluate_task(
    y_true_full,
    y_pred_full,
    task_name,
):
    """Returns:."""

    y_true = (
        y_true_full[data.test_mask]
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        y_pred_full[data.test_mask]
        .detach()
        .cpu()
        .numpy()
    )

    node_ids = (
        data.node_ids[data.test_mask]
        .detach()
        .cpu()
        .numpy()
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    elite = elite_diagnostics(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print(f"{task_name} TEST RESULTS")
    print("=" * 60)

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

        print(
            f"{key:<15}: "
            f"{metrics[key]:.4f}"
        )

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

        print(
            f"{key:<15}: "
            f"{metrics[key]:.4f}"
        )

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

        print(
            f"{key:<15}: "
            f"{elite[key]:.4f}"
        )

    return (
        metrics,
        elite,
        y_true,
        y_pred,
        node_ids,
    )

# Forward Pass

with torch.no_grad():

    ic_pred_full, lt_pred_full = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

# IC Evaluation

(
    ic_metrics,
    ic_elite,
    ic_true,
    ic_pred,
    ic_nodes,
) = evaluate_task(
    data.y_ic,
    ic_pred_full,
    "IC",
)

# LT Evaluation

(
    lt_metrics,
    lt_elite,
    lt_true,
    lt_pred,
    lt_nodes,
) = evaluate_task(
    data.y_lt,
    lt_pred_full,
    "LT",
)

# Prediction Export Function

def save_predictions(
    y_true,
    y_pred,
    node_ids,
    save_path,
):

    true_rank = (
        (-y_true)
        .argsort()
        .argsort()
        + 1
    )

    pred_rank = (
        (-y_pred)
        .argsort()
        .argsort()
        + 1
    )

    pred_df = pd.DataFrame({
        "node_id": node_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "split": "test",
        "true_rank": true_rank,
        "pred_rank": pred_rank,
    })

    pred_df.to_csv(
        save_path,
        index=False,
    )

# Save IC Predictions

save_predictions(
    ic_true,
    ic_pred,
    ic_nodes,
    IC_PREDICTIONS_PATH,
)

print(
    f"\nIC predictions saved to:\n"
    f"{IC_PREDICTIONS_PATH}"
)

# Save LT Predictions

save_predictions(
    lt_true,
    lt_pred,
    lt_nodes,
    LT_PREDICTIONS_PATH,
)

print(
    f"\nLT predictions saved to:\n"
    f"{LT_PREDICTIONS_PATH}"
)

# Save Result Files

def save_results(
    metrics,
    elite,
    save_path,
):

    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 60 + "\n"
        )

        f.write(
            "PGAT-IM Results\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Best Epoch: "
            f"{best_epoch}\n"
        )

        f.write(
            f"Best Validation Score: "
            f"{best_val_score:.6f}\n\n"
        )

        f.write(
            "Regression Metrics\n"
        )

        for k, v in metrics.items():

            f.write(
                f"{k}: "
                f"{v:.6f}\n"
            )

        f.write(
            "\nElite Diagnostics\n"
        )

        for k, v in elite.items():

            f.write(
                f"{k}: "
                f"{v:.6f}\n"
            )

save_results(
    ic_metrics,
    ic_elite,
    IC_RESULTS_PATH,
)

save_results(
    lt_metrics,
    lt_elite,
    LT_RESULTS_PATH,
)

print(
    f"\nIC results saved to:\n"
    f"{IC_RESULTS_PATH}"
)

print(
    f"\nLT results saved to:\n"
    f"{LT_RESULTS_PATH}"
)

# PGAT vs PGAT-IM Summary

print("\n" + "=" * 60)
print("PGAT vs PGAT-IM ABLATION")
print("=" * 60)

print("\nIC")

print(
    f"PGAT      MAE : 5.6255"
)

print(
    f"PGAT-IM   MAE : {ic_metrics['MAE']:.4f}"
)

print(
    f"PGAT      Top1 MAE : 77.9746"
)

print(
    f"PGAT-IM   Top1 MAE : {ic_elite['Top1_MAE']:.4f}"
)

print(
    f"PGAT      Spearman : 0.7008"
)

print(
    f"PGAT-IM   Spearman : {ic_metrics['Spearman']:.4f}"
)

print("\nLT")

print(
    f"PGAT      MAE : 8.7739"
)

print(
    f"PGAT-IM   MAE : {lt_metrics['MAE']:.4f}"
)

print(
    f"PGAT      Top1 MAE : 129.9698"
)

print(
    f"PGAT-IM   Top1 MAE : {lt_elite['Top1_MAE']:.4f}"
)

print(
    f"PGAT      Spearman : 0.6526"
)

print(
    f"PGAT-IM   Spearman : {lt_metrics['Spearman']:.4f}"
)

print("\n" + "=" * 60)
print("PGAT-IM Experiment Finished Successfully")
print("=" * 60)