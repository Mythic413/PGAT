# ============================================================
# train_pgat_ic.py
# Try2 Vanilla Benchmark
# Chunk 1: Imports, Reproducibility, Configuration,
#           Dataset Loading, Metrics Utilities
# ============================================================

import os
import copy
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ============================================================
# Metrics Imports
# ============================================================

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
    "best_pgat_ic_vanilla.pt"
)

RESULTS_PATH = (
    "Try2/results/ic/"
    "pgat_ic_vanilla_results.txt"
)

PREDICTIONS_PATH = (
    "Try2/results/ic/"
    "pgat_ic_vanilla_predictions.csv"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Frozen Vanilla Hyperparameters
# ============================================================

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
print("PGAT-IC Vanilla Benchmark")
print("=" * 60)

print("\nLoading Frozen Dataset...")
print("-" * 60)

data = torch.load(
    DATA_PATH,
    weights_only=False,
)

print(data)


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

valid = ~torch.isnan(
    data.y_ic
)

print(
    f"NaN Labels     : "
    f"{num_nan}"
)

print(
    f"Labeled Nodes  : "
    f"{valid.sum().item()}"
)

print(
    f"Mean Spread    : "
    f"{data.y_ic[valid].mean().item():.4f}"
)

print(
    f"Max Spread     : "
    f"{data.y_ic[valid].max().item():.4f}"
)


# ============================================================
# Metric Utilities
# ============================================================

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

    return len(
        true_topk.intersection(pred_topk)
    ) / k


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

    return len(
        true_topk.intersection(pred_topk)
    ) / len(true_topk)


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
        true_topk.intersection(pred_topk)
    )


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

    idx = np.argsort(y_true)[::-1][:10]
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

    idx = np.argsort(y_true)[::-1][:10]
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


# ============================================================
# Move Dataset to Device
# ============================================================

data = data.to(DEVICE)

print("\nUsing Device :", DEVICE)

print("\nDataset Ready.")
print("=" * 60)

# ============================================================
# Chunk 2: Final Vanilla PGAT-IC Architecture
# ============================================================

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# Scatter Utilities
# ============================================================

def scatter_softmax_temp(
    logits,
    dst_idx,
    N,
    degree,
):
    """
    Degree-temperature scaled softmax.

    Architectural contribution:
        tau = sqrt(degree)

    Prevents hub nodes from dominating
    attention normalization.
    """

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
    """
    Multi-head scatter aggregation.
    """

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


# ============================================================
# PGAT Convolution
# ============================================================

class PGATConv(nn.Module):
    """
    Final PGAT Layer.

    Preserved architectural innovations:

        Separate W_src / W_dst
        Separate a_src / a_dst
        Probability Gate
        Degree Temperature
        Double Probability Scaling
        Residual Update
        LayerNorm
    """

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

        # ----------------------------------------------------
        # Source / Destination Projections
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Separate Attention Vectors
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Probability Gate
        # ----------------------------------------------------

        self.w_g = nn.Parameter(
            torch.tensor(1.0)
        )

        self.b_g = nn.Parameter(
            torch.tensor(0.0)
        )

        # ----------------------------------------------------
        # Output Projection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Node Projections
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Attention Scores
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Probability Gate
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Degree Temperature Softmax
        # ----------------------------------------------------

        alpha = scatter_softmax_temp(
            scores,
            dst_idx,
            N,
            degree,
        )

        alpha = self.drop(alpha)

        # ----------------------------------------------------
        # Probability Normalization
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Double Probability Scaling
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Residual Update
        # ----------------------------------------------------

        return self.norm(
            x
            +
            F.elu(
                self.W_out(z)
            )
        )


# ============================================================
# Final Vanilla PGAT-IC
# ============================================================

class PGAT_IC(nn.Module):
    """
    Publication-grade PGAT.

    Fair comparison version.

    Preserved:
        PGAT innovations

    Removed:
        Multi-task heads
        Huber loss
        LLRD
        Cosine scheduler
        Warmup
    """

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

        # ----------------------------------------------------
        # Input Projection
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PGAT Layers
        # ----------------------------------------------------

        self.convs = nn.ModuleList([
            PGATConv(
                D,
                hidden_channels,
                heads,
                dropout,
            )
            for _ in range(num_layers)
        ])

        # ----------------------------------------------------
        # Output Projection
        #
        # Raw feature skip preserved
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IC Head
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Degree Computation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Input Projection
        # ----------------------------------------------------

        h = self.input_proj(x)

        # ----------------------------------------------------
        # PGAT Layers
        # ----------------------------------------------------

        for conv in self.convs:

            h = conv(
                h,
                edge_index,
                edge_attr,
                degree,
            )

        # ----------------------------------------------------
        # Raw Feature Skip
        # ----------------------------------------------------

        embed = self.output_proj(
            torch.cat(
                [h, x],
                dim=-1,
            )
        )

        pred = self.ic_head(
            embed
        )

        return pred.squeeze(-1)


# ============================================================
# Model Initialization
# ============================================================

model = PGAT_IC(
    in_channels=9,
    hidden_channels=HIDDEN_DIM,
    embed_dim=EMBED_DIM,
    heads=HEADS,
    num_layers=NUM_LAYERS,
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
# Frozen Vanilla Optimization
# ============================================================

criterion = nn.MSELoss()

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


# ============================================================
# Early Stopping
# ============================================================

best_val_mae = float("inf")

best_epoch = -1

patience_counter = 0

history = {
    "train_loss": [],
    "val_mae": [],
    "learning_rate": [],
}


print("\nPGAT Setup Complete.")
print("=" * 60)

# ============================================================
# Chunk 3: Training Loop, Validation,
#           Scheduler, Early Stopping
# ============================================================

print("\n" + "=" * 60)
print("Starting Vanilla PGAT Training (IC)")
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
        data.edge_attr,
    )

    # --------------------------------------------------------
    # Train Loss
    #
    # Frozen Rule:
    # Only train_mask
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
        data.edge_attr,
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
    #
    # Frozen Rule:
    # Validation MAE only
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
    # Frozen Rule:
    # Best validation MAE
    # --------------------------------------------------------

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch

        torch.save(
            model.state_dict(),
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
    # Frozen Rule:
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

    model.eval()

    predictions = model(
        data.x,
        data.edge_index,
        data.edge_attr,
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
# Regression Metrics
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

    print(
        f"{key:<15}: "
        f"{test_metrics[key]:.4f}"
    )


# ============================================================
# Ranking Metrics
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
    set(true_top10)
    - set(pred_top10)
)

recovered = sorted(
    set(pred_top10)
    - set(true_top10)
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

true_rank = (
    (-y_true_test)
    .argsort()
    .argsort()
    + 1
)

pred_rank = (
    (-y_pred_test)
    .argsort()
    .argsort()
    + 1
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
print("VANILLA PGAT IC SUMMARY")
print("=" * 60)

print("Model             : PGAT")
print("Task              : IC")
print("Benchmark Track   : Vanilla")
print("Target            : y_ic")

print(
    f"Heads             : "
    f"{HEADS}"
)

print(
    f"Hidden Dimension  : "
    f"{HIDDEN_DIM}"
)

print(
    f"PGAT Layers       : "
    f"{NUM_LAYERS}"
)

print(
    f"Embed Dimension   : "
    f"{EMBED_DIM}"
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


# ============================================================
# Test Performance Summary
# ============================================================

print("\nTest Performance")
print("-" * 40)

for key in regression_keys:

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
        "VANILLA PGAT IC RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
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
        "Regression Metrics\n"
    )

    for key in regression_keys:

        f.write(
            f"{key}: "
            f"{test_metrics[key]:.6f}\n"
        )

    f.write("\nRanking Metrics\n")

    for key in ranking_keys:

        f.write(
            f"{key}: "
            f"{test_metrics[key]:.6f}\n"
        )

    f.write("\nElite Diagnostics\n")

    for key in elite_keys:

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