# Training
# Configuration

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

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score,
)

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

CHECKPOINT_PATH = (
    "checkpoints/best_pgatv1_ic.pt"
)

RESULTS_PATH = (
    "results/ic/pgatv1_ic_results.txt"
)

PREDICTIONS_PATH = (
    "results/ic/pgatv1_ic_predictions.csv"
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results/ic", exist_ok=True)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IN_CHANNELS = 11

HIDDEN_CHANNELS = 32

EMBED_DIM = 32

HEADS = 4

NUM_LAYERS = 2

DROPOUT = 0.1

BASE_LR = 1e-3

WEIGHT_DECAY = 1e-4

LLRD = 0.8

WARMUP = 20

EPOCHS = 1000

PATIENCE = 100

HUBER_DELTA = 0.1

def precision_at_k(y_true, y_pred, k=10):

    k = min(k, len(y_true))

    true_topk = np.argsort(y_true)[-k:]
    pred_topk = np.argsort(y_pred)[-k:]

    overlap = len(
        set(true_topk).intersection(
            set(pred_topk)
        )
    )

    return overlap / k

def ndcg_at_k(y_true, y_pred, k=10):

    k = min(k, len(y_true))

    return ndcg_score(
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        k=k,
    )

def compute_metrics(y_true, y_pred):

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

    ndcg10 = ndcg_at_k(
        y_true,
        y_pred,
        k=10,
    )

    precision10 = precision_at_k(
        y_true,
        y_pred,
        k=10,
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

print("Loading PyG Dataset")

data = torch.load(
    DATA_PATH,
    weights_only=False,
)

print(data)

print("\nDataset Summary")
print("Nodes:", data.num_nodes)
print("Edges:", data.edge_index.shape[1])
print("Features:", data.x.shape[1])

print(
    "Train:",
    data.train_mask.sum().item()
)

print(
    "Validation:",
    data.val_mask.sum().item()
)

print(
    "Test:",
    data.test_mask.sum().item()
)

print("\nIC Labels")

print(
    "IC NaNs:",
    torch.isnan(data.y_ic).sum().item()
)

print(
    "IC Labeled:",
    data.num_nodes -
    torch.isnan(data.y_ic).sum().item()
)

data = data.to(DEVICE)

print("\nUsing Device:", DEVICE)

print("Dataset Ready")

def _scatter_softmax(logits, dst_idx, N):

    H = logits.size(1)

    idx = dst_idx.unsqueeze(-1).expand(-1, H)

    maxv = torch.full(
        (N, H),
        float("-inf"),
        device=logits.device,
    )

    maxv.scatter_reduce_(
        0,
        idx,
        logits,
        reduce="amax",
        include_self=True,
    )

    ex = torch.exp(
        torch.clamp(
            logits - maxv[dst_idx],
            min=-50,
            max=50,
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

def _scatter_add(src, dst_idx, N):

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

class PGATConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.heads = heads
        self.out_c = out_channels

        D = heads * out_channels

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

        self.a_src = nn.Parameter(
            torch.empty(
                heads,
                out_channels,
            )
        )

        self.a_dst = nn.Parameter(
            torch.empty(
                heads,
                out_channels,
            )
        )

        self.w_g = nn.Parameter(
            torch.tensor(1.0)
        )

        self.b_g = nn.Parameter(
            torch.tensor(0.0)
        )

        self.W_out = nn.Linear(
            D,
            D,
        )

        self.norm = nn.LayerNorm(D)

        self.drop = nn.Dropout(dropout)

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
    ):

        N = x.size(0)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        p = edge_attr.squeeze(-1)

        h_s = self.W_src(x).view(
            N,
            self.heads,
            self.out_c,
        )

        h_d = self.W_dst(x).view(
            N,
            self.heads,
            self.out_c,
        )

        a_src = self.a_src.unsqueeze(0)
        a_dst = self.a_dst.unsqueeze(0)

        s = (
            (h_s[src_idx] * a_src).sum(-1)
            +
            (h_d[dst_idx] * a_dst).sum(-1)
        )

        g = 1.0 + torch.tanh(
            self.w_g * p +
            self.b_g
        )

        e = F.leaky_relu(
            g.unsqueeze(-1) * s,
            negative_slope=0.2,
        )

        alpha = _scatter_softmax(
            e,
            dst_idx,
            N,
        )

        alpha = self.drop(alpha)

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
            p_sum[src_idx] + 1e-10
        )

        msg = h_s[src_idx] * (
            alpha *
            p_norm.unsqueeze(-1)
        ).unsqueeze(-1)

        z = _scatter_add(
            msg,
            dst_idx,
            N,
        )

        z = z.view(
            N,
            -1,
        )

        return self.norm(
            x +
            F.elu(
                self.W_out(z)
            )
        )

class PGATv1_IM(nn.Module):
    """PGAT model."""

    def __init__(
        self,
        in_channels=11,
        hidden_channels=32,
        embed_dim=32,
        heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        D = heads * hidden_channels

        self.input_proj = nn.Sequential(
            nn.Linear(
                in_channels,
                D,
            ),

            nn.GELU(),

            nn.LayerNorm(D),

            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([
            PGATConv(
                in_channels=D,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(

            nn.Linear(
                D + in_channels,
                64,
            ),

            nn.GELU(),

            nn.BatchNorm1d(64),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                embed_dim,
            ),
        )

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

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

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

        h = self.input_proj(x)

        for conv in self.convs:

            h = conv(
                h,
                edge_index,
                edge_attr,
            )

        embed = self.output_proj(
            torch.cat(
                [h, x],
                dim=-1,
            )
        )

        ic_pred = self.ic_head(embed)

        lt_pred = self.lt_head(embed)

        return (
            embed,
            ic_pred.squeeze(-1),
            lt_pred.squeeze(-1),
        )

model = PGATv1_IM(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    embed_dim=EMBED_DIM,
    heads=HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print("\nModel Architecture")

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


print("PGAT-v1 Architecture Ready")

criterion = nn.HuberLoss(
    delta=HUBER_DELTA,
)

def compute_masked_loss(
    ic_pred,
    y_ic,
    mask,
):
    """Computes IC loss only on labeled nodes."""

    ic_mask = (
        mask &
        (~torch.isnan(y_ic))
    )

    if ic_mask.sum() == 0:

        return torch.tensor(
            0.0,
            device=DEVICE,
        )

    loss = criterion(

        ic_pred[ic_mask],

        y_ic[ic_mask],
    )

    return loss

def build_optimizer(
    model,
    base_lr=1e-3,
    llrd=0.8,
    wd=1e-4,
    epochs=1000,
    warmup=20,
):

    num_layers = len(
        model.convs
    )

    groups = []

    groups.append({

        "params":
            model.input_proj.parameters(),

        "lr":
            base_lr *
            (llrd ** num_layers),
    })

    for k, conv in enumerate(
        model.convs
    ):

        groups.append({

            "params":
                conv.parameters(),

            "lr":
                base_lr *
                (
                    llrd **
                    (
                        num_layers - 1 - k
                    )
                ),
        })

    groups.append({

        "params":
            list(
                model.output_proj.parameters()
            )
            +
            list(
                model.ic_head.parameters()
            )
            +
            list(
                model.lt_head.parameters()
            ),

        "lr":
            base_lr,
    })

    optimizer = torch.optim.AdamW(

        groups,

        weight_decay=wd,
    )

    def lr_fn(epoch):

        if epoch < warmup:

            return (
                epoch /
                max(warmup, 1)
            )

        t = (
            epoch - warmup
        ) / max(
            epochs - warmup,
            1,
        )

        return (
            0.5 *
            (
                1.0 +
                math.cos(
                    math.pi * t
                )
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(

        optimizer,

        lr_lambda=lr_fn,
    )

    return (
        optimizer,
        scheduler,
    )

optimizer, scheduler = build_optimizer(

    model,

    base_lr=BASE_LR,

    llrd=LLRD,

    wd=WEIGHT_DECAY,

    epochs=EPOCHS,

    warmup=WARMUP,
)

print("\nTraining Configuration")

print(
    f"Loss              : "
    f"Huber(delta={HUBER_DELTA})"
)

print(
    "Task              : "
    "IC Only"
)

print(
    "Multitask Loss    : "
    "NO"
)

print(
    "Optimizer         : "
    "AdamW + LLRD"
)

print(
    f"Base LR           : "
    f"{BASE_LR}"
)

print(
    f"LLRD              : "
    f"{LLRD}"
)

print(
    f"Warmup            : "
    f"{WARMUP}"
)

print(
    "Scheduler         : "
    "Cosine"
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
    f"Weight Decay      : "
    f"{WEIGHT_DECAY}"
)

print(
    f"Dropout           : "
    f"{DROPOUT}"
)

print("Optimizer Setup Complete")
def train_one_epoch():

    model.train()

    optimizer.zero_grad()

    _, ic_pred, lt_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

    loss = compute_masked_loss(
        ic_pred,
        data.y_ic,
        data.train_mask,
    )

    loss.backward()

    optimizer.step()

    return loss.item()

# Validation

@torch.no_grad()
def evaluate_validation():

    model.eval()

    _, ic_pred, lt_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

    mask = (
        data.val_mask &
        (~torch.isnan(data.y_ic))
    )

    y_true = (
        data.y_ic[mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        ic_pred[mask]
        .cpu()
        .numpy()
    )

    val_mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    return val_mae

print("Starting PGAT-v1 IC Training")

best_val_mae = float("inf")

best_epoch = 0

patience_counter = 0

best_state = None

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()

    val_mae = evaluate_validation()

    scheduler.step()

    current_lr = (
        optimizer.param_groups[-1]["lr"]
    )

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch

        patience_counter = 0

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state,
            CHECKPOINT_PATH,
        )

    else:

        patience_counter += 1

    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:04d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Best: {best_val_mae:.4f} | "
            f"LR: {current_lr:.6f}"
        )

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        print(
            f"No improvement for "
            f"{PATIENCE} epochs."
        )

        break


print("Training Complete")

print(
    f"Best Epoch         : "
    f"{best_epoch}"
)

print(
    f"Best Validation MAE: "
    f"{best_val_mae:.4f}"
)

print(
    f"Checkpoint Saved   : "
    f"{CHECKPOINT_PATH}"
)

print("\nLoading best checkpoint")

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
)

print("Best checkpoint restored.")


@torch.no_grad()
def evaluate_test():
    """Evaluate model on test dataset."""

    model.eval()

    _, ic_pred, lt_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
    )

    test_mask = (
        data.test_mask &
        (~torch.isnan(data.y_ic))
    )

    y_true = (
        data.y_ic[test_mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        ic_pred[test_mask]
        .cpu()
        .numpy()
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    return (
        metrics,
        y_true,
        y_pred,
        test_mask,
    )


print("Final Evaluation on IC Test Set")

metrics, y_true, y_pred, test_mask = evaluate_test()

node_ids = np.where(
    test_mask.cpu().numpy()
)[0]

prediction_df = pd.DataFrame({

    "node_id": node_ids,

    "true_ic": y_true,

    "pred_ic": y_pred,

})

prediction_df.to_csv(

    PREDICTIONS_PATH,

    index=False,
)

print(
    f"\nPredictions saved to: "
    f"{PREDICTIONS_PATH}"
)

print("\nTest Metrics")

print(
    f"Test MAE          : "
    f"{metrics['MAE']:.4f}"
)

print(
    f"Test RMSE         : "
    f"{metrics['RMSE']:.4f}"
)

print(
    f"Test R²           : "
    f"{metrics['R2']:.4f}"
)

print(
    f"Spearman          : "
    f"{metrics['Spearman']:.4f}"
)

print(
    f"NDCG@10           : "
    f"{metrics['NDCG@10']:.4f}"
)

print(
    f"Precision@10      : "
    f"{metrics['Precision@10']:.4f}"
)

print("\nTop-10 Ranking Analysis")
print("-" * 40)

true_top10_idx = np.argsort(
    y_true
)[-10:][::-1]

print("\nGround Truth Top-10 IC Influence:")

for rank, idx in enumerate(

    true_top10_idx,

    start=1,
):
    print(
        f"{rank:2d}. "
        f"{y_true[idx]:8.4f}"
    )

pred_top10_idx = np.argsort(
    y_pred
)[-10:][::-1]

print("\nPredicted Top-10 IC Influence:")

for rank, idx in enumerate(

    pred_top10_idx,

    start=1,
):
    print(
        f"{rank:2d}. "
        f"{y_pred[idx]:8.4f}"
    )

true_set = set(true_top10_idx)

pred_set = set(pred_top10_idx)

overlap = len(
    true_set.intersection(pred_set)
)

print("\nTop-10 Overlap")

print(
    f"Common Nodes       : "
    f"{overlap}/10"
)

print(
    f"Precision@10       : "
    f"{overlap/10:.4f}"
)

print("\nRanking Consistency")
print(
    f"Spearman Correlation : "
    f"{metrics['Spearman']:.4f}"
)

if metrics["Spearman"] >= 0.90:

    interpretation = "Excellent"

elif metrics["Spearman"] >= 0.80:

    interpretation = "Very Strong"

elif metrics["Spearman"] >= 0.70:

    interpretation = "Strong"

elif metrics["Spearman"] >= 0.50:

    interpretation = "Moderate"

else:

    interpretation = "Weak"

print(
    f"Interpretation       : "
    f"{interpretation}"
)


PGAT_V2_RESULTS = {
    "MAE": 9.6599,
    "RMSE": 12.8027,
    "R2": 0.8115,
    "Spearman": 0.8885,
    "NDCG@10": 0.9322,
    "Precision@10": 0.7000,
}

print("PGAT-v1 IC vs PGAT-v2-MT IC")
print(
    f"{'Metric':<15}"
    f"{'PGAT-v2':>12}"
    f"{'PGAT-v1':>12}"
    f"{'Δ':>12}"
)


comparison_metrics = [
    ("MAE", "↓"),
    ("RMSE", "↓"),
    ("R2", "↑"),
    ("Spearman", "↑"),
    ("NDCG@10", "↑"),
    ("Precision@10", "↑"),
]

for metric_name, direction in comparison_metrics:

    v2_value = PGAT_V2_RESULTS[metric_name]

    v1_value = metrics[metric_name]

    if direction == "↓":
        delta = v2_value - v1_value
    else:
        delta = v1_value - v2_value

    print(
        f"{metric_name:<15}"
        f"{v2_value:>12.4f}"
        f"{v1_value:>12.4f}"
        f"{delta:>12.4f}"
    )

print("\nMultitask Learning Impact")

mae_change = (
    (PGAT_V2_RESULTS["MAE"] - metrics["MAE"])
    /
    PGAT_V2_RESULTS["MAE"]
) * 100

rmse_change = (
    (PGAT_V2_RESULTS["RMSE"] - metrics["RMSE"])
    /
    PGAT_V2_RESULTS["RMSE"]
) * 100

spearman_change = (
    metrics["Spearman"]
    -
    PGAT_V2_RESULTS["Spearman"]
)

precision_change = (
    metrics["Precision@10"]
    -
    PGAT_V2_RESULTS["Precision@10"]
)

print(
    f"MAE Change (%)      : "
    f"{mae_change:.2f}"
)

print(
    f"RMSE Change (%)     : "
    f"{rmse_change:.2f}"
)

print(
    f"Spearman Change     : "
    f"{spearman_change:.4f}"
)

print(
    f"P@10 Change         : "
    f"{precision_change:.4f}"
)


print("PGAT-v1 IC SUMMARY")

print(
    "Model               : PGAT-v1"
)

print(
    "Task                : IC Influence Estimation"
)

print(
    "Supervision         : IC Only"
)

print(
    f"Best Epoch          : "
    f"{best_epoch}"
)

print(
    f"Best Validation MAE : "
    f"{best_val_mae:.4f}"
)

print("\nArchitecture")

print("Probability Gate    : YES")
print("Temperature Scaling : NO")
print("Double Probability  : YES")
print("Residual            : YES")
print("LayerNorm           : YES")
print("Multitask Learning  : NO")
print("LLRD                : YES")
print("Cosine Scheduler    : YES")

print("\nIC Test Performance")

print(
    f"MAE                 : "
    f"{metrics['MAE']:.4f}"
)

print(
    f"RMSE                : "
    f"{metrics['RMSE']:.4f}"
)

print(
    f"R²                  : "
    f"{metrics['R2']:.4f}"
)

print(
    f"Spearman            : "
    f"{metrics['Spearman']:.4f}"
)

print(
    f"NDCG@10             : "
    f"{metrics['NDCG@10']:.4f}"
)

print(
    f"Precision@10        : "
    f"{metrics['Precision@10']:.4f}"
)

with open(RESULTS_PATH, "w") as f:

    f.write(
        "PGAT-v1 IC FINAL RESULTS\n"
    )

    f.write(
        f"Best Epoch: {best_epoch}\n"
    )

    f.write(
        f"Best Validation MAE: "
        f"{best_val_mae:.4f}\n\n"
    )

    f.write("Architecture\n")


    f.write(
        "Probability Gate    : YES\n"
    )

    f.write(
        "Temperature Scaling : NO\n"
    )

    f.write(
        "Double Probability  : YES\n"
    )

    f.write(
        "Residual            : YES\n"
    )

    f.write(
        "Multitask Learning  : NO\n"
    )

    f.write(
        "LLRD                : YES\n"
    )

    f.write(
        "Cosine Scheduler    : YES\n\n"
    )

    f.write("IC Test Metrics\n")

    for key, value in metrics.items():

        f.write(
            f"{key:<15}: "
            f"{value:.4f}\n"
        )

    f.write("\nPGAT-v2 Comparison\n")

    for metric_name, direction in comparison_metrics:

        v2_value = PGAT_V2_RESULTS[metric_name]

        v1_value = metrics[metric_name]

        if direction == "↓":
            delta = v2_value - v1_value
        else:
            delta = v1_value - v2_value

        f.write(
            f"{metric_name:<15}"
            f"V2={v2_value:.4f} | "
            f"V1={v1_value:.4f} | "
            f"del={delta:.4f}\n"
        )


print(
    "\nResults saved to:",
    RESULTS_PATH,
)

print(
    "\nPredictions saved to:",
    PREDICTIONS_PATH,
)


print(
    "PGAT-v1 IC Training Finished"
)
