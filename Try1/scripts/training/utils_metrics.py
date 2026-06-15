# ============================================================
# utils_metrics.py
# Chunk 1: Imports and Ranking Metrics
# ============================================================

import numpy as np
import torch

from scipy.stats import spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score,
)


# ============================================================
# Precision@K
# ============================================================

def precision_at_k(y_true, y_pred, k=10):
    """
    Precision@K based on overlap of top-K nodes.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    k : int

    Returns
    -------
    float
    """

    k = min(k, len(y_true))

    true_topk = np.argsort(y_true)[-k:]
    pred_topk = np.argsort(y_pred)[-k:]

    overlap = len(
        set(true_topk).intersection(
            set(pred_topk)
        )
    )

    return overlap / k


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(y_true, y_pred, k=10):
    """
    NDCG@K using sklearn.
    """

    k = min(k, len(y_true))

    return ndcg_score(
        y_true.reshape(1, -1),
        y_pred.reshape(1, -1),
        k=k
    )

# ============================================================
# Complete Evaluation Metrics
# ============================================================

def compute_metrics(y_true, y_pred):
    """
    Compute all regression and ranking metrics.

    Returns
    -------
    dict
    """

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

# ============================================================
# Masked Huber Loss
# ============================================================

def masked_huber_loss(
    predictions,
    targets,
    mask,
    loss_fn,
):
    """
    Compute Huber loss only on masked nodes.
    """

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


# ============================================================
# Extract Predictions
# ============================================================

@torch.no_grad()
def get_mask_predictions(
    predictions,
    targets,
    mask,
):
    """
    Convert masked predictions to numpy.
    """

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

# ============================================================
# Save Predictions
# ============================================================

def save_predictions(
    path,
    y_true,
    y_pred,
):
    """
    Save predictions as CSV.

    Columns:
        y_true,y_pred
    """

    import pandas as pd

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Predictions saved to: {path}"
    )