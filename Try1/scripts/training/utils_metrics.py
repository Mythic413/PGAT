import numpy as np
import torch

from scipy.stats import spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score,
)

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
        k=k
    )

def compute_metrics(y_true, y_pred):
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

def masked_huber_loss(
    predictions,
    targets,
    mask,
    loss_fn,
):

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

@torch.no_grad()
def get_mask_predictions(
    predictions,
    targets,
    mask,
):

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

def save_predictions(
    path,
    y_true,
    y_pred,
):

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