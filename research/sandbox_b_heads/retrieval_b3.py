"""Train-only cosine kNN diagnostics for Sandbox B-3.

The retrieval results are development diagnostics over frozen B8 representations.
They are empirical retrieval ceilings for this particular data and feature map,
not a proof of a Bayes-optimal ceiling.  Validation outcomes are only read after
all neighbor scores have been produced from train rows and labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import torch


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert an array-like table column without silently moving a CUDA tensor to CPU."""
    result = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return result.to(device=device, dtype=dtype) if dtype is not None else result.to(device=device)


def _exact(table: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
    """Return the exact-label indicator for either archive or runtime table forms."""
    status = table["status"]
    if isinstance(status, torch.Tensor):
        if status.dtype == torch.bool:
            return status.to(device=device)
        if status.dtype.is_floating_point or status.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            return status.to(device=device, dtype=torch.bool)
        status = status.detach().cpu().numpy()
    array = np.asarray(status)
    if array.dtype == np.bool_:
        return torch.as_tensor(array, device=device)
    return torch.as_tensor(array == "EXACT", device=device)


def _selectable(table: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
    mask = _tensor(table["action_mask"], device=device, dtype=torch.bool).clone()
    # Match serving/B-2's legal fallback: malformed all-false masks still expose
    # the three non-boost actions, rather than producing an undefined argmax.
    mask[~mask.any(1), :3] = True
    return mask


def _known_legal(table: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
    return (
        _exact(table, device=device)
        & _tensor(table["legal"], device=device, dtype=torch.bool)
        & _selectable(table, device=device)
    )


def _decisive_and_ordinary(
    table: Mapping[str, Any], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    known = _known_legal(table, device=device)
    died = _tensor(table["died"], device=device, dtype=torch.bool)
    base = _tensor(table["bare"], device=device, dtype=torch.long)
    rows = torch.arange(len(base), device=device)
    decisive = known[rows, base] & died[rows, base] & (known & ~died).any(1)
    ordinary = _tensor(table["ordinary"], device=device, dtype=torch.bool)
    return decisive, ordinary


def _auc(positive: torch.Tensor, negative: torch.Tensor) -> float | None:
    """Mann-Whitney AUROC with half credit for ties, no sklearn dependency."""
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


def _life_macro_retention(
    retained: torch.Tensor, decisive: torch.Tensor, life_id: Any
) -> float | None:
    if not decisive.any():
        return None
    lives = life_id if isinstance(life_id, torch.Tensor) else torch.as_tensor(np.asarray(life_id))
    lives = lives.to(device=decisive.device)
    values = [
        retained[decisive & (lives == life)].float().mean()
        for life in torch.unique(lives[decisive])
    ]
    return float(torch.stack(values).mean()) if values else None


def action_oracle_retention(
    probabilities: torch.Tensor, table: Mapping[str, Any]
) -> dict[str, float | int | None]:
    """Score the selected retrieved action against exact validation labels.

    The word ``oracle`` here refers only to the action-labelled development
    diagnostic: it asks whether the retrieval score ranks a true surviving
    action first on decisive rows.  It is not a deployment or world evaluation.
    """
    device = probabilities.device
    selectable = _selectable(table, device=device)
    known = _known_legal(table, device=device)
    died = _tensor(table["died"], device=device, dtype=torch.bool)
    base = _tensor(table["bare"], device=device, dtype=torch.long)
    rows = torch.arange(len(base), device=device)
    selected = probabilities.masked_fill(~selectable, -1.0).argmax(1)
    decisive, _ = _decisive_and_ordinary(table, device=device)
    selected_survives = known[rows, selected] & ~died[rows, selected]
    retained = decisive & selected_survives
    return {
        "action_oracle_rescue_retention": (
            float(retained[decisive].float().mean()) if decisive.any() else None
        ),
        "action_oracle_life_macro_rescue_retention": _life_macro_retention(
            retained, decisive, table["life_id"]
        ),
        "action_oracle_decisive_count": int(decisive.sum()),
    }


def retrieval(
    train_features: torch.Tensor | np.ndarray,
    val_features: torch.Tensor | np.ndarray,
    train_table: Mapping[str, Any],
    val_table: Mapping[str, Any],
    k_values: Sequence[int] = (5, 15, 50),
    *,
    chunk_size: int = 1024,
    feature_name: str = "phi",
    progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Produce per-action exact/legal survival estimates from train-only neighbors.

    Queries retrieve only rows with the same B8 served/base action.  For every
    query action, only train examples whose candidate action is exact, legal and
    selectable vote.  Unknown labels are omitted.  An action with no known
    neighbors gets probability zero.  The cosine matrix is evaluated in chunks
    to keep the A100 diagnostic bounded.
    """
    if not k_values or any(type(k) is not int or k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integer k values")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    device = (
        train_features.device if isinstance(train_features, torch.Tensor) else torch.device("cpu")
    )
    train = _tensor(train_features, device=device, dtype=torch.float32)
    validation = _tensor(val_features, device=device, dtype=torch.float32)
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("train_features and val_features must be rank-2 with a shared width")
    if len(train) != len(train_table["bare"]) or len(validation) != len(val_table["bare"]):
        raise ValueError("feature/table row count mismatch")
    train_base = _tensor(train_table["bare"], device=device, dtype=torch.long)
    val_base = _tensor(val_table["bare"], device=device, dtype=torch.long)
    if (
        (train_base < 0).any()
        or (train_base >= 6).any()
        or (val_base < 0).any()
        or (val_base >= 6).any()
    ):
        raise ValueError("bare actions must be in [0, 6)")
    train_known = _known_legal(train_table, device=device)
    train_survives = train_known & ~_tensor(train_table["died"], device=device, dtype=torch.bool)
    val_selectable = _selectable(val_table, device=device)
    train_decisive, _ = _decisive_and_ordinary(train_table, device=device)
    normalized_train = torch.nn.functional.normalize(train, p=2, dim=1, eps=1e-12)
    normalized_val = torch.nn.functional.normalize(validation, p=2, dim=1, eps=1e-12)
    ordered_k = tuple(sorted(set(k_values)))
    estimates = {
        k: torch.zeros((len(validation), 6), dtype=torch.float32, device=device) for k in ordered_k
    }
    decisive_scores = {
        k: torch.zeros(len(validation), dtype=torch.float32, device=device) for k in ordered_k
    }

    # Retrieval pools are split only by the frozen served action.  Each action's
    # labels stay separate inside the pool, preventing unknown action outcomes
    # from turning into implicit deaths.
    for base_action in range(6):
        query_indices = torch.where(val_base == base_action)[0]
        candidate_indices = torch.where(train_base == base_action)[0]
        if not len(query_indices) or not len(candidate_indices):
            continue
        candidate_features = normalized_train[candidate_indices]
        candidate_known = train_known[candidate_indices]
        candidate_survives = train_survives[candidate_indices]
        candidate_decisive = train_decisive[candidate_indices]
        for start in range(0, len(query_indices), chunk_size):
            target = query_indices[start : start + chunk_size]
            similarities = normalized_val[target] @ candidate_features.T
            # A single top-k ranking is sufficient for every requested k.
            max_k = min(max(ordered_k), len(candidate_indices))
            _, neighbors = torch.topk(similarities, max_k, dim=1, largest=True, sorted=True)
            neighbor_known = candidate_known[neighbors]
            neighbor_survives = candidate_survives[neighbors]
            for k in ordered_k:
                take = min(k, len(candidate_indices))
                decisive_scores[k][target] = candidate_decisive[neighbors[:, :take]].float().mean(1)
                # Same p(x,a) estimate applies to each action's independently
                # labelled neighbor subset.  Vectorize all six action columns.
                # (The action-specific known masks select distinct vote sets.)
                for action in range(6):
                    action_known = neighbor_known[:, :take, action]
                    action_count = action_known.sum(1)
                    action_num = neighbor_survives[:, :take, action].sum(1).to(torch.float32)
                    estimates[k][target, action] = torch.where(
                        action_count > 0,
                        action_num / action_count.to(torch.float32),
                        torch.zeros_like(action_num),
                    )
            if progress is not None:
                progress()

    decisive, ordinary = _decisive_and_ordinary(val_table, device=device)
    diagnostics: dict[int, dict[str, Any]] = {}
    for k, probabilities in estimates.items():
        # The score is the train-neighborhood fraction of decisive examples,
        # rather than a validation-outcome-calibrated threshold.  It asks whether
        # frozen feature space separates base-death rescue situations from
        # ordinary rows.
        score = decisive_scores[k]
        ordinary_scores = score[ordinary]
        max_ordinary = float(ordinary_scores.max()) if len(ordinary_scores) else None
        diagnostics[k] = {
            "feature_name": feature_name,
            "k": k,
            "decisive_fraction": float(decisive.float().mean()) if len(decisive) else None,
            "decisive_count": int(decisive.sum()),
            "ordinary_count": int(ordinary.sum()),
            "decisive_score_mean": float(score[decisive].mean()) if decisive.any() else None,
            "ordinary_score_mean": float(score[ordinary].mean()) if ordinary.any() else None,
            "decisive_vs_ordinary_auroc": _auc(score[decisive], ordinary_scores),
            "strict_fpr0_threshold": max_ordinary,
            "decisive_tpr_strict_above_max_ordinary": (
                float((score[decisive] > max_ordinary).float().mean())
                if decisive.any() and max_ordinary is not None
                else None
            ),
            **action_oracle_retention(probabilities, val_table),
        }
    masked = {k: values.masked_fill(~val_selectable, 0.0) for k, values in estimates.items()}
    return {
        "feature_name": feature_name,
        "probabilities": masked,
        "diagnostics": diagnostics,
        "caveat": "Empirical frozen-feature retrieval diagnostic; not a proven Bayes bound.",
    }
