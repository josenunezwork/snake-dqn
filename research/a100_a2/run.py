"""Deterministic single-unit A2 fit entrypoint.

No filesystem data discovery occurs here: the transport layer supplies the
already validated, normalized table.  A caller gets held-out predictions only.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from src.model.ego2s_network import load_ego2s_checkpoint

from .model import HORIZONS, A2Predictor, EncoderName, TargetName
from .protocol import NAMESPACE

_REQUIRED = {
    "local",
    "global",
    "scalars",
    "world",
    "fold",
    "legal",
    "died",
    "alive_steps",
    "observed",
    "decisive",
    "a_star",
    "q",
}


def _validate(data: Mapping[str, np.ndarray]) -> int:
    missing = _REQUIRED.difference(data)
    if missing:
        raise ValueError(f"A2 table missing keys: {sorted(missing)}")
    n = len(data["world"])
    expected = {
        "local": (n, 7, 31, 31),
        "global": (n, 4, 37, 37),
        "scalars": (n, 60),
        "fold": (n,),
        "legal": (n, 6),
        "died": (n, 6),
        "alive_steps": (n, 6),
        "observed": (n, 6),
        "decisive": (n,),
        "a_star": (n,),
        "q": (n, 6),
    }
    for key, shape in expected.items():
        if tuple(data[key].shape) != shape:
            raise ValueError(f"{key} must have shape {shape}, got {data[key].shape}")
    if data["local"].dtype != np.uint8 or data["global"].dtype != np.uint8:
        raise ValueError("local and global inputs must be normalized uint8 rasters")
    world = np.asarray(data["world"]).astype(str)
    folds = np.asarray(data["fold"])
    for value in np.unique(world):
        if np.unique(folds[world == value]).size != 1:
            raise ValueError(f"parent world spans frozen folds: {value}")
    legal, observed, died = (
        np.asarray(data[key]).astype(bool) for key in ("legal", "observed", "died")
    )
    alive = np.asarray(data["alive_steps"])
    if np.any(observed & ~legal):
        raise ValueError("an unlegal branch cannot be observed")
    known = legal & observed
    if np.any(alive[known] < 0) or np.any(alive[known] > HORIZONS[-1]):
        raise ValueError("observed alive_steps must be inside the registered horizon")
    if np.any(died[known] & (alive[known] >= HORIZONS[-1])):
        raise ValueError("a recorded D300 death must have alive_steps below 300")
    if np.any(~died[known] & (alive[known] != HORIZONS[-1])):
        raise ValueError("a recorded survivor must be right-censored at 300")
    return n


def _ordered_training_rows(data: Mapping[str, np.ndarray], fold: int, dose: int) -> np.ndarray:
    if dose not in (25, 50, 100):
        raise ValueError("dose must be 25, 50, or 100")
    world = np.asarray(data["world"]).astype(str)
    train_worlds = sorted(set(world[np.asarray(data["fold"]) != fold].tolist()))
    ranked = sorted(
        train_worlds,
        key=lambda value: hashlib.sha256(f"{NAMESPACE}{value}".encode()).digest(),
    )
    keep = set(ranked[: int(np.ceil(len(ranked) * dose / 100.0))])
    return np.flatnonzero(np.isin(world, list(keep)) & (np.asarray(data["fold"]) != fold))


def _batch(
    data: Mapping[str, np.ndarray], rows: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.from_numpy(np.ascontiguousarray(data[key][rows])).to(device)
        for key in ("local", "global", "scalars", "legal", "died", "alive_steps", "observed")
    )


def _hazard_loss(
    logits: torch.Tensor,
    legal: torch.Tensor,
    died: torch.Tensor,
    alive: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Conditional discrete-hazard likelihood through a death or censor endpoint."""
    valid = legal.bool() & observed.bool()
    event_step = alive.long() + 1
    horizons = torch.as_tensor(HORIZONS, device=logits.device).view(1, 1, -1)
    event_bucket = torch.searchsorted(
        torch.as_tensor(HORIZONS, device=logits.device), event_step.clamp(max=HORIZONS[-1])
    )
    labels = torch.zeros_like(logits)
    death = died.bool() & valid
    labels.scatter_(
        2, event_bucket.unsqueeze(-1).clamp_max(len(HORIZONS) - 1), death.unsqueeze(-1).float()
    )
    # Death at interval k observes every earlier survival interval and event k.
    event_mask = torch.arange(len(HORIZONS), device=logits.device).view(
        1, 1, -1
    ) <= event_bucket.unsqueeze(-1)
    censor_mask = horizons <= alive.long().unsqueeze(-1)
    mask = valid.unsqueeze(-1) & torch.where(died.bool().unsqueeze(-1), event_mask, censor_mask)
    per_branch = (
        F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none") * mask
    ).sum(dim=-1)
    return per_branch[valid].mean() if valid.any() else logits.sum() * 0.0


def _loss(
    logits: torch.Tensor,
    target: TargetName,
    legal: torch.Tensor,
    died: torch.Tensor,
    alive: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    valid = legal.bool() & observed.bool()
    if target == "hazard":
        return _hazard_loss(logits, legal, died, alive, observed)
    terms = F.binary_cross_entropy_with_logits(logits.float(), died.float(), reduction="none")
    return terms[valid].mean() if valid.any() else logits.sum() * 0.0


def _choose_action(
    death_probability: torch.Tensor, legal: torch.Tensor, observed: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Apply the registered A5 agreement choice on the device.

    T5 has no recorded B8 Q (its values are NaN), so an all-nonfinite Q row
    falls back to maximum predicted survival.  A5 rows with finite Q retain
    the preregistered survivor-threshold then-Q ranking.
    """
    mask = legal.bool() & observed.bool()
    survival = 1.0 - death_probability
    survivors = mask & (survival >= 0.5)
    q_eligible = survivors & torch.isfinite(q)
    by_q = q.masked_fill(~q_eligible, -torch.inf).argmax(dim=-1)
    by_survival = survival.masked_fill(~mask, -torch.inf).argmax(dim=-1)
    return torch.where(q_eligible.any(dim=-1), by_q, by_survival)


def fit_unit(
    data: Mapping[str, np.ndarray],
    checkpoint: str | None,
    encoder: EncoderName,
    target: TargetName,
    dose: int,
    seed: int,
    fold: int,
    device: str,
    epochs: int = 20,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Fit one fixed A2 unit and return predictions for only its held-out fold.

    ``p_hazards`` is cumulative death probability at (30,60,100,200,300), not
    per-interval hazards.  It is absent for a D300 head because that arm has no
    registered earlier-horizon score.
    """
    _validate(data)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device)
    base = load_ego2s_checkpoint(checkpoint, dev) if encoder in ("c0", "s_b8") else None
    model = A2Predictor(encoder, target, base).to(dev).train()
    train_rows = _ordered_training_rows(data, int(fold), int(dose))
    if not len(train_rows):
        raise ValueError("training selection is empty")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    last_heartbeat = time.monotonic()
    batches_completed = 0
    for _epoch in range(epochs):
        order = train_rows[torch.randperm(len(train_rows), generator=generator).numpy()]
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            local, glob, scalars, legal, died, alive, observed = _batch(data, rows, dev)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(local, glob, scalars), target, legal, died, alive, observed)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite A2 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if any(
                not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise FloatingPointError("nonfinite A2 gradient")
            optimizer.step()
            batches_completed += 1
            if time.monotonic() - last_heartbeat >= 90.0:
                print(
                    json.dumps(
                        {
                            "stage": "a2-fit",
                            "epoch": _epoch + 1,
                            "batches_completed": batches_completed,
                        }
                    ),
                    flush=True,
                )
                last_heartbeat = time.monotonic()
    heldout = np.flatnonzero(np.asarray(data["fold"]) == fold)
    model.eval()
    probabilities: list[np.ndarray] = []
    choices_out: list[np.ndarray] = []
    horizons: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(heldout), batch_size):
            rows = heldout[start : start + batch_size]
            local, glob, scalars, legal, _died, _alive, observed = _batch(data, rows, dev)
            logits = model(local, glob, scalars)
            if target == "hazard":
                survival = torch.cumprod(1.0 - torch.sigmoid(logits), dim=-1)
                cumulative = 1.0 - survival
                p = cumulative[:, :, -1]
                horizons.append(cumulative.cpu().numpy().astype(np.float32))
            else:
                p = torch.sigmoid(logits)
            # The registered A5 agreement rule: among legal observed actions,
            # select a predicted survivor by recorded B8 Q; otherwise choose
            # maximum predicted survival. ``argmax`` is intentionally on the
            # six-action device tensor so canary receipts exercise this path.
            q = torch.from_numpy(np.ascontiguousarray(data["q"][rows])).to(dev).float()
            choices = _choose_action(p, legal, observed, q)
            probabilities.append(p.cpu().numpy().astype(np.float32))
            choices_out.append(choices.cpu().numpy().astype(np.int64))
    p_d300 = np.concatenate(probabilities, axis=0)
    result: dict[str, np.ndarray] = {
        "row_index": heldout.astype(np.int64),
        "world": np.asarray(data["world"])[heldout],
        "fold": np.asarray(data["fold"])[heldout],
        "p_d300": p_d300,
        "choice": np.concatenate(choices_out, axis=0),
        "legal": np.asarray(data["legal"])[heldout],
        "died": np.asarray(data["died"])[heldout],
        "alive_steps": np.asarray(data["alive_steps"])[heldout],
        "observed": np.asarray(data["observed"])[heldout],
        "decisive": np.asarray(data["decisive"])[heldout],
        "a_star": np.asarray(data["a_star"])[heldout],
        "q": np.asarray(data["q"])[heldout],
    }
    if horizons:
        result["p_hazards"] = np.concatenate(horizons, axis=0)
        if not np.array_equal(result["p_d300"], result["p_hazards"][:, :, -1]):
            raise RuntimeError("hazard D300 prediction must equal its final cumulative horizon")
    return result
