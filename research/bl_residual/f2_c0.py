"""Sealed F2 fallback for the registered A2 C0 D300 fits.

This is deliberately a small, in-memory adapter around the A2 recipe.  It
does not discover inputs, write predictions, calculate an F2 statistic, or
emit progress.  The F2 disposition runner is the only code allowed to open
its returned probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from research.a100_a2.model import A2Predictor
from research.a100_a2.run import _loss, _ordered_training_rows, _validate
from src.model.ego2s_network import load_ego2s_checkpoint

_DOSES = (25, 100)
_SEEDS = (20261022, 20261023)
_FOLDS = (0, 1, 2, 3, 4)


def _features(
    data: Mapping[str, np.ndarray], model: A2Predictor, device: torch.device, batch_size: int
) -> torch.Tensor:
    """Compute C0's frozen B8 embedding once without touching the head RNG."""
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(data["world"]), batch_size):
            end = start + batch_size
            local = torch.from_numpy(np.ascontiguousarray(data["local"][start:end])).to(device)
            glob = torch.from_numpy(np.ascontiguousarray(data["global"][start:end])).to(device)
            scalars = torch.from_numpy(np.ascontiguousarray(data["scalars"][start:end])).to(device)
            chunks.append(model.features(local, glob, scalars).detach().cpu())
    return torch.cat(chunks, dim=0)


def _fit_c0(
    data: Mapping[str, np.ndarray],
    checkpoint: Path,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    doses: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Implementation seam with injectable tiny-test parameters.

    The public entry point fixes every argument to the F2/A2 registration.
    """
    n = _validate(data)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if set(np.unique(np.asarray(data["fold"])).tolist()) != set(_FOLDS):
        raise ValueError("F2 requires the five frozen A2 folds 0..4")
    if not checkpoint.is_file():
        raise ValueError("F2 C0 checkpoint is missing")
    dev = torch.device(device)
    cached: torch.Tensor | None = None
    sums = {dose: np.zeros((n, 6), dtype=np.float64) for dose in doses}
    counts = {dose: np.zeros(n, dtype=np.int8) for dose in doses}

    for dose in doses:
        if dose not in _DOSES:
            raise ValueError("F2 permits only the registered 25% and 100% doses")
        for seed in seeds:
            for fold in _FOLDS:
                # Keep this ordering identical to ``a100_a2.run.fit_unit``.
                # Caching starts only after the original model construction, so
                # it cannot consume random numbers used by the linear head.
                torch.manual_seed(seed)
                np.random.seed(seed)
                base = load_ego2s_checkpoint(str(checkpoint), dev)
                model = A2Predictor("c0", "d300", base).to(dev).train()
                if cached is None:
                    cached = _features(data, model, dev, batch_size)
                train_rows = _ordered_training_rows(data, fold, dose)
                if not len(train_rows):
                    raise ValueError("F2 C0 training selection is empty")
                optimizer = torch.optim.AdamW(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    lr=1e-3,
                    weight_decay=1e-4,
                )
                generator = torch.Generator(device="cpu").manual_seed(seed)
                for _epoch in range(epochs):
                    order = train_rows[torch.randperm(len(train_rows), generator=generator).numpy()]
                    for start in range(0, len(order), batch_size):
                        rows = order[start : start + batch_size]
                        embedding = cached[rows].to(dev)
                        legal = torch.from_numpy(np.ascontiguousarray(data["legal"][rows])).to(dev)
                        died = torch.from_numpy(np.ascontiguousarray(data["died"][rows])).to(dev)
                        alive = torch.from_numpy(
                            np.ascontiguousarray(data["alive_steps"][rows])
                        ).to(dev)
                        observed = torch.from_numpy(
                            np.ascontiguousarray(data["observed"][rows])
                        ).to(dev)
                        optimizer.zero_grad(set_to_none=True)
                        loss = _loss(model.head(embedding), "d300", legal, died, alive, observed)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite A2 loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        if any(
                            not torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        ):
                            raise FloatingPointError("nonfinite A2 gradient")
                        optimizer.step()
                heldout = np.flatnonzero(np.asarray(data["fold"]) == fold)
                with torch.inference_mode():
                    probabilities = torch.sigmoid(model.head(cached[heldout].to(dev))).cpu().numpy()
                sums[dose][heldout] += probabilities.astype(np.float64, copy=False)
                counts[dose][heldout] += 1
    result: dict[int, np.ndarray] = {}
    for dose in doses:
        if not np.array_equal(counts[dose], np.full(n, len(seeds), dtype=np.int8)):
            raise RuntimeError("F2 C0 did not produce exactly one prediction per seed and row")
        result[dose] = (sums[dose] / len(seeds)).astype(np.float32)
    return result


def fit_c0(data: dict[str, np.ndarray], checkpoint: Path) -> dict[int, np.ndarray]:
    """Return two-seed averaged, five-fold OOF C0 D300 probabilities for F2.

    This retains A2's exact frozen B8 encoder, world-level nested dose selection,
    D300 loss, two seeds, five folds, 20 epochs, batch size 256, AdamW, and
    clipping.  It is intentionally silent and has no filesystem output.
    """
    return _fit_c0(
        data,
        Path(checkpoint),
        device="cpu",
        epochs=20,
        batch_size=256,
        doses=_DOSES,
        seeds=_SEEDS,
    )
