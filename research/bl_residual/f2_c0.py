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

from research.a100_a2.run import _validate, fit_unit

_DOSES = (25, 100)
_SEEDS = (20261022, 20261023)
_FOLDS = (0, 1, 2, 3, 4)


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
    sums = {dose: np.zeros((n, 6), dtype=np.float32) for dose in doses}
    counts = {dose: np.zeros(n, dtype=np.int8) for dose in doses}

    for dose in doses:
        if dose not in _DOSES:
            raise ValueError("F2 permits only the registered 25% and 100% doses")
        for seed in seeds:
            for fold in _FOLDS:
                # Literal reuse is intentional.  A frozen-embedding cache
                # changes the CPU convolution batch grouping and is therefore
                # not bit-identical to A2's registered unit runner.
                unit = fit_unit(
                    data,
                    str(checkpoint),
                    "c0",
                    "d300",
                    dose,
                    seed,
                    fold,
                    device,
                    epochs,
                    batch_size,
                )
                heldout = unit["row_index"]
                sums[dose][heldout] += unit["p_d300"]
                counts[dose][heldout] += 1
    result: dict[int, np.ndarray] = {}
    for dose in doses:
        if not np.array_equal(counts[dose], np.full(n, len(seeds), dtype=np.int8)):
            raise RuntimeError("F2 C0 did not produce exactly one prediction per seed and row")
        result[dose] = sums[dose] / len(seeds)
    return result


def fit_c0(data: dict[str, np.ndarray], checkpoint: Path) -> dict[int, np.ndarray]:
    """Return two-seed averaged, five-fold OOF C0 D300 probabilities for F2.

    This literally reuses A2's unit runner so its frozen B8 forwards, world-level
    selection, D300 loss, two seeds, five folds, 20 epochs, batch size 256,
    AdamW, clipping, and inference batches remain bit-identical.  It has no
    filesystem output.
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
