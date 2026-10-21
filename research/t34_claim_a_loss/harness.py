"""Claim A loss-diagnostic harness: can a student flip decisive rows and keep the rest?

Inputs: a cache from :mod:`research.t34_claim_a_loss.cache` (excluded A5 v2
exports, B8 s3 as the previous student).  Everything is CPU, deterministic,
``torch.set_num_threads(threads)`` (default 2).

Two families of measurement:

1. **Free-Q probe** (no network): each decisive row's six Q values are free
   parameters initialised at ``q_prev`` and optimised directly under a loss.
   This is the most permissive case (no shared trunk, no generalisation) and
   isolates the loss's own fixed point.
2. **Network cells**: a fit on world-level folds (K folds, stratified on
   decisive-bearing worlds, ordered by sha256(world_id)).  Each cell is
   ``(loss, trunk, row rule, lr)``:

   * trunk ``trainable``: the full B8 network (Ego2sNet, 1.64M params) on the
     regenerated rasters;
   * trunk ``residual``: B8 frozen, ``Q = Q_B8 + R(h)`` with ``h`` B8's fused
     256-D feature and ``R`` a 256-256-6 MLP whose last layer is zero-initialised
     (step-0 Q is B8's exactly);
   * rule ``ratified``: the frozen 1:1 rule (every searched row + an equal
     uniform anchor sample per epoch, ratified ``sample_weights``, batch 1024,
     one update per 1024 rows);
   * rule ``streams``: separate decisive and anchor streams, ``n_dec`` +
     ``n_anchor`` rows per update, ``all_die_delay`` rows excluded.

   Metrics are recorded at fixed update counts on held-out worlds only.

Nothing here is gate data or evidence about a student.  It informs a pre-data
design choice (loss and update dose) only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .cache import KIND_ANCHOR, KIND_DECISIVE, KIND_DELAY, sha256_file
from .losses import LOSSES

FIELDS = (
    "decision",
    "length",
    "local",
    "global",
    "scalars",
    "q_prev",
    "h",
    "kind",
    "a_star",
    "forbidden",
    "tested",
    "survive",
    "student_action",
    "shadow_id",
)


@dataclass
class Bank:
    world_ids: list[str]
    world: np.ndarray
    arrays: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    @property
    def n(self) -> int:
        return int(self.world.shape[0])


def load_bank(cache_dir: Path, *, rasters: bool = True) -> Bank:
    """Concatenate every accepted world of a cache, verifying each npz digest."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["rejected"]:
        raise ValueError(f"cache has rejected worlds: {manifest['rejected']}")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in FIELDS}
    world_ids, world = [], []
    for index, receipt in enumerate(sorted(manifest["worlds"], key=lambda r: r["world_id"])):
        path = cache_dir / receipt["npz"]
        if sha256_file(path) != receipt["npz_sha256"]:
            raise ValueError(f"cache file changed: {path}")
        with np.load(path) as z:
            for key in FIELDS:
                if not rasters and key in ("local", "global", "scalars"):
                    continue
                parts[key].append(z[key])
            world_ids.append(str(z["world_id"]))
            world.append(np.full(len(z["kind"]), index, dtype=np.int64))
    arrays = {k: np.concatenate(v) for k, v in parts.items() if v}
    return Bank(world_ids, np.concatenate(world), arrays)


def assign_folds(world_ids: Sequence[str], bearing: Sequence[bool], k: int) -> np.ndarray:
    """World-level folds: within each stratum sort by sha256(id), deal round-robin."""

    def key(index: int) -> str:
        return hashlib.sha256(f"t34-claim-a-loss/v1:{world_ids[index]}".encode()).hexdigest()

    folds = np.zeros(len(world_ids), dtype=np.int64)
    for flag in (True, False):
        stratum = sorted((i for i in range(len(world_ids)) if bool(bearing[i]) is flag), key=key)
        for position, index in enumerate(stratum):
            folds[index] = position % k
    return folds


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class ResidualHead(nn.Module):
    """``Q = q_prev + W2 relu(W1 h)``, ``W2`` zero-initialised (step-0 Q == B8)."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.hidden = nn.Linear(256, width)
        self.out = nn.Linear(width, 6)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, q_prev: torch.Tensor) -> torch.Tensor:
        return q_prev + self.out(torch.relu(self.hidden(h)))


def make_model(trunk: str, checkpoint: Path | None, device: str = "cpu") -> nn.Module:
    if trunk == "residual":
        return ResidualHead().to(device)
    if trunk == "trainable":
        from src.model.ego2s_network import load_ego2s_checkpoint

        if checkpoint is None:
            raise ValueError("trainable trunk needs the B8 checkpoint")
        return load_ego2s_checkpoint(str(checkpoint), "cpu").to(device).train()
    raise ValueError(f"unknown trunk {trunk}")


def _t(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def model_q(
    model: nn.Module, trunk: str, bank: Bank, idx: np.ndarray, device: str = "cpu"
) -> torch.Tensor:
    if trunk == "residual":
        return model(_t(bank["h"][idx], device), _t(bank["q_prev"][idx], device))
    return model(
        _t(bank["local"][idx], device),
        _t(bank["global"][idx], device),
        _t(bank["scalars"][idx], device),
    )


def predict(
    model: nn.Module, trunk: str, bank: Bank, idx: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 2048):
            out.append(model_q(model, trunk, bank, idx[s : s + 2048], device).cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, 6), np.float32)


# ---------------------------------------------------------------------------
# batches and row rules
# ---------------------------------------------------------------------------
def batch_tensors(
    bank: Bank, idx: np.ndarray, weight: np.ndarray, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    kind = bank["kind"][idx]
    decisive = kind == KIND_DECISIVE
    action = np.where(decisive, bank["a_star"][idx], bank["q_prev"][idx].argmax(1))
    forbidden = bank["forbidden"][idx] & decisive[:, None]
    return {
        "q_prev": _t(bank["q_prev"][idx], device),
        "action_star": _t(action.astype(np.int64), device),
        "forbidden_mask": _t(forbidden, device),
        "sample_weight": _t(weight.astype(np.float32), device),
        "decisive": _t(decisive, device),
        "kl_anchor_only": _t(kind == KIND_ANCHOR, device),
    }


def ratified_epoch(
    bank: Bank, train_worlds: np.ndarray, rng: np.random.Generator, batch: int = 1024
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The frozen 1:1 rule with the ratified ``sample_weights`` (one epoch of batches)."""
    from research.t34_exit_b8_fit.data import sample_weights

    in_train = np.isin(bank.world, train_worlds)
    kind = bank["kind"]
    searched = np.flatnonzero(in_train & (kind != KIND_ANCHOR))
    anchors = np.flatnonzero(in_train & (kind == KIND_ANCHOR))
    picked = np.sort(rng.choice(anchors, size=len(searched), replace=False))
    selected = np.concatenate([searched, picked])
    rows = [
        {
            "round_age": 0,
            "decisive": bool(kind[i] == KIND_DECISIVE),
            "outcome_kind": "all_die_delay" if kind[i] == KIND_DELAY else None,
            "death_window": bool(kind[i] != KIND_ANCHOR),
            "death_window_id": int(bank["shadow_id"][i]),
            "world_id": bank.world_ids[int(bank.world[i])],
        }
        for i in selected
    ]
    weights = sample_weights(rows)
    order = rng.permutation(len(selected))
    return [
        (selected[order[s : s + batch]], weights[order[s : s + batch]])
        for s in range(0, len(selected), batch)
    ]


class StreamSampler:
    """Independent cycling permutations over train decisive rows and train anchors."""

    def __init__(
        self,
        bank: Bank,
        train_worlds: np.ndarray,
        n_dec: int,
        n_anchor: int,
        rng: np.random.Generator,
        decisive_weight: float,
    ) -> None:
        in_train = np.isin(bank.world, train_worlds)
        self.dec = np.flatnonzero(in_train & (bank["kind"] == KIND_DECISIVE))
        self.anc = np.flatnonzero(in_train & (bank["kind"] == KIND_ANCHOR))
        if not len(self.dec) or not len(self.anc):
            raise ValueError("training fold lacks decisive or anchor rows")
        self.n_dec, self.n_anchor, self.rng = n_dec, n_anchor, rng
        self.decisive_weight = decisive_weight
        self._queues = {"dec": np.zeros(0, np.int64), "anc": np.zeros(0, np.int64)}

    def _take(self, name: str, pool: np.ndarray, count: int) -> np.ndarray:
        out = []
        while count > 0:
            if not len(self._queues[name]):
                self._queues[name] = self.rng.permutation(pool)
            take = self._queues[name][:count]
            self._queues[name] = self._queues[name][len(take) :]
            out.append(take)
            count -= len(take)
        return np.concatenate(out)

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        dec = self._take("dec", self.dec, self.n_dec)
        anc = self._take("anc", self.anc, self.n_anchor)
        idx = np.concatenate([dec, anc])
        weight = np.concatenate(
            [np.full(len(dec), self.decisive_weight), np.ones(len(anc))]
        ).astype(np.float32)
        return idx, weight


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def decisive_metrics(q: np.ndarray, bank: Bank, idx: np.ndarray) -> dict[str, Any]:
    if not len(idx):
        return {"n": 0}
    pred = q.argmax(1)
    a_star = bank["a_star"][idx]
    bare = bank["student_action"][idx]
    rows = np.arange(len(idx))
    q_star = q[rows, a_star]
    others = q.copy()
    others[rows, a_star] = -np.inf
    margin = q_star - others.max(1)
    gap_bare = q_star - q[rows, bare]
    flip = pred == a_star
    worlds = bank.world[idx]
    per_world = [float(flip[worlds == w].mean()) for w in np.unique(worlds)]
    return {
        "n": int(len(idx)),
        "worlds": int(len(per_world)),
        "flip": float(flip.mean()),
        "flip_world_weighted": float(np.mean(per_world)),
        "argmax_bare": float((pred == bare).mean()),
        "argmax_not_forbidden": float((~bank["forbidden"][idx][rows, pred]).mean()),
        "argmax_known_survivor": float(
            (bank["tested"][idx][rows, pred] & bank["survive"][idx][rows, pred]).mean()
        ),
        "margin_pos": float((margin > 0).mean()),
        "margin_median": float(np.median(margin)),
        "gap_vs_bare_pos": float((gap_bare > 0).mean()),
        "gap_vs_bare_median": float(np.median(gap_bare)),
    }


def anchor_metrics(q: np.ndarray, bank: Bank, idx: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    if not len(idx):
        return {"n": 0}
    pred = q.argmax(1)
    prev = bank["q_prev"][idx].argmax(1)
    disagree = pred != prev
    bins = np.clip(np.searchsorted(edges, bank["length"][idx], side="right") - 1, 0, 9)
    deciles = [float(disagree[bins == b].mean()) if (bins == b).any() else 0.0 for b in range(10)]
    boost_prev = float((prev >= 3).mean())
    boost = float((pred >= 3).mean())
    return {
        "n": int(len(idx)),
        "agree": float(1.0 - disagree.mean()),
        "max_decile_disagree": float(max(deciles)),
        "decile_disagree": deciles,
        "boost_share": boost,
        "boost_share_prev": boost_prev,
        "boost_ratio": boost / boost_prev if boost_prev > 0 else None,
        "max_action_share": float(np.bincount(pred, minlength=6).max() / len(pred)),
    }


# ---------------------------------------------------------------------------
# free-Q probe
# ---------------------------------------------------------------------------
def free_q_probe(
    bank: Bank, loss: str, *, steps: int = 5000, lr: float = 1e-2, weight: float = 1.0
) -> dict[str, Any]:
    """Optimise each decisive row's own six Q values (no network) under ``loss``."""
    idx = np.flatnonzero(bank["kind"] == KIND_DECISIVE)
    batch = batch_tensors(bank, idx, np.full(len(idx), weight))
    q = torch.nn.Parameter(batch["q_prev"].clone())
    opt = torch.optim.Adam([q], lr=lr)
    fn = LOSSES[loss]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        fn(q, batch).backward()
        opt.step()
    out = decisive_metrics(q.detach().numpy(), bank, idx)
    out.update({"loss": loss, "steps": steps, "lr": lr, "weight": weight})
    return out


# ---------------------------------------------------------------------------
# network cells
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    name: str
    loss: str
    trunk: str
    rule: str = "streams"
    lr: float = 2.5e-5
    n_dec: int = 128
    n_anchor: int = 896
    epochs: int = 5  # ratified rule only
    checkpoints: tuple[int, ...] = (0, 8, 16, 32, 64, 128, 256)


@dataclass
class FoldResult:
    fold: int
    train_decisive: int
    heldout_decisive: int
    heldout_anchor: int
    heldout_worlds: list[int] = field(default_factory=list)
    curve: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


def run_fold(
    bank: Bank,
    folds: np.ndarray,
    fold: int,
    cell: Cell,
    checkpoint: Path | None,
    edges: np.ndarray,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[FoldResult, dict[int, np.ndarray]]:
    started = time.perf_counter()
    torch.manual_seed(seed + 1000 * fold)
    rng = np.random.default_rng(seed + 1000 * fold)
    train_worlds = np.flatnonzero(folds != fold)
    held_worlds = np.flatnonzero(folds == fold)
    in_held = np.isin(bank.world, held_worlds)
    in_train = ~in_held
    held_dec = np.flatnonzero(in_held & (bank["kind"] == KIND_DECISIVE))
    held_anc = np.flatnonzero(in_held & (bank["kind"] == KIND_ANCHOR))
    train_dec = np.flatnonzero(in_train & (bank["kind"] == KIND_DECISIVE))
    model = make_model(cell.trunk, checkpoint, device)
    opt = torch.optim.Adam(model.parameters(), lr=cell.lr)
    fn = LOSSES[cell.loss]
    result = FoldResult(
        fold, len(train_dec), len(held_dec), len(held_anc), [int(w) for w in held_worlds]
    )
    held_q: dict[int, np.ndarray] = {}

    def record(updates: int, rows_seen: int, dec_seen: int) -> None:
        qh = predict(model, cell.trunk, bank, held_dec, device)
        held_q[updates] = qh
        result.curve.append(
            {
                "updates": updates,
                "rows_seen": rows_seen,
                "decisive_seen": dec_seen,
                "decisive_passes": dec_seen / max(1, len(train_dec)),
                "heldout_decisive": decisive_metrics(qh, bank, held_dec),
                "train_decisive": decisive_metrics(
                    predict(model, cell.trunk, bank, train_dec, device), bank, train_dec
                ),
                "heldout_anchor": anchor_metrics(
                    predict(model, cell.trunk, bank, held_anc, device), bank, held_anc, edges
                ),
            }
        )

    def step(idx: np.ndarray, weight: np.ndarray) -> None:
        q = model_q(model, cell.trunk, bank, idx, device)
        loss = fn(q, batch_tensors(bank, idx, weight, device))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    updates = rows_seen = dec_seen = 0
    record(0, 0, 0)
    if cell.rule == "ratified":
        for _ in range(cell.epochs):
            for idx, weight in ratified_epoch(bank, train_worlds, rng):
                step(idx, weight)
                updates += 1
                rows_seen += len(idx)
                dec_seen += int((bank["kind"][idx] == KIND_DECISIVE).sum())
            record(updates, rows_seen, dec_seen)
    elif cell.rule == "streams":
        sampler = StreamSampler(
            bank, train_worlds, cell.n_dec, cell.n_anchor, rng, cell.n_anchor / cell.n_dec
        )
        targets = sorted(set(cell.checkpoints) - {0})
        while targets:
            idx, weight = sampler.next()
            step(idx, weight)
            updates += 1
            rows_seen += len(idx)
            dec_seen += cell.n_dec
            if updates == targets[0]:
                record(updates, rows_seen, dec_seen)
                targets.pop(0)
    else:
        raise ValueError(f"unknown rule {cell.rule}")
    result.seconds = time.perf_counter() - started
    return result, held_q


def pooled(bank: Bank, fold_results: list[tuple[FoldResult, dict[int, np.ndarray]]], edges):
    """Pool held-out decisive predictions over folds at each common update count."""
    common = set.intersection(*(set(q) for _, q in fold_results))
    out = []
    for updates in sorted(common):
        idx = np.concatenate(
            [
                np.flatnonzero(
                    (bank["kind"] == KIND_DECISIVE) & np.isin(bank.world, r.heldout_worlds)
                )
                for r, _ in fold_results
            ]
        )
        q = np.concatenate([qs[updates] for _, qs in fold_results])
        anchors = [
            next(c for c in r.curve if c["updates"] == updates)["heldout_anchor"]
            for r, _ in fold_results
        ]
        # Empty held-out folds carry only {"n": 0}, with no measured metrics.
        anchors = [a for a in anchors if a["n"] > 0]
        n = sum(a["n"] for a in anchors)
        out.append(
            {
                "updates": updates,
                "heldout_decisive": decisive_metrics(q, bank, idx),
                "heldout_anchor_agree": (
                    sum(a["agree"] * a["n"] for a in anchors) / n if n else None
                ),
                "heldout_anchor_max_decile_disagree": max(
                    (a["max_decile_disagree"] for a in anchors), default=None
                ),
                "heldout_boost_ratio_max": max(
                    ((a["boost_ratio"] or 0.0) for a in anchors), default=None
                ),
                "decisive_passes_mean": float(
                    np.mean(
                        [
                            next(c for c in r.curve if c["updates"] == updates)["decisive_passes"]
                            for r, _ in fold_results
                        ]
                    )
                ),
            }
        )
    return out


def run_cell(
    bank: Bank,
    folds: np.ndarray,
    cell: Cell,
    checkpoint: Path | None,
    k: int,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    lengths = bank["length"][bank["kind"] == KIND_ANCHOR]
    edges = np.quantile(lengths, np.linspace(0, 1, 11)[:-1])
    results = []
    for fold in range(k):
        result, held_q = run_fold(bank, folds, fold, cell, checkpoint, edges, seed, device)
        results.append((result, held_q))
        print(json.dumps({"cell": cell.name, "fold": fold, "seconds": round(result.seconds, 1)}))
    return {
        "cell": asdict(cell),
        "folds": [asdict(r) for r, _ in results],
        "pooled": pooled(bank, results, edges),
    }


LONG = (0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
# D0 grid: every candidate production dose (512/1024/2048/4096 updates) is a checkpoint.
D0_CONTROL = (0, 32, 128, 512, 1024, 2048)
D0_CORRECTED = (0, 32, 128, 512, 1024, 2048, 4096)

DEFAULT_CELLS = {
    # baseline: the frozen recipe exactly (loss, 1:1 rule, weights, lr, 5 epochs)
    "frozen_recipe": Cell("frozen_recipe", "frozen", "trainable", rule="ratified", epochs=5),
    # L3 2x2 under the stream rule and an explicit dose
    "frozen_trainable": Cell("frozen_trainable", "frozen", "trainable", checkpoints=D0_CONTROL),
    "split_trainable": Cell("split_trainable", "split", "trainable", checkpoints=D0_CORRECTED),
    "frozen_residual": Cell(
        "frozen_residual", "frozen", "residual", lr=1e-4, checkpoints=D0_CONTROL
    ),
    "split_residual": Cell(
        "split_residual", "split", "residual", lr=1e-4, checkpoints=D0_CORRECTED
    ),
    # coefficient / step-size variants of the corrected objective
    "split_trainable_lr1e-4": Cell(
        "split_trainable_lr1e-4", "split", "trainable", lr=1e-4, checkpoints=D0_CORRECTED
    ),
    "split_tau2_trainable": Cell(
        "split_tau2_trainable", "split_tau2", "trainable", checkpoints=D0_CORRECTED
    ),
    "split_tau2_residual": Cell(
        "split_tau2_residual", "split_tau2", "residual", lr=1e-4, checkpoints=D0_CORRECTED
    ),
    "split_residual_lr1e-3": Cell(
        "split_residual_lr1e-3", "split", "residual", lr=1e-3, checkpoints=D0_CORRECTED
    ),
    # unregistered extras (exploration only; never read by the D0 rule)
    "frozen_tau2_trainable": Cell("frozen_tau2_trainable", "frozen_tau2", "trainable"),
    "frozen_tau2_residual": Cell(
        "frozen_tau2_residual", "frozen_tau2", "residual", lr=1e-4, checkpoints=LONG
    ),
}

# The registered D0 cell set (spec 2026-10-21 §4): three frozen-loss controls and six
# corrected-objective cells.  ``decide.py`` refuses a report that lacks any of them.
D0_CELLS = (
    "frozen_recipe",
    "frozen_trainable",
    "frozen_residual",
    "split_trainable",
    "split_trainable_lr1e-4",
    "split_tau2_trainable",
    "split_residual",
    "split_residual_lr1e-3",
    "split_tau2_residual",
)
D0_FREE_Q = ("frozen", "frozen_tau2", "split", "split_tau2")
D0_FOLDS = 3
D0_FREE_Q_STEPS = 5000


def run_report(
    cache: Path,
    checkpoint: Path,
    out: Path,
    *,
    cells: Sequence[str] = D0_CELLS,
    free_q: Sequence[str] = D0_FREE_Q,
    free_q_steps: int = D0_FREE_Q_STEPS,
    folds_k: int = D0_FOLDS,
    threads: int = 2,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Free-Q probes then every cell; ``out`` is written create-only at the end."""
    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        # fp32 everywhere (no TF32) so the diagnostic is comparable to the CPU tests
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    names = list(cells)
    unknown = [c for c in names if c not in DEFAULT_CELLS]
    if unknown:
        raise ValueError(f"unknown cells {unknown}")
    needs_rasters = any(DEFAULT_CELLS[c].trunk == "trainable" for c in names)
    bank = load_bank(cache, rasters=needs_rasters)
    bearing = [
        bool(((bank.world == w) & (bank["kind"] == KIND_DECISIVE)).any())
        for w in range(len(bank.world_ids))
    ]
    folds = assign_folds(bank.world_ids, bearing, folds_k)
    report: dict[str, Any] = {
        "schema": "t34-claim-a-loss-diagnostics/v1",
        "claim": "D0_DESIGN_DIAGNOSTIC_EXCLUDED_A5_NOT_EVIDENCE",
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "threads": threads,
        "device": device,
        "torch": torch.__version__,
        "cells_requested": names,
        "folds": folds_k,
        "seed": seed,
        "bank": {
            "worlds": len(bank.world_ids),
            "decisive_bearing_worlds": int(sum(bearing)),
            "rows": bank.n,
            "decisive": int((bank["kind"] == KIND_DECISIVE).sum()),
            "delay": int((bank["kind"] == KIND_DELAY).sum()),
            "anchors": int((bank["kind"] == KIND_ANCHOR).sum()),
        },
        "fold_of_world": {w: int(f) for w, f in zip(bank.world_ids, folds)},
        "free_q": [],
        "cells": [],
    }
    if any(DEFAULT_CELLS[name].rule == "streams" for name in names):
        unsupported = []
        for fold in range(folds_k):
            train = folds[bank.world] != fold
            decisive = int((train & (bank["kind"] == KIND_DECISIVE)).sum())
            anchors = int((train & (bank["kind"] == KIND_ANCHOR)).sum())
            if not decisive or not anchors:
                unsupported.append(
                    {"fold": fold, "train_decisive": decisive, "train_anchor": anchors}
                )
        if unsupported:
            # A tiny fixture can lack stream training support. Preserve the fixed
            # folds and cells; an incomplete report makes the frozen rule refuse.
            report["status"] = "PRECONDITION_FAILED"
            report["reason"] = "training fold lacks decisive or anchor rows"
            report["unsupported_training_folds"] = unsupported
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x") as stream:
                json.dump(report, stream, indent=1, sort_keys=True)
                stream.write("\n")
            return report
    for loss in free_q:
        started = time.perf_counter()
        report["free_q"].append(
            free_q_probe(bank, loss, steps=free_q_steps)
            | {"seconds": time.perf_counter() - started}
        )
        print(json.dumps(report["free_q"][-1]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.json")
    for name in names:
        report["cells"].append(
            run_cell(bank, folds, DEFAULT_CELLS[name], checkpoint, folds_k, seed, device)
        )
        # partial receipt after each cell (overwritten), final create-only below
        partial.write_text(json.dumps(report, indent=1))
    with out.open("x") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
        stream.write("\n")
    partial.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", default=",".join(D0_CELLS))
    parser.add_argument("--free-q", default=",".join(D0_FREE_Q))
    parser.add_argument("--free-q-steps", type=int, default=D0_FREE_Q_STEPS)
    parser.add_argument("--folds", type=int, default=D0_FOLDS)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run_report(
        args.cache,
        args.checkpoint,
        args.out,
        cells=[c for c in args.cells.split(",") if c],
        free_q=[x for x in args.free_q.split(",") if x],
        free_q_steps=args.free_q_steps,
        folds_k=args.folds,
        threads=args.threads,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
