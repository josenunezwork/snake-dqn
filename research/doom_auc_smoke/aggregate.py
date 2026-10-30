"""The registered Mac-only, create-once scoring look over complete predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.doom_auc_smoke.data import load_gzip
from research.doom_auc_smoke.evidence import ARMS, build_ledger, digest, json_once, load_unit
from research.doom_auc_smoke.prelook import KINDS, attach_kinds, read_prelook


class AUC:
    """Tie-aware weighted Mann-Whitney AUC; weights resample parent worlds."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray, worlds: np.ndarray) -> None:
        if not (
            scores.ndim == labels.ndim == worlds.ndim == 1
            and len(scores) == len(labels) == len(worlds)
        ):
            raise ValueError("unaligned AUC events")
        if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("nonfinite score/nonbinary doom")
        order = np.argsort(scores, kind="stable")
        scores = scores[order]
        self.labels, self.worlds = labels[order].astype(float), worlds[order]
        self.starts = (
            np.r_[0, np.flatnonzero(np.diff(scores)) + 1] if len(scores) else np.array([], int)
        )

    def value(self, counts: np.ndarray) -> float:
        if not len(self.labels):
            return float("nan")
        weights = counts[self.worlds]
        positive = np.add.reduceat(weights * self.labels, self.starts)
        negative = np.add.reduceat(weights * (1 - self.labels), self.starts)
        p, n = positive.sum(), negative.sum()
        if p == 0 or n == 0:
            return float("nan")
        return float(np.sum(positive * (np.cumsum(negative) - 0.5 * negative)) / (p * n))

    def support(self) -> dict[str, int]:
        positive = self.labels == 1
        return dict(
            events=len(self.labels),
            positives=int(positive.sum()),
            negatives=int((~positive).sum()),
            positive_worlds=len(set(self.worlds[positive].tolist())),
            negative_worlds=len(set(self.worlds[~positive].tolist())),
        )


def bars(per_action: float, b8_action: float) -> str:
    if not np.isfinite([per_action, b8_action]).all():
        return "INCOMPLETE_SUPPORT"
    minimum = min(per_action, b8_action)
    return "GO" if minimum >= 0.85 else "GRAY" if minimum >= 0.75 else "KILL"


def analyze(held: dict, scores: dict[str, torch.Tensor], *, resamples: int = 2000) -> dict:
    world_ids = held["world"].numpy()
    _, world = np.unique(world_ids, return_inverse=True)
    worlds = int(world.max()) + 1
    y, legal, action = held["doom"].numpy(), held["legal"].numpy(), held["action"].numpy()
    if y.shape != (*legal.shape, 4) or not legal[np.arange(len(action)), action].all():
        raise ValueError("unaligned held-out outcomes/illegal B8 action")
    indices = np.arange(len(action))
    report: dict[str, Any] = dict(
        schema="doom-auc-result/v1",
        bootstrap_resamples=resamples,
        bootstrap_seed=343012,
        intervals="two-sided percentile 95%",
        parent_worlds=worlds,
        endpoints={},
    )
    endpoints: dict[str, dict[str, AUC]] = {}
    kind = np.asarray(held["kind"])
    if kind.shape != (len(y),) or not np.isin(kind, KINDS).all():
        raise ValueError("unaligned/foreign root kind")
    strata = {}
    for length in ("late", "all"):
        included = held["length"].numpy() >= 600 if length == "late" else np.ones(len(y), bool)
        strata[length] = included
        for root_kind in KINDS:
            strata[length + "/" + root_kind] = included & (kind == root_kind)
    for stratum, included in strata.items():
        for endpoint in ("per_action", "b8_action", *(f"action_{a}" for a in range(6))):
            mask = legal & included[:, None]
            if endpoint == "b8_action":
                mask = np.zeros_like(mask)
                mask[indices[included], action[included]] = True
            elif endpoint.startswith("action_"):
                mask[:, np.arange(6) != int(endpoint[-1])] = False
            events = np.broadcast_to(mask[..., None], y.shape)
            event_world = np.broadcast_to(world[:, None, None], y.shape)[events]
            name = stratum + "/" + endpoint
            endpoints[name] = {}
            for arm in ARMS:
                prediction = scores[arm].numpy()
                if prediction.shape != legal.shape:
                    raise ValueError("unaligned predictions")
                endpoints[name][arm] = AUC(
                    np.broadcast_to(prediction[..., None], y.shape)[events], y[events], event_world
                )
    counts = np.ones(worlds)
    replicates = {name: {arm: [] for arm in ARMS} for name in endpoints}
    rng = np.random.default_rng(343012)
    for _ in range(resamples):
        counts = np.bincount(rng.integers(worlds, size=worlds), minlength=worlds)
        for name, arms in endpoints.items():
            for arm, metric in arms.items():
                replicates[name][arm].append(metric.value(counts))
    support_ok = True
    for name, arms in endpoints.items():
        summary = dict(support=arms["spatial"].support(), arms={})
        samples = {}
        for arm, metric in arms.items():
            samples[arm] = np.array(replicates[name][arm])
            valid = np.isfinite(samples[arm])
            value = metric.value(np.ones(worlds))
            summary["arms"][arm] = dict(
                auc=value if np.isfinite(value) else None,
                ci95=(
                    np.quantile(samples[arm][valid], [0.025, 0.975]).tolist()
                    if valid.any()
                    else None
                ),
                valid_bootstraps=int(valid.sum()),
                undefined_bootstraps=int((~valid).sum()),
            )
        delta = samples["spatial"] - samples["b8"]
        finite = np.isfinite(delta)
        a, b = (summary["arms"][arm]["auc"] for arm in ARMS)
        summary["spatial_minus_b8"] = dict(
            auc_delta=a - b if a is not None and b is not None else None,
            ci95=np.quantile(delta[finite], [0.025, 0.975]).tolist() if finite.any() else None,
        )
        report["endpoints"][name] = summary
        if name in {"late/per_action", "late/b8_action"}:
            support_ok &= (
                summary["support"]["positive_worlds"] >= 10
                and summary["support"]["negative_worlds"] >= 10
                and finite.sum() >= 0.95 * resamples
            )
    primary = [
        report["endpoints"][name]["arms"]["spatial"]["auc"]
        for name in ("late/per_action", "late/b8_action")
    ]
    report["verdict"] = bars(*primary) if support_ok else "INCOMPLETE_SUPPORT"
    report["decision_rule"] = (
        "min(late per-action AUC, late B8-action AUC): GO>=.85; GRAY>=.75; KILL<.75"
    )
    report["interpretation"] = dict(
        sample="death-lookback-enriched searched roots",
        go="UPPER BOUND learnability screen; not natural-state danger AUC",
        upper_bound_is_mathematical=False,
        root_kind_splits="descriptive only; no change to primary support or bars",
        spatial_parameters=630182,
        train_roots=2048,
        sampled_training_set_equivalents=3000 * 32 / 2048,
        limitation="Finite-bank overfitting/selection bias; final update only, no early stopping",
    )
    return report


def aggregate(packet: Path, outputs: list[Path], result: Path, prelook: Path | None = None) -> dict:
    """Reserve look path BEFORE reading predictions/outcomes; never overwrite or peek twice."""
    if result.exists():
        raise FileExistsError("scoring look already reserved")
    config = json.loads((packet / "config.json").read_text())
    ledger = build_ledger(outputs, config["identity"])
    units = {r["unit"]: r for r in ledger["units"]}
    if not all(arm + "-prediction" in units for arm in ARMS):
        raise ValueError("INCOMPLETE: both final predictions required before look")
    if prelook is None:
        raise ValueError("pre-look amendment and root-kind metadata required")
    metadata = read_prelook(prelook, config)
    held_path = packet / "assessment-MAC-ONLY.pt.gz"
    if digest(held_path) != config["assessment_sha256"]:
        raise ValueError("assessment input hash differs")
    # The fixed lock is inherited by completion packets. A new result path cannot
    # create a second scoring look for the same frozen study.
    look = Path(config["mac_look_file"])
    if not look.is_absolute():
        raise ValueError("absolute Mac scoring-look reservation required")
    json_once(
        look,
        dict(
            identity=config["identity"],
            result=str(result),
            outputs=[str(p) for p in outputs],
            prelook_sha256=digest(prelook),
            documents=metadata["documents"],
        ),
    )
    result.mkdir(parents=True, exist_ok=False)
    scores = {}
    held = attach_kinds(load_gzip(held_path), metadata)
    for arm in ARMS:
        row = units[arm + "-prediction"]
        owners = [
            p / "exports/doom" for p in outputs if (p / "exports/doom" / row["path"]).exists()
        ]
        if len(owners) != 1:
            raise ValueError("ambiguous unit ownership")
        pred = load_unit(owners[0], row, config["identity"])
        if pred["steps"] != 3000 or pred["root_uids"] != held["root_uids"]:
            raise ValueError("INVALID: incomplete or unaligned final predictions")
        scores[arm] = pred["logits"]
    report = analyze(held, scores)
    report.update(
        identity=config["identity"],
        registration_sha256=config["registration_sha256"],
        union_ledger=ledger,
        serving_authorized=False,
        prelook_sha256=digest(prelook),
        amendment_documents=metadata["documents"],
    )
    json_once(result / "result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, nargs="+", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prelook", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(
        args.packet.resolve(),
        [p.resolve() for p in args.outputs],
        args.result.resolve(),
        args.prelook.resolve(),
    )
    print(json.dumps(dict(verdict=report["verdict"], result=str(args.result))))


if __name__ == "__main__":
    main()
