"""T4c-final extension of the hash-bound same-init drift capture.

This wrapper owns only fresh CPU captures in its caller-selected output.  It
reuses the historical CTRL r62 replay identity and raw-Q convention, records a
T4c final alongside it, and never opens tmux or starts a learner.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.t4c_20261016 import drift, drift_report  # noqa: E402

ARM = "TWO_SIDED_T"
SEEDS = tuple(range(10, 15))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights(summary: Mapping[str, Any]) -> Mapping[str, float]:
    """Recover immutable pooled standardization weights from the verified report."""
    rows = summary["rounds"]["62"]["arms"]["B8"]["consequential"]["strata"]
    return {key: float(value["weight"]) for key, value in rows.items()}


def _bank_fingerprints(summary: Mapping[str, Any]) -> np.ndarray:
    path = drift.POSTHOC.parent / "bank" / "TWO_SIDED_s10.npz"
    receipt_path = drift.POSTHOC.parent / "bank" / "TWO_SIDED_s10.receipt.json"
    evidence = summary["evidence_sha256"]
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("out") != str(path)
        or receipt.get("out_sha256") != sha256(path)
        or evidence.get(str(path)) != sha256(path)
        or evidence.get(str(receipt_path)) != sha256(receipt_path)
    ):
        raise ValueError("T4a bank fingerprint capture binding mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return archive["fingerprint"].copy()


def standardize_capture(
    capture: Mapping[str, np.ndarray], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the verified r62 pooled cuts/weights to one T4c final capture."""
    cuts = np.asarray(summary["rounds"]["62"]["arms"]["B8"]["global_decile_cutpoints"])
    return drift_report.seed_rates(
        capture, "t4c", cuts, _weights(summary), _bank_fingerprints(summary)
    )


def capture(seed: int, checkpoint: Path, out: Path) -> dict[str, Any]:
    """Replay one r62 CTRL row with an added T4c-final raw-Q column.

    ``drift.replay`` supplies the audited game replay and CTRL/B8 capture.  The
    local observer adds only an extra no-grad CPU forward on the exact active
    rows; it cannot influence CTRL's action selection or simulator state.
    """
    if seed not in SEEDS:
        raise ValueError("T4c drift seed must be 10..14")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_digest = sha256(checkpoint)
    original = drift.CaptureObserver

    class T4cObserver(original):
        def __init__(self, replay_seed: int, round_index: int, outside_collector=None) -> None:
            import torch
            from src.model.ego2s_network import load_ego2s_checkpoint

            super().__init__(replay_seed, round_index, outside_collector)
            torch.set_num_threads(1)
            self.t4c = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
            self.rows["t4c_q"] = []

        def observe(self, sim, slots, ctrl_q) -> None:
            from src.simd_env.ego2s_policy import forward_q
            from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

            rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
            active = sim.get_alive()[rows[:, 0], rows[:, 1]].astype(bool)
            # Keep this observation construction identical to CaptureObserver.
            obs = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
            self.rows["t4c_q"].append(forward_q(self.t4c, obs, "rowwise").numpy()[active])
            super().observe(sim, slots, ctrl_q)

    drift.CaptureObserver = T4cObserver
    try:
        base = drift.replay(seed, 62, out, collect_outside=False)
    finally:
        drift.CaptureObserver = original
    if sha256(checkpoint) != checkpoint_digest:
        raise RuntimeError("T4c final checkpoint changed during drift capture")
    payload = out / "captures" / f"CTRL_s{seed}__r62__scripted.npz"
    receipt = {
        "schema": "t4c-final-drift-capture/v1",
        "seed": seed,
        "round_index": 62,
        "capture": str(payload),
        "capture_sha256": sha256(payload),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "base_capture_receipt": base,
        "base_capture_receipt_sha256": sha256(payload.with_suffix(".receipt.json")),
    }
    target = payload.with_suffix(".t4c-final.receipt.json")
    if target.exists():
        raise FileExistsError(f"refusing existing T4c-final drift receipt: {target}")
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def summarize(out: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Read five fresh r62 captures and compare matched seed means to B8."""
    per_seed = {}
    for seed in SEEDS:
        path = Path(out) / "captures" / f"CTRL_s{seed}__r62__scripted.npz"
        receipt = path.with_suffix(".t4c-final.receipt.json")
        item = json.loads(receipt.read_text())
        if (
            item.get("schema") != "t4c-final-drift-capture/v1"
            or item.get("seed") != seed
            or item.get("round_index") != 62
            or item.get("capture_sha256") != sha256(path)
            or item.get("checkpoint_sha256") != sha256(Path(item["checkpoint"]))
            or item.get("base_capture_receipt_sha256") != sha256(path.with_suffix(".receipt.json"))
        ):
            raise ValueError(f"invalid T4c-final capture receipt: {receipt}")
        base = item.get("base_capture_receipt", {})
        for raw_path, digest in base.get("inputs_sha256", {}).items():
            source = Path(raw_path)
            if str(source).startswith(str(drift.ARTIFACTS)):
                if (
                    not source.is_file()
                    or sha256(source) != digest
                    or summary["evidence_sha256"].get(str(source)) != digest
                ):
                    raise ValueError(f"changed permitted base-capture input: {source}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if "t4c_q" not in arrays:
            raise ValueError(f"missing T4c final Q capture: {path}")
        per_seed[str(seed)] = standardize_capture(arrays, summary)
    consequential = [item["consequential_change_rate"] for item in per_seed.values()]
    raw = [item["raw_change_rate"] for item in per_seed.values()]
    if any(rate is None for rate in consequential):
        rate, within = None, None
    else:
        rate = sum(float(item) for item in consequential) / len(consequential)
        baseline = float(
            summary["rounds"]["62"]["arms"]["B8"]["five_seed_mean"]["consequential_change_rate"]
        )
        within = abs(rate - baseline) <= 0.10
    return {
        "per_seed": per_seed,
        "candidate_raw_change_rate": (
            None
            if any(value is None for value in raw)
            else sum(float(value) for value in raw) / len(raw)
        ),
        "candidate_consequential_change_rate": rate,
        "baseline_five_seed_mean": summary["rounds"]["62"]["arms"]["B8"]["five_seed_mean"],
        "baseline_per_seed": summary["rounds"]["62"]["arms"]["B8"]["per_seed"],
        "within_ten_percentage_points": within,
    }


def main() -> int:
    """Capture one final in a watcher-owned single-threaded CPU worker."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = capture(args.seed, args.checkpoint, args.out)
    print(json.dumps({"seed": args.seed, "capture": receipt["capture"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
