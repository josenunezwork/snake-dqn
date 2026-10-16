"""Immutable T4c distance weights and bounded, matched outside-counter data.

Only artifact producers live here. No optimizer, training or accelerator path exists.
The outside score is the stored/raw B8 ``-max Q`` on both bank and probe rows;
cutpoints are fixed by the decisive bank and ties enter the upper decile.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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

from research.t4a_posthoc_20261016.spillover import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    deciles,
    length_bucket,
    observation_fingerprints,
    sha256,
)

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
V2 = ARTIFACTS / "t4-prep-20261015/bank-v2"
ROOT = ARTIFACTS / "t4c-data-20261016"


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load non-pickled arrays once, avoiding repeated compressed decompression."""
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def bound_files(directory: Path) -> dict[str, str]:
    """Bind exactly the published manifest and its listed shards."""
    meta = json.loads((directory / "manifest.json").read_text())
    declared = set(meta["shards"])
    present = {path.name for path in directory.glob("*.npz")}
    if declared != present:
        raise ValueError("bank .npz set does not match its manifest")
    result = {"manifest.json": sha256(directory / "manifest.json")}
    for name, digest in meta["shards"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"bank shard identity mismatch: {name}")
        result[name] = digest
    return result


def distance_weights(decisive: np.ndarray, decision: np.ndarray, death: np.ndarray) -> np.ndarray:
    """Apply advisor distance weights; counters remain one, all decisive gaps <300."""
    gap = death - decision
    if np.any(decisive & ((gap < 0) | (gap >= 300))):
        raise ValueError("decisive row is outside the strict stored 300-decision window")
    return np.where(decisive & (gap >= 100), 0.3, 1.0).astype(np.float32)


def derive_v3(source: Path = V2, out: Path = ROOT / "bank-v3") -> dict[str, Any]:
    """Copy v2, attach distance fields and weights, and bind original provenance."""
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or (out.exists() and any(out.iterdir())):
        raise ValueError("derived output must be empty and outside its immutable source")
    bindings = bound_files(source)
    meta = json.loads((source / "manifest.json").read_text())
    original = Path(meta["source"]["path"])
    provenance = original / "bank-00000.provenance.jsonl"
    if sha256(provenance) != meta["source"]["files"][provenance.name]:
        raise ValueError("original provenance hash mismatch")
    rows = [json.loads(line) for line in provenance.read_text().splitlines() if line]
    if sorted(row["row"] for row in rows) != list(range(len(rows))):
        raise ValueError("provenance row keys must be unique contiguous indices")
    rows.sort(key=lambda row: row["row"])
    receipts: dict[str, str] = {}
    death_by_source = {}
    for key in sorted({row["source_key"] for row in rows}):
        relative = f"worlds/{key}.json"
        path = original / relative
        if sha256(path) != meta["source"]["files"][relative]:
            raise ValueError(f"world receipt hash mismatch: {key}")
        receipt = json.loads(path.read_text())
        death_by_source[key] = receipt["death_decision"]
        receipts[str(path)] = sha256(path)
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(source / "bank-00000.npz")
    n = len(arrays["action"])
    if len(rows) != n:
        raise ValueError("provenance does not align with bank rows")
    decisive = ~arrays["surviving_original"][np.arange(n), arrays["action"]]
    if not np.array_equal(decisive, [row["decisive"] for row in rows]):
        raise ValueError("provenance decisive flags disagree with preserved outcomes")
    arrays["decision"] = np.asarray([row["decision"] for row in rows], np.int64)
    arrays["death_decision"] = np.asarray(
        [
            (
                death_by_source[row["source_key"]]
                if death_by_source[row["source_key"]] is not None
                else -1
            )
            for row in rows
        ],
        np.int64,
    )
    arrays["row_weight"] = distance_weights(decisive, arrays["decision"], arrays["death_decision"])
    arrays["row_family"] = (~decisive).astype(np.int8)
    _atomic_npz(out / "bank-00000.npz", **arrays)
    value = {
        "schema": "t4c-derived-label-bank/v3",
        "rows": n,
        "source": {"path": str(source), "files": bindings},
        "provenance": {
            "path": str(provenance),
            "sha256": sha256(provenance),
            "world_receipts": receipts,
        },
        "weight_contract": (
            "effective=weight*row_weight; decisive gap<100:1,100<=gap<300:0.3; "
            "in-window counter:1"
        ),
        "family_counts": {
            "decisive": int(decisive.sum()),
            "in_window_counter": int((~decisive).sum()),
        },
        "decisive_weight_counts": {
            str(weight): int(np.sum(decisive & (arrays["row_weight"] == weight)))
            for weight in (1.0, 0.3)
        },
        "shards": {"bank-00000.npz": sha256(out / "bank-00000.npz")},
    }
    if bound_files(source) != bindings:
        raise RuntimeError("v2 source changed during derivation")
    _atomic_json(out / "manifest.json", value)
    return value


def strata(length: np.ndarray, q: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Use one fixed B8-score and length contract for targets and candidates."""
    return np.char.add(
        np.char.add(length_bucket(length).astype("U8"), ":D"),
        deciles(-q.max(axis=1), cuts).astype("U2"),
    )


def target_definition(bank: Path = V2) -> tuple[np.ndarray, dict[str, int], set[str]]:
    """Return fixed decisive Q deciles, exact stratum quotas, bank fingerprints."""
    from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

    arrays = load_arrays(bank / "bank-00000.npz")
    decisive = ~arrays["surviving_original"][np.arange(len(arrays["action"])), arrays["action"]]
    cuts = np.quantile(-arrays["q"][decisive].max(axis=1), np.arange(1, 10) / 10)
    quotas = dict(Counter(strata(arrays["length"][decisive], arrays["q"][decisive], cuts)))
    local, glob = unpack_ref(arrays["packed"], train_layout())
    fingerprints = set(observation_fingerprints(local, glob, arrays["scalars"]).tolist())
    return cuts, quotas, fingerprints


class OutsideCollector:
    """Bounded deterministic per-stratum reservoir for one round-62 replay.

    ``observe(sim, rows, obs, ctrl_q, b8_q)`` accepts the already-built raster
    mapping (local/global uint8, scalars float32), rows shape(n,2), Q shape(n,6).
    ``finish(out, death_frame, world_seeds, inputs)`` accepts the hero's observed
    one-based death_frame vector (-1 for survival), and publishes candidates.
    Reservoirs are sampled before post-hoc death exclusion, never oversampled
    with duplicate rows. Only states with >=300 observed continuation decisions
    enter final output; censored last-300 states are excluded even for survivors.
    """

    def __init__(self, seed: int, owner: str, bank: Path = V2) -> None:
        from research.redesign_m3c_20261008.obs_pack import train_layout

        self.layout = train_layout()
        self.seed, self.owner, self.bank = int(seed), owner, Path(bank)
        self.cuts, self.quotas, self.bank_fingerprints = target_definition(bank)
        self.rng = np.random.default_rng(
            np.random.SeedSequence([20261016, seed, 0 if owner == "CTRL" else 1])
        )
        self.rows: dict[str, list[dict[str, Any]]] = {key: [] for key in self.quotas}
        self.seen = Counter()
        self.calls = Counter()

    def observe(
        self,
        sim: Any,
        rows: np.ndarray,
        obs: Mapping[str, np.ndarray],
        ctrl_q: np.ndarray,
        b8_q: np.ndarray,
    ) -> None:
        """Observe a prepared hero batch without changing any action or RNG."""
        from research.redesign_m3c_20261008.obs_pack import pack

        rows = np.asarray(rows).reshape(-1, 2)
        length = sim.get_lengths()[rows[:, 0], rows[:, 1]]
        active = sim.get_alive()[rows[:, 0], rows[:, 1]].astype(bool)
        labels = strata(length, b8_q, self.cuts)
        masks = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
        masks[~masks.any(axis=1), :3] = True
        for i in np.flatnonzero(active):
            env = int(rows[i, 0])
            decision = self.calls[env]
            self.calls[env] += 1
            key = str(labels[i])
            if length[i] < 700 or decision > 4699 or key not in self.quotas:
                continue
            self.seen[key] += 1
            capacity = max(64, int(np.ceil(self.quotas[key] * 0.6)))
            slot = len(self.rows[key])
            if slot >= capacity:
                slot = int(self.rng.integers(self.seen[key]))
                if slot >= capacity:
                    continue
            fp = str(
                observation_fingerprints(
                    obs["local"][i : i + 1], obs["global"][i : i + 1], obs["scalars"][i : i + 1]
                )[0]
            )
            if fp in self.bank_fingerprints:
                continue
            row = {
                "packed": pack(obs["local"][i : i + 1], obs["global"][i : i + 1], self.layout)[0],
                "scalars": obs["scalars"][i].copy(),
                "mask": masks[i].copy(),
                "action": int(np.where(masks[i], b8_q[i].astype(np.float64), -np.inf).argmax()),
                "q": b8_q[i].astype(np.float32).copy(),
                "length": int(length[i]),
                "fingerprint": fp,
                "env": env,
                "decision": decision,
                "frame": int(sim.frame[env]),
                "stratum": key,
            }
            if slot == len(self.rows[key]):
                self.rows[key].append(row)
            else:
                self.rows[key][slot] = row

    def finish(
        self, out: Path, death_frame: np.ndarray, world_seeds: list[int], inputs: Mapping[str, str]
    ) -> dict[str, Any]:
        """Exclude near-death/censored rows and publish a fresh candidate part."""
        path = Path(out) / f"{self.owner}_s{self.seed}.npz"
        receipt_path = path.with_suffix(".receipt.json")
        if any(target.exists() or target.is_symlink() for target in (path, receipt_path)):
            raise FileExistsError(f"refusing existing outside candidate payload or receipt: {path}")
        kept = []
        for rows in self.rows.values():
            for row in rows:
                # One decision per frame in the hermetic H5000 evaluator. Fatal
                # one-based frame D corresponds to zero-based decision D-1.
                fatal = int(death_frame[row["env"]])
                if fatal >= 0 and (fatal - 1) - row["decision"] < 300:
                    continue
                row["death_decision"] = fatal - 1 if fatal >= 0 else -1
                row["world_seed"] = int(world_seeds[row["env"]])
                kept.append(row)
        if not kept:
            raise ValueError("outside replay produced no eligible candidates")
        arrays = {key: np.asarray([row[key] for row in kept]) for key in kept[0]}
        _atomic_npz(path, **arrays)
        receipt = {
            "schema": "t4c-outside-candidates/v1",
            "owner": self.owner,
            "seed": self.seed,
            "round_index": 62,
            "mix": "scripted",
            "rows": len(kept),
            "counts": dict(Counter(arrays["stratum"].tolist())),
            "seen": dict(self.seen),
            "cuts": self.cuts.tolist(),
            "bank": str(self.bank),
            "bank_manifest_sha256": sha256(self.bank / "manifest.json"),
            "inputs_sha256": dict(inputs),
            "path": str(path),
            "sha256": sha256(path),
            "death_convention": (
                "terminal hero only; >=300 decisions to observed death; "
                "require 300 decisions observed future for survivors"
            ),
        }
        _atomic_json(receipt_path, receipt)
        return receipt


def build_outside(
    candidates: Path = ROOT / "candidates", out: Path = ROOT / "outside-bank", bank: Path = V2
) -> dict[str, Any]:
    """Select unique outside observations with exact decisive stratum counts."""
    from research.t4c_20261016.data_sources import verify_capture_inputs

    if out.resolve() == bank.resolve() or bank.resolve() in out.resolve().parents:
        raise ValueError("outside bank output must be outside its immutable source")
    if out.exists() and any(out.iterdir()):
        raise ValueError("outside bank output must be empty")
    bank_binding = bound_files(bank)
    cuts, quotas, bank_fingerprints = target_definition(bank)
    parts, bindings, source_snapshots = [], {}, {}
    for owner in ("CTRL", "B8"):
        for seed in range(10, 15):
            path = candidates / f"{owner}_s{seed}.npz"
            receipt_path = path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["schema"] != "t4c-outside-candidates/v1"
                or receipt["owner"] != owner
                or receipt["seed"] != seed
            ):
                raise ValueError("outside candidate identity mismatch")
            if receipt["bank_manifest_sha256"] != sha256(
                bank / "manifest.json"
            ) or not np.array_equal(receipt["cuts"], cuts):
                raise ValueError("outside candidate target-bank mismatch")
            if receipt["sha256"] != sha256(path):
                raise ValueError("outside candidate payload changed")
            snapshots = verify_capture_inputs(
                receipt["inputs_sha256"],
                snapshot_root=ROOT / "source-snapshots",
                repo=Path(__file__).resolve().parents[2],
            )
            source_snapshots.update(snapshots)
            for snapshot in snapshots.values():
                bindings[snapshot["snapshot_path"]] = snapshot["snapshot_sha256"]
            arrays = load_arrays(path)
            arrays["source_owner"] = np.full(len(arrays["action"]), owner)
            arrays["source_training_seed"] = np.full(len(arrays["action"]), seed, np.int16)
            arrays["source_candidate_row"] = np.arange(len(arrays["action"]), dtype=np.int32)
            parts.append(arrays)
            bindings[str(path)] = sha256(path)
            bindings[str(receipt_path)] = sha256(receipt_path)
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if not np.isfinite(arrays["q"]).all():
        raise ValueError("outside candidate raw Q must be finite")
    computed = strata(arrays["length"], arrays["q"], cuts)
    if not np.array_equal(computed, arrays["stratum"]):
        raise ValueError("candidate strata disagree with captured B8 Q")
    valid = (arrays["length"] >= 700) & (arrays["decision"] <= 4699)
    valid &= (arrays["death_decision"] < 0) | (
        (arrays["death_decision"] - arrays["decision"]) >= 300
    )
    valid &= np.asarray([fp not in bank_fingerprints for fp in arrays["fingerprint"]])
    if not valid.all():
        raise ValueError("candidate data violates outside-bank or death-distance constraints")
    # Exact-observation identity is stricter than trajectory row identity. Never
    # repeat an observation to repair an unsupported stratum.
    _, unique = np.unique(arrays["fingerprint"], return_index=True)
    unique_mask = np.zeros(len(valid), bool)
    unique_mask[unique] = True
    selected, available = [], {}
    rng = np.random.default_rng(20261016)
    for key, count in sorted(quotas.items()):
        indices = np.flatnonzero((computed == key) & unique_mask)
        available[key] = len(indices)
        if len(indices) < count:
            raise ValueError(f"outside stratum insufficient {key}: {len(indices)} < {count}")
        selected.extend(rng.choice(indices, count, replace=False).tolist())
    indices = np.sort(selected)
    result = {key: value[indices] for key, value in arrays.items()}
    n = len(indices)
    result["action"] = result["action"].astype(np.int64)
    result["mask"] = result["mask"].astype(bool)
    result["surviving"] = np.zeros((n, 6), bool)
    result["surviving"][np.arange(n), result["action"]] = True
    result["surviving_original"] = result["surviving"].copy()
    result["weight"] = np.ones(n, np.float32)
    result["row_weight"] = np.full(n, 0.5, np.float32)
    result["row_family"] = np.full(n, 2, np.int8)
    action = np.where(result["mask"], result["q"].astype(np.float64), -np.inf).argmax(axis=1)
    if not np.array_equal(action, result["action"]):
        raise ValueError("outside label differs from B8 advisory argmax")
    if bank_binding != bound_files(bank) or any(
        sha256(Path(path)) != digest for path, digest in bindings.items()
    ):
        raise RuntimeError("outside assembly inputs changed during preparation")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_npz(out / "bank-00000.npz", **result)
    value = {
        "schema": "t4c-outside-counter-bank/v1",
        "rows": n,
        "bank": str(bank),
        "bank_binding": bank_binding,
        "inputs_sha256": bindings,
        "capture_source_snapshots": source_snapshots,
        "assembly_source_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("data_sources.py").resolve(),
            )
        },
        "b8_seed_mapping": {str(seed): seed - 10 for seed in range(10, 15)},
        "outside_label_semantics": (
            "surviving/surviving_original are singleton preservation-label aliases; "
            "no counterfactual six-action survival outcomes were measured"
        ),
        "matching": {
            "score": "negative max raw B8 Q, both decisive and trajectory observations",
            "decile_cutpoints": cuts.tolist(),
            "ties": "upper bin",
            "length_buckets": ["700_999", "ge1000"],
            "target_counts": quotas,
            "selected_counts": dict(Counter(result["stratum"].tolist())),
            "available_unique_counts": available,
            "replacement": False,
            "ratio_to_decisive": 1.0,
            "seed": 20261016,
        },
        "source_counts": dict(Counter(result["source_owner"].tolist())),
        "minimum_length": int(result["length"].min()),
        "minimum_observed_death_distance": int(
            (result["death_decision"] - result["decision"])[result["death_decision"] >= 0].min()
        ),
        "counter_contract": (
            "stored B8 advisory argmax singleton; original weight=1, row_weight=0.5; "
            "own margin scale=1 times0.0632"
        ),
        "death_contract": (
            "terminal hero death; >=300 decisions from observed death; "
            "last300 decisions of censored survivor worlds excluded"
        ),
        "unique_fingerprints": len(set(result["fingerprint"].tolist())),
        "bank_overlap": 0,
        "shards": {"bank-00000.npz": sha256(out / "bank-00000.npz")},
    }
    _atomic_json(out / "manifest.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--outside", action="store_true")
    parser.add_argument("--source", type=Path, default=V2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--candidates", type=Path, default=ROOT / "candidates")
    args = parser.parse_args()
    if args.derive:
        print(json.dumps(derive_v3(args.source, args.out or ROOT / "bank-v3"), sort_keys=True))
    if args.outside:
        print(
            json.dumps(
                build_outside(args.candidates, args.out or ROOT / "outside-bank", args.source),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
