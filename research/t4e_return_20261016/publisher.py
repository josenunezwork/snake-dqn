"""Disk-backed publication of the T4e all-action return bank.

The two collection passes deliberately return records a world at a time.  This
module keeps that property through publication: a completed world becomes one
immutable, trainer-readable NPZ shard plus a small receipt.  The root manifest
is assembled only from those receipts and shard hashes; it never concatenates
the branch rows into process memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from research.t4d_return_regression_20261016.collector import (
    FAMILY_DEATH_WINDOW,
    FAMILY_OUTSIDE,
    GAMMA,
    KAPPA,
)

SCHEMA = "t4e-return-bank/v1"
WORKER_LIMIT = 12


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one published file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write a JSON receipt, refusing to replace evidence."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite published receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite published receipt: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def embedded_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Build the byte-identical manifest embedded in every world shard."""
    from research.redesign_m3c_20261008.obs_pack import train_layout

    layout = train_layout()
    return {
        "schema_version": 1,
        "collection_schema": SCHEMA,
        "gamma": GAMMA,
        "kappa": KAPPA,
        "phi": "length/10,dead=0",
        "reward_semantics": "mass+pbrs+death_no_overlays_or_kill",
        "fields": [
            "state_packed",
            "state_scalars",
            "action",
            "T",
            "died",
            "S",
            "L0",
            "L_T",
            "terminal_packed",
            "terminal_scalars",
            "terminal_mask",
            "row_weight",
            "family",
        ],
        "layout": layout.to_dict(),
        "packed_shape": [int(layout.nbytes)],
        "scalars_shape": [60],
        "packing_version": "ego2s-bv8/obs_pack-v1",
        "feature_order": (
            "ego2s-bv8 local,global packed; bv8 scalars as emitted by build_ego_raster"
        ),
        "provenance": dict(provenance),
    }


def _world_key(job: Mapping[str, Any]) -> str:
    """Return a portable, deterministic identity for one planned source world."""
    index = int(job["plan_index"])
    arm = str(job["arm"])
    mix = str(job["mix"])
    seed = int(job["seed"])
    return f"world-{index:05d}-{arm}-{mix}-s{seed}"


def _branch_rows(
    records: Iterable[Mapping[str, Any]], family: int, *, source_seed: int | None = None
) -> Iterable[dict[str, Any]]:
    """Yield flattened action rows without retaining an all-world intermediate."""
    for record in records:
        branches = record.get("branches")
        if not isinstance(branches, Mapping) or not branches:
            raise ValueError("published state has no actual all-action branch outcomes")
        legal = np.asarray(record.get("legal", ()), dtype=np.int64)
        branch_actions = sorted(int(value) for value in branches)
        if not len(legal) or set(branch_actions) != set(legal.tolist()):
            raise ValueError("published state must retain every and only advisory-legal branch")
        packed = np.asarray(record["state_packed"], dtype=np.uint8)
        scalars = np.asarray(record["state_scalars"], dtype=np.float32)
        if packed.ndim != 1 or scalars.shape != (60,):
            raise ValueError("published state packing has an invalid shape")
        if not np.isfinite(scalars).all():
            raise ValueError("published state scalars must be finite")
        if family == FAMILY_DEATH_WINDOW:
            distance = int(record["death_distance"])
            if not 0 <= distance < 300:
                raise ValueError("decisive state has an invalid death distance")
            weight = 1.0 if distance < 100 else 0.3
        elif family == FAMILY_OUTSIDE:
            distance = -1
            weight = 0.5
        else:  # pragma: no cover - module constants make this an internal contract.
            raise ValueError("unknown return-bank row family")
        for action in branch_actions:
            branch = branches[action]
            if not isinstance(branch, Mapping):
                raise ValueError("branch outcome is not a mapping")
            died = bool(branch["died"])
            terminal = branch.get("terminal")
            if died:
                terminal_packed = np.zeros_like(packed)
                terminal_scalars = np.zeros(60, dtype=np.float32)
                terminal_mask = np.zeros(6, dtype=bool)
            else:
                if terminal is None:
                    raise ValueError("surviving branch lacks terminal observation")
                from research.redesign_m3c_20261008.obs_pack import (
                    pack_ref,
                    train_layout,
                )

                terminal_packed = pack_ref(
                    np.asarray(terminal["local"]), np.asarray(terminal["global"]), train_layout()
                )[0]
                terminal_scalars = np.asarray(terminal["scalars"][0], dtype=np.float32)
                terminal_mask = np.asarray(branch["terminal_mask"], dtype=bool)
                if terminal_scalars.shape != (60,) or terminal_mask.shape != (6,):
                    raise ValueError("surviving branch terminal inputs have an invalid shape")
            yield {
                "state_packed": packed,
                "state_scalars": scalars,
                "action": action,
                "T": int(branch["T"]),
                "died": died,
                "S": float(branch["S"]),
                "L0": int(branch["L0"]),
                "L_T": int(branch["L_T"]),
                "terminal_packed": terminal_packed,
                "terminal_scalars": terminal_scalars,
                "terminal_mask": terminal_mask,
                "row_weight": weight,
                "family": family,
                "source_seed": int(
                    source_seed if source_seed is not None else record.get("seed", -1)
                ),
                "source_decision": int(record["source_decision"]),
                "death_distance": distance,
            }


def _arrays(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Convert one bounded world to exact ReturnBank arrays."""
    if not rows:
        raise ValueError("refusing to publish an empty world shard")
    values = {
        "state_packed": np.stack([row["state_packed"] for row in rows]).astype(np.uint8),
        "state_scalars": np.stack([row["state_scalars"] for row in rows]).astype(np.float32),
        "action": np.asarray([row["action"] for row in rows], dtype=np.int64),
        "T": np.asarray([row["T"] for row in rows], dtype=np.int32),
        "died": np.asarray([row["died"] for row in rows], dtype=bool),
        "S": np.asarray([row["S"] for row in rows], dtype=np.float64),
        "L0": np.asarray([row["L0"] for row in rows], dtype=np.int32),
        "L_T": np.asarray([row["L_T"] for row in rows], dtype=np.int32),
        "terminal_packed": np.stack([row["terminal_packed"] for row in rows]).astype(np.uint8),
        "terminal_scalars": np.stack([row["terminal_scalars"] for row in rows]).astype(np.float32),
        "terminal_mask": np.stack([row["terminal_mask"] for row in rows]).astype(bool),
        "row_weight": np.asarray([row["row_weight"] for row in rows], dtype=np.float32),
        "family": np.asarray([row["family"] for row in rows], dtype=np.uint8),
        "source_seed": np.asarray([row["source_seed"] for row in rows], dtype=np.int64),
        "source_decision": np.asarray([row["source_decision"] for row in rows], dtype=np.int64),
        "death_distance": np.asarray([row["death_distance"] for row in rows], dtype=np.int64),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True, separators=(",", ":"))),
    }
    if (
        not np.isfinite(values["S"]).all()
        or (values["S"] < 0).any()
        or (values["T"] < 1).any()
        or (values["row_weight"] <= 0).any()
    ):
        raise ValueError("world shard contains invalid return statistics")
    return values


def _publish_rows(
    root: Path,
    job: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    *,
    suffix: str = "",
) -> dict[str, Any]:
    """Publish already-flattened action rows as one immutable source shard."""
    root = Path(root)
    key = _world_key(job) + suffix
    shard = root / f"{key}.npz"
    receipt_path = root / "worlds" / f"{key}.receipt.json"
    if shard.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite published world: {key}")
    manifest = embedded_manifest(provenance)
    source_seed = int(job["seed"])
    normalized = []
    for row in rows:
        value = dict(row)
        if int(value["source_seed"]) not in (-1, source_seed):
            raise ValueError("published row source seed disagrees with its planned world")
        value["source_seed"] = source_seed
        normalized.append(value)
    arrays = _arrays(normalized, manifest)
    return _publish_arrays(root, job, arrays, manifest, suffix=suffix)


def _publish_arrays(
    root: Path,
    job: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    suffix: str = "",
) -> dict[str, Any]:
    """Persist one already-validated, bounded source shard and its receipt."""
    root = Path(root)
    key = _world_key(job) + suffix
    shard = root / f"{key}.npz"
    receipt_path = root / "worlds" / f"{key}.receipt.json"
    if shard.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite published world: {key}")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=root, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, shard)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "key": key,
        "job": {name: job[name] for name in ("plan_index", "arm", "mix", "seed")},
        "shard": shard.name,
        "sha256": _sha256(shard),
        "rows": int(len(arrays["action"])),
        "decisive_rows": int(np.sum(arrays["family"] == FAMILY_DEATH_WINDOW)),
        "outside_rows": int(np.sum(arrays["family"] == FAMILY_OUTSIDE)),
        "embedded_manifest_sha256": hashlib.sha256(
            arrays["manifest_json"].item().encode()
        ).hexdigest(),
    }
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def publish_world_family(
    root: Path,
    job: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    family: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one world/family immediately from a collection worker.

    Phase A uses this for training decisive rows before its compact candidate
    spool is returned to the coordinator.  Phase B uses it for selected outside
    rows.  The suffix keeps their independently durable evidence disjoint.
    """
    suffix = "-decisive" if family == FAMILY_DEATH_WINDOW else "-outside"
    return _publish_rows(
        root,
        job,
        list(_branch_rows(records, family, source_seed=int(job["seed"]))),
        provenance,
        suffix=suffix,
    )


def publish_world(
    root: Path,
    job: Mapping[str, Any],
    decisive: Iterable[Mapping[str, Any]],
    outside: Iterable[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one source world's return rows and an atomic, hash-bound receipt.

    ``decisive`` and ``outside`` may be generators.  Only this one world's
    flattened action rows are materialized before its compressed shard is
    written, which bounds collector memory independently of bank size.
    """
    rows = list(_branch_rows(decisive, FAMILY_DEATH_WINDOW, source_seed=int(job["seed"])))
    rows.extend(_branch_rows(outside, FAMILY_OUTSIDE, source_seed=int(job["seed"])))
    return _publish_rows(root, job, rows, provenance)


def publish_manifest(root: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Create the final manifest from hash-bound world receipts without loading rows."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite published manifest: {manifest_path}")
    expected = embedded_manifest(provenance)
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    receipts = sorted((root / "worlds").glob("*.receipt.json"))
    if not receipts:
        raise ValueError("cannot publish aggregate manifest without world receipts")
    shards: dict[str, str] = {}
    total = decisive = outside = 0
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != SCHEMA or receipt.get("status") != "COMPLETE":
            raise ValueError(f"invalid world receipt: {receipt_path}")
        name, digest = receipt.get("shard"), receipt.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or name in shards:
            raise ValueError(f"invalid world shard binding: {receipt_path}")
        shard = root / name
        if not shard.is_file() or _sha256(shard) != digest:
            raise ValueError(f"world shard hash mismatch: {receipt_path}")
        with np.load(shard, allow_pickle=False) as archive:
            if (
                "manifest_json" not in archive.files
                or str(archive["manifest_json"].item()) != expected_json
            ):
                raise ValueError(f"world shard embedded manifest mismatch: {shard}")
        shards[name] = digest
        total += int(receipt["rows"])
        decisive += int(receipt["decisive_rows"])
        outside += int(receipt["outside_rows"])
    present = {path.name for path in root.glob("*.npz")}
    if present != set(shards):
        raise ValueError("root shard set does not exactly match published world receipts")
    value = {
        **expected,
        "shards": dict(sorted(shards.items())),
        "publication": {
            "world_receipts": {path.name: _sha256(path) for path in receipts},
            "rows": total,
            "decisive_rows": decisive,
            "outside_rows": outside,
        },
    }
    _atomic_json(manifest_path, value)
    return value


def _candidate_spool(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    """Persist phase-A candidates in a non-pickle, replayable bounded spool."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate spool: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(candidates)
    from research.redesign_m3c_20261008.obs_pack import train_layout

    width = int(train_layout().nbytes)
    arrays = {
        "plan_index": np.asarray([row["plan_index"] for row in candidates], np.int64),
        "source_decision": np.asarray([row["source_decision"] for row in candidates], np.int64),
        "length": np.asarray([row["length"] for row in candidates], np.int32),
        "q": (
            np.stack([row["q"] for row in candidates]).astype(np.float32)
            if n
            else np.zeros((0, 6), np.float32)
        ),
        "state_packed": (
            np.stack([row["state_packed"] for row in candidates]).astype(np.uint8)
            if n
            else np.zeros((0, width), np.uint8)
        ),
        "state_scalars": (
            np.stack([row["state_scalars"] for row in candidates]).astype(np.float32)
            if n
            else np.zeros((0, 60), np.float32)
        ),
        "fingerprint": np.asarray([row["fingerprint"] for row in candidates], dtype="U128"),
        "holdout": np.asarray([row["holdout"] for row in candidates], bool),
        "death_distance": np.asarray(
            [row.get("death_distance", -1) for row in candidates], np.int64
        ),
        "b8_action": np.asarray([row.get("b8_action", -1) for row in candidates], np.int64),
    }
    legal = np.full((n, 6), -1, dtype=np.int64)
    for index, row in enumerate(candidates):
        values = np.asarray(row["legal"], dtype=np.int64)
        legal[index, : len(values)] = values
    arrays["legal"] = legal
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_spool(path: Path) -> list[dict[str, Any]]:
    """Load one bounded phase-A metadata spool as replay-ready state mappings."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "plan_index",
            "source_decision",
            "length",
            "q",
            "state_packed",
            "state_scalars",
            "fingerprint",
            "holdout",
            "legal",
            "death_distance",
            "b8_action",
        }
        if not required <= set(archive.files):
            raise ValueError(f"invalid T4e metadata spool: {path}")
        values = {name: archive[name].copy() for name in required}
    n = len(values["plan_index"])
    if any(len(value) != n for value in values.values()):
        raise ValueError(f"unaligned T4e metadata spool: {path}")
    result = []
    for index in range(n):
        legal = values["legal"][index]
        result.append(
            {
                "plan_index": int(values["plan_index"][index]),
                "source_decision": int(values["source_decision"][index]),
                "length": int(values["length"][index]),
                "q": values["q"][index],
                "state_packed": values["state_packed"][index],
                "state_scalars": values["state_scalars"][index],
                "fingerprint": str(values["fingerprint"][index]),
                "holdout": bool(values["holdout"][index]),
                "legal": legal[legal >= 0].astype(np.int64, copy=False),
                "death_distance": int(values["death_distance"][index]),
                "b8_action": int(values["b8_action"][index]),
            }
        )
    return result


def publish_selected_staged_world(
    final_root: Path,
    staging_root: Path,
    phase_a_receipt: Mapping[str, Any],
    source_decisions: Iterable[int],
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Copy selected action families from one staged shard into the final bank.

    Selection identity is a source-world receipt plus source-decision numbers;
    the coordinator derives it from the metadata spool's canonical fingerprint.
    This reads and writes exactly one world shard, never the complete bank.
    """
    published = phase_a_receipt.get("published")
    requested = {int(value) for value in source_decisions}
    if not requested:
        return None
    if not isinstance(published, Mapping):
        raise ValueError("selected world has no staged decisive branch shard")
    shard_name = published.get("shard")
    if not isinstance(shard_name, str):
        raise ValueError("staged publication has no shard name")
    staged = Path(staging_root) / shard_name
    if not staged.is_file() or _sha256(staged) != published.get("sha256"):
        raise ValueError("staged decisive shard hash mismatch")
    with np.load(staged, allow_pickle=False) as archive:
        rows = np.asarray(archive["source_decision"], dtype=np.int64)
        present = set(rows.tolist())
        if not requested <= present:
            raise ValueError("selected source decision missing from staged decisive shard")
        keep = np.isin(rows, sorted(requested))
        arrays = {
            name: archive[name][keep].copy() for name in archive.files if name != "manifest_json"
        }
    if not len(
        arrays["action"]
    ):  # pragma: no cover - subset validation above makes this defensive.
        raise ValueError("selected staged shard has no branch rows")
    manifest = embedded_manifest(provenance)
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    )
    job = phase_a_receipt.get("job")
    if not isinstance(job, Mapping):
        raise ValueError("phase-A receipt lacks source job")
    return _publish_arrays(final_root, job, arrays, manifest, suffix="-decisive")


def _phase_b_to_final(
    args: tuple[
        Path, Mapping[str, Any], Sequence[Mapping[str, Any]], np.ndarray, Mapping[str, Any]
    ],
) -> dict[str, Any] | None:
    """Replay selected outside states for one source world and publish one final shard."""
    from research.t4e_return_20261016 import collection

    final_root, job, candidates, cuts, provenance = args
    rows = collection.replay_selected_outside_world(job, candidates, cuts)
    if not rows:
        return None
    return publish_world_family(final_root, job, rows, FAMILY_OUTSIDE, provenance)


def publish_selected_staged(
    root: Path,
    jobs: Mapping[int, Mapping[str, Any]],
    selected_decisive: Iterable[tuple[int, int]],
    selected_outside: Iterable[tuple[int, int]],
    provenance: Mapping[str, Any],
    *,
    workers: int = WORKER_LIMIT,
) -> dict[str, Any]:
    """Reconstruct final train/holdout banks from staged metadata and branch shards.

    The input identity sets are ``(plan_index, source_decision)`` pairs.  They
    are intentionally the coordinator's post-deduplication output, so a staged
    row never becomes training evidence merely because it was collected.
    """
    if not 1 <= workers <= WORKER_LIMIT:
        raise ValueError("T4e collection permits from one through twelve workers")
    root = Path(root)
    staging = root / "staging"
    train, holdout = root / "banks" / "train", root / "banks" / "holdout"
    decisive_by_plan: dict[int, set[int]] = {}
    for plan_index, decision in selected_decisive:
        decisive_by_plan.setdefault(int(plan_index), set()).add(int(decision))
    outside_by_plan: dict[int, set[int]] = {}
    for plan_index, decision in selected_outside:
        outside_by_plan.setdefault(int(plan_index), set()).add(int(decision))
    phase_receipts = {
        int(json.loads(path.read_text())["job"]["plan_index"]): json.loads(path.read_text())
        for path in (staging / "phase-a").glob("*.json")
    }
    if set(decisive_by_plan) - set(phase_receipts) or set(outside_by_plan) - set(phase_receipts):
        raise ValueError("selection refers to a world without a staged phase-A receipt")
    decisive_receipts = []
    outside_candidates: dict[int, list[dict[str, Any]]] = {}
    for plan_index, receipt in sorted(phase_receipts.items()):
        job = jobs.get(plan_index)
        if job is None:
            raise ValueError("staged phase-A receipt lacks a planned job")
        decisions = decisive_by_plan.get(plan_index, set())
        target = holdout if bool(job.get("holdout", False)) else train
        published = publish_selected_staged_world(target, staging, receipt, decisions, provenance)
        if published is not None:
            decisive_receipts.append(published)
        wanted = outside_by_plan.get(plan_index, set())
        if wanted:
            rows = load_spool(staging / receipt["candidate_spool"])
            selected = [row for row in rows if int(row["source_decision"]) in wanted]
            if {int(row["source_decision"]) for row in selected} != wanted:
                raise ValueError("selected outside state missing from staged candidate spool")
            outside_candidates[plan_index] = selected
    arguments = [
        (train, jobs[index], rows, np.empty(0), dict(provenance))
        for index, rows in sorted(outside_candidates.items())
    ]
    # Candidate rows retain their fixed stratum but phase-B's replay needs the
    # same fixed cuts.  The coordinator stores them in binding.json only after
    # the reference validation; recover those cuts from the immutable reference.
    if arguments:
        from research.t4e_return_20261016 import collection

        reference = Path(provenance["reference_bank"]["path"])
        cuts, _excluded = collection.historical_cuts(reference)
        arguments = [(path, job, rows, cuts, data) for path, job, rows, _empty, data in arguments]
    with ProcessPoolExecutor(max_workers=workers, initializer=_configure_cpu_worker) as executor:
        outside_receipts = [item for item in executor.map(_phase_b_to_final, arguments) if item]
    result = {
        "train": publish_manifest(train, provenance),
        "holdout": (
            publish_manifest(holdout, provenance)
            if decisive_receipts and holdout.exists()
            else None
        ),
        "decisive_shards": decisive_receipts,
        "outside_shards": outside_receipts,
        "train_path": str(train),
        "holdout_path": str(holdout),
    }
    _atomic_json(root / "publication.json", result)
    return result


def validate_completed_banks(root: Path, published: Mapping[str, Any]) -> dict[str, Any]:
    """Open final banks through the learner's validator and bind their row counts."""
    from research.t4d_return_regression_20261016.bank import ReturnBank

    train_path = Path(published["train_path"])
    train = ReturnBank(train_path, 60)
    result: dict[str, Any] = {
        "status": "PASS",
        "train_rows": train.size,
        "train_path": str(train_path),
    }
    holdout = published.get("holdout")
    if holdout is not None:
        holdout_path = Path(published["holdout_path"])
        value = ReturnBank(holdout_path, 60)
        result.update({"holdout_rows": value.size, "holdout_path": str(holdout_path)})
    return result


def _configure_cpu_worker() -> None:
    """Apply the collection's one-thread rule inside every spawned worker."""
    from research.t4d_return_regression_20261016.collector import _configure_device

    _configure_device("cpu")


def _phase_a_to_disk(
    args: tuple[Path, Mapping[str, Any], np.ndarray, Mapping[str, Any]],
) -> dict[str, Any]:
    """Collect and persist one source world without returning its record lists."""
    from research.t4e_return_20261016 import collection

    root, job, cuts, provenance = args
    decisive, candidates, receipt = collection.collect_phase_a_world(job, cuts)
    training_decisive = [row for row in decisive if not bool(row.get("holdout", False))]
    if decisive:
        published = publish_world_family(root, job, decisive, FAMILY_DEATH_WINDOW, provenance)
    else:
        published = None
    key = _world_key(job)
    spool = root / "candidates" / f"{key}.npz"
    _candidate_spool(spool, candidates)
    decisive_spool = root / "decisive" / f"{key}.npz"
    _candidate_spool(decisive_spool, decisive)
    result = {
        "schema": "t4e-phase-a-world/v1",
        "status": "COMPLETE",
        "key": key,
        "job": {name: job[name] for name in ("plan_index", "arm", "mix", "seed")},
        "decisive_states": len(decisive),
        "training_decisive_states": len(training_decisive),
        "candidate_states": len(candidates),
        "candidate_spool": str(spool.relative_to(root)),
        "candidate_spool_sha256": _sha256(spool),
        "decisive_spool": str(decisive_spool.relative_to(root)),
        "decisive_spool_sha256": _sha256(decisive_spool),
        "published": published,
        "world": receipt,
    }
    _atomic_json(root / "phase-a" / f"{key}.json", result)
    return result


def run_phase_a_to_disk(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    cuts: np.ndarray,
    provenance: Mapping[str, Any],
    *,
    workers: int = WORKER_LIMIT,
) -> list[dict[str, Any]]:
    """Run a bounded phase-A batch whose only parent results are tiny receipts.

    This is intentionally separate from ``collection.execute_phase_a``: the
    latter returns all raw candidates and branch payloads to its parent.  Here
    each worker writes its own candidate spool and optional decisive shard.
    """
    if not 1 <= workers <= WORKER_LIMIT:
        raise ValueError("T4e collection permits from one through twelve workers")
    from research.t4e_return_20261016 import collection

    ordered = sorted((dict(job) for job in jobs), key=lambda row: int(row["plan_index"]))
    collection.validate_mix_cycle(ordered)
    # These are staged artifacts only.  A later global matcher must create the
    # final bank root after it has proved exact decisive/outside matching; phase
    # A evidence must never be mistaken for a ready trainer input.
    staging = Path(root) / "staging"
    arguments = [
        (staging, job, np.asarray(cuts, dtype=np.float64), dict(provenance)) for job in ordered
    ]
    with ProcessPoolExecutor(max_workers=workers, initializer=_configure_cpu_worker) as executor:
        results = list(executor.map(_phase_a_to_disk, arguments))
    write_progress(root, results, phase="phase_a")
    return results


def write_progress(
    root: Path, receipts: Sequence[Mapping[str, Any]], *, phase: str
) -> dict[str, Any]:
    """Write one create-only bounded progress receipt; it never claims bank completion."""
    root = Path(root)
    values = {
        "schema": "t4e-return-collection-progress/v1",
        "status": "IN_PROGRESS",
        "phase": phase,
        "completed_worlds": len(receipts),
        "keys": [str(row["key"]) for row in receipts],
        "training_decisive_states": sum(
            int(row.get("training_decisive_states", 0)) for row in receipts
        ),
        "candidate_states": sum(int(row.get("candidate_states", 0)) for row in receipts),
    }
    progress = root / "progress"
    # Each full continuation batch has the same world count.  Sequence by
    # already-published receipts so a later batch cannot overwrite phase_a-00200.
    sequence = len(list(progress.glob(f"{phase}-*.json")))
    _atomic_json(progress / f"{phase}-{sequence:05d}.json", values)
    return values
