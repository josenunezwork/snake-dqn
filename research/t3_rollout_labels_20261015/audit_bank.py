"""Independent final audit for the T3 rollout-label bank.

The collector writes a deliberately split artifact: namespace and world
receipts describe the source worlds, world ``.npz`` files preserve raw rows,
calibration sidecars enrich the 50 legacy worlds, and root-level shards are
the train-facing export.  This module checks those boundaries independently
and writes a compact, machine-readable PASS receipt only after every check
has succeeded.

Usage::

    PYTHONPATH=$PWD ./venv/bin/python -m research.t3_rollout_labels_20261015.audit_bank \
        --bank /path/to/bank --out /path/to/audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HORIZON = 5000
DEPTH = 300
NAMESPACE = "t3-rollout-labels-dense/v1"
ARMS = ("b8_s0", "b8_s1", "b8_s2", "b8_s3", "b8_s4")
MIX_COUNTS = {"scripted": 27, "mixed": 9, "frozen": 4}
ALLOWED_ROOT_FILES = {"manifest.json", "README.md", "audit.json"}


class AuditError(RuntimeError):
    """Raised when a persisted bank violates a required audit invariant."""


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise AuditError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON object required: {path}")
    return value


def _same(left: Any, right: Any, label: str) -> None:
    if left != right:
        _fail(f"{label} mismatch: {left!r} != {right!r}")


def _key(job: dict[str, Any], plan: dict[str, Any]) -> str:
    return f"{job['arm']}-{job['mix']}-c{job['cycle']}-i{job['index']}-s{int(plan['seed'])}"


def _expected_mix(index: int) -> str:
    if index < MIX_COUNTS["scripted"]:
        return "scripted"
    if index < MIX_COUNTS["scripted"] + MIX_COUNTS["mixed"]:
        return "mixed"
    return "frozen"


def _np_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
        _fail(f"{label} differs: {left.shape}/{left.dtype} != {right.shape}/{right.dtype}")


def _finite(value: np.ndarray, label: str) -> None:
    if not np.isfinite(value).all():
        _fail(f"non-finite {label}")


def _load_manifest(bank: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = bank / "manifest.json"
    if not manifest_path.is_file():
        _fail(f"missing manifest: {manifest_path}")
    manifest = _json(manifest_path)
    _same(manifest.get("schema"), "t3-label-bank/v1", "manifest schema")
    files = manifest.get("files")
    shards = manifest.get("shards")
    receipts = manifest.get("world_receipts")
    if (
        not isinstance(files, dict)
        or not isinstance(shards, dict)
        or not isinstance(receipts, dict)
    ):
        _fail("manifest files, shards, and world_receipts must be mappings")
    for name, digest in files.items():
        path = bank / str(name)
        if not path.is_file() or sha256(path) != digest:
            _fail(f"manifest file hash mismatch: {name}")
    for name, digest in shards.items():
        path = bank / str(name)
        if not path.is_file() or sha256(path) != digest:
            _fail(f"manifest train-shard hash mismatch: {name}")
        if path.parent != bank or path.suffix != ".npz":
            _fail(f"manifest shard is not a root .npz: {name}")
    root_shards = {path.name for path in bank.glob("*.npz")}
    if root_shards != set(shards):
        _fail(f"root train shards differ from manifest: {root_shards ^ set(shards)}")

    world_dir = bank / "worlds"
    if not world_dir.is_dir():
        _fail("missing worlds directory")
    actual_receipts = {path.name for path in world_dir.glob("*.json")}
    if actual_receipts != set(receipts):
        _fail(f"world receipt inventory differs from manifest: {actual_receipts ^ set(receipts)}")
    for name, digest in receipts.items():
        path = world_dir / str(name)
        if sha256(path) != digest:
            _fail(f"world receipt hash mismatch: {name}")
        shard = path.with_suffix(".npz")
        if not shard.is_file():
            _fail(f"world receipt has no world shard: {name}")
        receipt = _json(path)
        if receipt.get("rows_sha256") != sha256(shard):
            _fail(f"world shard hash mismatch from receipt: {name}")

    # The collector's ``files`` map is the complete non-world inventory.  This
    # catches an unbound sidecar or an accidentally left temporary file while
    # allowing the human README and this audit receipt beside the hash-bound
    # manifest.
    listed = set(files) | {f"worlds/{name}" for name in receipts}
    listed |= {f"worlds/{name[:-5]}.npz" for name in receipts}
    for path in bank.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(bank).as_posix()
        if relative in ALLOWED_ROOT_FILES or relative in listed:
            continue
        _fail(f"unlisted bank file: {relative}")
    return manifest, {"files": files, "shards": shards, "receipts": receipts}


def _load_namespace(bank: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    namespace = _json(bank / "namespace.json")
    _same(namespace.get("schema"), "t3-label-namespace/v1", "namespace schema")
    _same(namespace.get("namespace"), NAMESPACE, "namespace")
    jobs = namespace.get("jobs")
    seeds = namespace.get("seeds")
    plans = namespace.get("plans")
    if not isinstance(jobs, list) or not isinstance(seeds, list) or not isinstance(plans, list):
        _fail("namespace jobs, seeds, and plans must be lists")
    if not (len(jobs) == len(seeds) == len(plans)) or not jobs:
        _fail("namespace jobs, seeds, and plans have unequal or empty lengths")
    if len(seeds) != len({int(seed) for seed in seeds}):
        _fail("namespace seeds are not unique")
    expected: dict[str, dict[str, Any]] = {}
    job_keys: set[tuple[str, str, int, int]] = set()
    for index, (job, seed, plan) in enumerate(zip(jobs, seeds, plans)):
        if not isinstance(job, dict) or not isinstance(plan, dict):
            _fail(f"namespace entry {index} is not an object")
        required = {"arm", "mix", "cycle", "index"}
        if not required <= set(job):
            _fail(f"namespace job {index} is missing {required - set(job)}")
        arm, mix = str(job["arm"]), str(job["mix"])
        cycle, job_index = int(job["cycle"]), int(job["index"])
        if arm not in ARMS or mix not in MIX_COUNTS or job_index != (job_index % 40):
            _fail(f"invalid namespace job: {job}")
        if mix != _expected_mix(job_index):
            _fail(f"namespace mix order is not 27/9/4: {job}")
        token = (arm, mix, cycle, job_index)
        if token in job_keys:
            _fail(f"duplicate namespace job: {token}")
        job_keys.add(token)
        if int(seed) != int(plan.get("seed", -1)) or str(plan.get("mix")) != mix:
            _fail(f"namespace plan does not bind seed/mix at {index}")
        if not isinstance(plan.get("specs"), list) or not isinstance(
            plan.get("member_sha256s"), list
        ):
            _fail(f"namespace roster missing at {index}")
        if len(plan["specs"]) != len(plan["member_sha256s"]):
            _fail(f"namespace roster lengths differ at {index}")
        if int(plan.get("world_index", -1)) != index:
            _fail(f"namespace world_index mismatch at {index}")
        key = _key(job, plan)
        if key in expected:
            _fail(f"duplicate namespace key: {key}")
        expected[key] = {"job": job, "plan": plan, "seed": int(seed)}
    return namespace, expected, {key: value for key, value in expected.items()}


def _load_receipts(
    bank: Path, manifest_meta: dict[str, Any], expected: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    receipts: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for filename in sorted(manifest_meta["receipts"]):
        receipt = _json(bank / "worlds" / filename)
        key = str(receipt.get("key"))
        _same(Path(filename).stem, key, f"receipt filename {key}")
        if key in receipts:
            _fail(f"duplicate world receipt key: {key}")
        if key not in expected:
            _fail(f"world receipt key is absent from namespace: {key}")
        binding = expected[key]
        _same(receipt.get("status"), "PASS", f"receipt status {key}")
        for name in ("arm", "mix", "cycle"):
            _same(receipt.get(name), binding["job"][name], f"receipt {name} {key}")
        _same(int(receipt.get("seed", -1)), binding["seed"], f"receipt seed {key}")
        _same(receipt.get("roster"), binding["plan"], f"receipt roster {key}")
        if not isinstance(receipt.get("rows"), int) or receipt["rows"] < 0:
            _fail(f"invalid receipt row count: {key}")
        if receipt.get("run_identity", {}).get("depth") != DEPTH:
            _fail(f"receipt depth is not D{DEPTH}: {key}")
        if receipt.get("run_identity", {}).get("horizon") != HORIZON:
            _fail(f"receipt horizon is not H{HORIZON}: {key}")
        if not (bank / "worlds" / f"{key}.npz").is_file():
            _fail(f"missing world shard: {key}")
        receipts[key] = receipt
        sources[key] = {"receipt": receipt, "job": binding["job"], "plan": binding["plan"]}
    if not receipts:
        _fail("no world receipts")
    by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts.values():
        by_cycle[int(receipt["cycle"])].append(receipt)
    for cycle, rows in by_cycle.items():
        if len(rows) != len(ARMS) * sum(MIX_COUNTS.values()):
            _fail(f"cycle {cycle} is incomplete: {len(rows)} receipts")
        counts = Counter((str(row["arm"]), str(row["mix"])) for row in rows)
        for arm in ARMS:
            for mix, count in MIX_COUNTS.items():
                if counts[(arm, mix)] != count:
                    _fail(f"cycle {cycle} has {counts[(arm, mix)]} {arm}/{mix}, expected {count}")
    return receipts, sources


def _load_source_rows(
    bank: Path,
    key: str,
    source: dict[str, Any],
    legacy_keys: set[str],
    enrichment_hashes: dict[str, str],
) -> dict[int, dict[str, Any]]:
    path = bank / "worlds" / f"{key}.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {"local", "global", "scalars", "mask", "surviving", "action"}
    if not required <= set(data):
        _fail(f"world source missing fields for {key}: {required - set(data)}")
    n = len(data["action"])
    for name in ("decision", "frame"):
        if name not in data:
            if key in legacy_keys and n == 0:
                data[name] = np.zeros(0, dtype=np.int64)
            else:
                _fail(f"world source missing {name}: {key}")
    if any(len(data[name]) != n for name in (*required, "decision", "frame")):
        _fail(f"world source row lengths differ: {key}")
    if (
        data["scalars"].dtype != np.float32
        or data["scalars"].ndim != 2
        or data["scalars"].shape[1] != 60
    ):
        _fail(f"world source scalar schema mismatch: {key}")
    if data["mask"].dtype != np.bool_ or data["mask"].shape != (n, 6):
        _fail(f"world source mask schema mismatch: {key}")
    if data["surviving"].dtype != np.bool_ or data["surviving"].shape != (n, 6):
        _fail(f"world source surviving schema mismatch: {key}")
    if data["local"].dtype != np.uint8 or data["local"].shape != (n, 7, 31, 31):
        _fail(f"world source local schema mismatch: {key}")
    if data["global"].dtype != np.uint8 or data["global"].shape != (n, 4, 37, 37):
        _fail(f"world source global schema mismatch: {key}")
    if data["action"].dtype != np.int64 or ((data["action"] < 0) | (data["action"] > 5)).any():
        _fail(f"world source action schema mismatch: {key}")
    if data["decision"].dtype != np.int64 or data["frame"].dtype != np.int64:
        _fail(f"world source decision/frame dtype mismatch: {key}")
    if "decisive" not in data or data["decisive"].dtype != np.bool_:
        _fail(f"world source decisive schema mismatch: {key}")
    _finite(data["scalars"], f"source scalars {key}")
    sidecar: dict[str, np.ndarray] | None = None
    if key in legacy_keys:
        sidecar_path = bank / "calibration" / f"{key}.npz"
        if not sidecar_path.is_file() or enrichment_hashes.get(key) != sha256(sidecar_path):
            _fail(f"legacy calibration sidecar hash mismatch: {key}")
        with np.load(sidecar_path, allow_pickle=False) as archive:
            sidecar = {name: archive[name].copy() for name in archive.files}
        if not {"q", "length", "decision", "frame"} <= set(sidecar):
            _fail(f"legacy calibration sidecar incomplete: {key}")
        for name in ("q", "length", "decision", "frame"):
            if len(sidecar[name]) != n:
                _fail(f"legacy sidecar length mismatch {name}: {key}")
        _np_equal(data["decision"], sidecar["decision"], f"legacy decision {key}")
        _np_equal(data["frame"], sidecar["frame"], f"legacy frame {key}")
    elif not {"q", "length"} <= set(data):
        _fail(f"current world source lacks q/length: {key}")

    q = sidecar["q"] if sidecar is not None else data["q"]
    length = sidecar["length"] if sidecar is not None else data["length"]
    if q.dtype != np.float32 or q.shape != (n, 6) or not np.isfinite(q).all():
        _fail(f"source q schema/non-finite values: {key}")
    if length.shape != (n,) or not np.issubdtype(length.dtype, np.integer):
        _fail(f"source length schema mismatch: {key}")
    rows: dict[int, dict[str, Any]] = {}
    for index in range(n):
        decision = int(data["decision"][index])
        if decision in rows:
            _fail(f"duplicate source decision: {key}/{decision}")
        frame, size = int(data["frame"][index]), int(length[index])
        if frame < 2000 or size < 700 or frame > HORIZON - DEPTH + 1:
            _fail(f"source metadata outside frame/length/full-horizon bounds: {key}/{decision}")
        rows[decision] = {
            "local": data["local"][index],
            "global": data["global"][index],
            "scalars": data["scalars"][index],
            "mask": data["mask"][index],
            "surviving": data["surviving"][index],
            "action": int(data["action"][index]),
            "decisive": bool(data["decisive"][index]),
            "frame": frame,
            "length": size,
            "q": q[index],
        }
    return rows


def _provenance(bank: Path, shard_names: Iterable[str]) -> list[dict[str, Any]]:
    paths = sorted(bank.glob("bank-*.provenance.jsonl"))
    if not paths:
        _fail("missing final provenance JSONL")
    if {path.name.removesuffix(".provenance.jsonl") + ".npz" for path in paths} != set(shard_names):
        _fail("final provenance does not have one file per train shard")
    entries: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(f"invalid provenance JSON {path}:{line_number}: {exc}")
            if not isinstance(value, dict):
                _fail(f"provenance entry is not an object: {path}:{line_number}")
            entries.append(value)
    seen: set[tuple[str, int]] = set()
    for index, value in enumerate(entries):
        if int(value.get("row", -1)) != index:
            _fail(f"provenance row numbering is not contiguous at {index}")
        source_key = str(value.get("source_key"))
        decision = int(value.get("decision", -1))
        pair = (source_key, decision)
        if pair in seen:
            _fail(f"duplicate final provenance source_key+decision: {pair}")
        seen.add(pair)
        if not isinstance(value.get("decisive"), bool):
            _fail(f"provenance decisive flag is not bool at row {index}")
    return entries


def _audit_export(
    bank: Path,
    manifest_meta: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    legacy_keys: set[str],
    enrichment_hashes: dict[str, str],
    threshold: float,
) -> dict[str, Any]:
    entries = _provenance(bank, manifest_meta["shards"].keys())
    loaded_sources = {
        key: _load_source_rows(bank, key, {}, legacy_keys, enrichment_hashes) for key in receipts
    }
    exported_pairs = {(str(entry["source_key"]), int(entry["decision"])) for entry in entries}
    for key, rows in loaded_sources.items():
        receipt = receipts[key]
        if int(receipt["rows"]) != len(rows):
            _fail(f"receipt/source row count differs: {key}")
        source_decisive = sum(bool(row["decisive"]) for row in rows.values())
        if int(receipt.get("decisive_rows", -1)) != source_decisive:
            _fail(f"receipt/source decisive count differs: {key}")
        missing_decisive = [
            decision
            for decision, row in rows.items()
            if row["decisive"] and (key, decision) not in exported_pairs
        ]
        if missing_decisive:
            _fail(f"decisive source rows were dropped from export: {key}/{missing_decisive[0]}")
    arrays: list[dict[str, np.ndarray]] = []
    for name in sorted(manifest_meta["shards"]):
        with np.load(bank / name, allow_pickle=False) as archive:
            value = {field: archive[field].copy() for field in archive.files}
        required = {"scalars", "mask", "surviving", "action", "weight", "q", "length"}
        if not required <= set(value):
            _fail(f"final shard missing fields {required - set(value)}: {name}")
        if not ("packed" in value or {"local", "global"} <= set(value)):
            _fail(f"final shard has no observations: {name}")
        n = len(value["action"])
        if value["scalars"].shape != (n, 60) or value["scalars"].dtype != np.float32:
            _fail(f"final scalar schema mismatch: {name}")
        if value["mask"].shape != (n, 6) or value["mask"].dtype != np.bool_:
            _fail(f"final mask schema mismatch: {name}")
        if value["surviving"].shape != (n, 6) or value["surviving"].dtype != np.bool_:
            _fail(f"final surviving schema mismatch: {name}")
        if value["action"].shape != (n,) or value["weight"].shape != (n,):
            _fail(f"final action/weight shape mismatch: {name}")
        if value["action"].dtype != np.int64 or value["weight"].dtype != np.float32:
            _fail(f"final action/weight dtype mismatch: {name}")
        if value["q"].dtype != np.float32 or value["q"].shape != (n, 6):
            _fail(f"final q schema mismatch: {name}")
        if value["length"].shape != (n,):
            _fail(f"final length schema mismatch: {name}")
        if "packed" in value:
            if value["packed"].dtype != np.uint8:
                _fail(f"final packed dtype mismatch: {name}")
            from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

            layout = train_layout()
            if value["packed"].shape != (n, int(layout.nbytes)):
                _fail(f"final packed layout mismatch: {name}")
            local, global_ = unpack_ref(value["packed"], layout)
            value["local"], value["global"] = local, global_
        elif (
            value["local"].dtype != np.uint8
            or value["global"].dtype != np.uint8
            or value["local"].shape != (n, 7, 31, 31)
            or value["global"].shape != (n, 4, 37, 37)
        ):
            _fail(f"final observation schema mismatch: {name}")
        _finite(value["scalars"], f"final scalars {name}")
        _finite(value["q"], f"final q {name}")
        if not np.isfinite(value["weight"]).all() or (value["weight"] < 0).any():
            _fail(f"final weights are invalid: {name}")
        if not np.array_equal(value["weight"], np.ones(n, dtype=np.float32)):
            _fail(f"final weights are not collector unit weights: {name}")
        if value["surviving"].shape[0] and not value["surviving"].any(axis=1).all():
            _fail(f"final surviving set is empty: {name}")
        arrays.append(value)

    final = {
        field: np.concatenate([value[field] for value in arrays])
        for field in (
            "local",
            "global",
            "scalars",
            "mask",
            "surviving",
            "action",
            "weight",
            "q",
            "length",
        )
    }
    if len(entries) != len(final["action"]):
        _fail(f"provenance rows {len(entries)} do not match export rows {len(final['action'])}")
    decisive = 0
    for index, entry in enumerate(entries):
        key, decision = str(entry["source_key"]), int(entry["decision"])
        if key not in loaded_sources or decision not in loaded_sources[key]:
            _fail(f"provenance source row is absent: {key}/{decision}")
        source = loaded_sources[key][decision]
        for field in ("local", "global", "scalars", "mask", "surviving"):
            _np_equal(final[field][index], source[field], f"export {field} {key}/{decision}")
        for field in ("action", "length"):
            if int(final[field][index]) != int(source[field]):
                _fail(f"export {field} differs: {key}/{decision}")
        if not np.array_equal(final["q"][index], source["q"]):
            _fail(f"export q differs: {key}/{decision}")
        if int(entry.get("frame", -1)) != source["frame"]:
            _fail(f"provenance frame differs: {key}/{decision}")
        if bool(entry.get("decisive")) != source["decisive"]:
            _fail(f"provenance decisive differs: {key}/{decision}")
        mask, q, action, survive = (
            source["mask"],
            source["q"],
            source["action"],
            source["surviving"],
        )
        if not mask.any() or not mask[action] or int(np.where(mask, q, -np.inf).argmax()) != action:
            _fail(f"Q/action/advisory mismatch: {key}/{decision}")
        if np.any(survive & ~mask):
            _fail(f"surviving set contains advisory-illegal action: {key}/{decision}")
        if source["decisive"]:
            if survive[action] or not np.any(survive):
                _fail(f"decisive relation is false: {key}/{decision}")
            decisive += 1
        elif not survive[action] or not np.any(source["mask"] & ~survive):
            _fail(f"counterweight relation is false: {key}/{decision}")
        if key not in legacy_keys and float(-np.max(q)) <= threshold:
            _fail(f"current row fails strict prefilter threshold: {key}/{decision}")
    counts = {
        "rows": len(entries),
        "decisive_rows": decisive,
        "counterweight_rows": len(entries) - decisive,
        "surviving_action_density": float(final["surviving"].mean()) if entries else 0.0,
        "rows_with_boost_survivor_fraction": (
            float(final["surviving"][:, 3:].any(axis=1).mean()) if entries else 0.0
        ),
        "surviving_boost_share": float(
            final["surviving"][:, 3:].sum() / max(1, final["surviving"].sum())
        ),
        "b8_agree": (
            float(final["surviving"][np.arange(len(entries)), final["action"]].mean())
            if entries
            else 0.0
        ),
    }
    if counts["counterweight_rows"] > counts["decisive_rows"]:
        _fail("global counterweights exceed decisive rows")
    return {"provenance_rows": len(entries), "stats": counts, "train_shard_rows": len(entries)}


def _audit_prefilter(
    bank: Path,
    legacy_keys: set[str],
    receipts: dict[str, dict[str, Any]],
    manifest_meta: dict[str, Any],
) -> dict[str, Any]:
    path = bank / "prefilter.json"
    if not path.is_file():
        _fail("missing prefilter.json")
    prefilter = _json(path)
    _same(prefilter.get("schema"), "t3-prefilter/v1", "prefilter schema")
    if prefilter.get("status") != "PASS" or not np.isfinite(float(prefilter.get("threshold"))):
        _fail("prefilter is not a finite PASS receipt")
    if float(prefilter.get("coverage", 0.0)) < 0.95:
        _fail("prefilter decisive coverage is below 0.95")
    enrichment = prefilter.get("enrichment_sha256")
    if not isinstance(enrichment, dict) or set(enrichment) != legacy_keys:
        _fail("prefilter enrichment does not cover exactly the legacy worlds")
    for key, digest in enrichment.items():
        sidecar = bank / "calibration" / f"{key}.npz"
        if not sidecar.is_file() or sha256(sidecar) != digest:
            _fail(f"prefilter enrichment hash mismatch: {key}")

    replay_path = bank / "calibration" / "replay-scores.npz"
    if not replay_path.is_file() or prefilter.get("replay_scores_sha256") != sha256(replay_path):
        _fail("replay-scores.npz is absent or not hash-bound by prefilter")
    with np.load(replay_path, allow_pickle=False) as archive:
        required = {"score", "key", "decision"}
        if not required <= set(archive.files):
            _fail(f"replay score NPZ missing {required - set(archive.files)}")
        score = archive["score"].astype(np.float64)
        keys = archive["key"].astype(str)
        decisions = archive["decision"].astype(np.int64)
    if len(score) != len(keys) or len(score) != len(decisions) or not np.isfinite(score).all():
        _fail("replay score NPZ arrays are inconsistent")
    if not set(keys) <= legacy_keys:
        _fail("replay score NPZ contains a key outside the legacy worlds")
    threshold = float(prefilter["threshold"])
    skipped = int((score <= threshold).sum())
    replay = prefilter.get("replay", {})
    _same(int(replay.get("dense_batches", -1)), len(score), "prefilter dense batch count")
    _same(int(replay.get("skipped_dense_batches", -1)), skipped, "prefilter skipped dense batches")
    expected_fraction = skipped / max(1, len(score))
    if not math.isclose(
        float(replay.get("skip_fraction_all_dense_batches", -1)), expected_fraction
    ):
        _fail("prefilter skip fraction cannot be reconstructed from replay NPZ")
    by_mix: dict[str, dict[str, int]] = defaultdict(
        lambda: {"worlds": 0, "dense_batches": 0, "skipped": 0}
    )
    for key in legacy_keys:
        mix = str(receipts[key]["mix"])
        by_mix[mix]["worlds"] += 1
        mask = keys == key
        by_mix[mix]["dense_batches"] += int(mask.sum())
        by_mix[mix]["skipped"] += int((score[mask] <= threshold).sum())
    declared_by_mix = replay.get("by_mix", {})
    for mix, values in by_mix.items():
        declared = declared_by_mix.get(mix)
        if not isinstance(declared, dict):
            _fail(f"prefilter replay lacks mix bucket: {mix}")
        for field in ("worlds", "dense_batches", "skipped"):
            _same(int(declared.get(field, -1)), values[field], f"prefilter {mix} {field}")
    coverage_by_arm: dict[str, dict[str, int]] = defaultdict(
        lambda: {"decisive_rows": 0, "retained_decisive_rows": 0}
    )
    coverage_by_mix: dict[str, dict[str, int]] = defaultdict(
        lambda: {"decisive_rows": 0, "retained_decisive_rows": 0}
    )
    for key in legacy_keys:
        with (
            np.load(bank / "worlds" / f"{key}.npz", allow_pickle=False) as source,
            np.load(bank / "calibration" / f"{key}.npz", allow_pickle=False) as sidecar,
        ):
            decisive = source["decisive"].astype(bool)
            scores = -sidecar["q"].astype(np.float64).max(axis=1)
        if len(scores) != len(decisive):
            _fail(f"legacy calibration rows differ for coverage: {key}")
        for bucket in (
            coverage_by_arm[str(receipts[key]["arm"])],
            coverage_by_mix[str(receipts[key]["mix"])],
        ):
            bucket["decisive_rows"] += int(decisive.sum())
            bucket["retained_decisive_rows"] += int((scores[decisive] > threshold).sum())
    for name, declared in (
        ("decisive_coverage_by_arm", coverage_by_arm),
        ("decisive_coverage_by_mix", coverage_by_mix),
    ):
        raw = prefilter.get("replay", {}).get(name, {})
        for key, values in declared.items():
            item = raw.get(key)
            if not isinstance(item, dict):
                _fail(f"prefilter lacks {name} bucket: {key}")
            for field in values:
                _same(int(item.get(field, -1)), values[field], f"prefilter {name} {key} {field}")
    decisive_total = sum(item["decisive_rows"] for item in coverage_by_arm.values())
    retained_total = sum(item["retained_decisive_rows"] for item in coverage_by_arm.values())
    _same(prefilter.get("decisive_rows"), decisive_total, "calibration decisive population")
    _same(
        prefilter.get("retained_decisive_rows"), retained_total, "calibration retained population"
    )
    if not decisive_total or not math.isclose(
        float(prefilter["coverage"]), retained_total / decisive_total
    ):
        _fail("pooled calibration coverage differs from recovered source rows")
    return {
        "threshold": threshold,
        "coverage": float(prefilter["coverage"]),
        "legacy_worlds": len(legacy_keys),
        "replay_dense_batches": len(score),
        "replay_skipped_dense_batches": skipped,
        "replay_skip_fraction": expected_fraction,
    }


def audit_bank(bank: Path, out: Path, *, min_decisive: int = 10_000) -> dict[str, Any]:
    """Audit a completed bank and write ``out``; raise ``AuditError`` on failure."""
    bank, out = Path(bank), Path(out)
    if not bank.is_dir():
        _fail(f"bank is not a directory: {bank}")
    manifest, manifest_meta = _load_manifest(bank)
    namespace, expected, _ = _load_namespace(bank)
    receipts, _ = _load_receipts(bank, manifest_meta, expected)
    legacy_path = bank / "legacy-resume.json"
    legacy = _json(legacy_path) if legacy_path.is_file() else {"entries": {}}
    _same(legacy.get("schema"), "t3-legacy-resume/v1", "legacy resume schema")
    legacy_entries = legacy.get("entries")
    if not isinstance(legacy_entries, dict) or len(legacy_entries) != 50:
        _fail(f"legacy resume must retain exactly 50 worlds, got {len(legacy_entries or {})}")
    legacy_keys = set(legacy_entries)
    if not legacy_keys <= set(receipts):
        _fail("legacy resume has missing world receipts")
    current_identity = manifest.get("provenance", {}).get("run_identity")
    for key in set(receipts) - legacy_keys:
        if receipts[key].get("run_identity") != current_identity:
            _fail(f"current world run identity differs from manifest: {key}")
        if receipts[key].get("q_action_mismatches") != 0:
            _fail(f"current world has Q action mismatches: {key}")
    for key, entry in legacy_entries.items():
        receipt_path = bank / "worlds" / f"{key}.json"
        rows_path = bank / "worlds" / f"{key}.npz"
        if entry.get("receipt_sha256") != sha256(receipt_path) or entry.get(
            "rows_sha256"
        ) != sha256(rows_path):
            _fail(f"legacy immutable receipt/shard hash changed: {key}")
        if entry.get("run_identity") != receipts[key].get("run_identity"):
            _fail(f"legacy immutable run identity changed: {key}")
    prefilter = _audit_prefilter(bank, legacy_keys, receipts, manifest_meta)
    export = _audit_export(
        bank,
        manifest_meta,
        receipts,
        legacy_keys,
        _json(bank / "prefilter.json")["enrichment_sha256"],
        prefilter["threshold"],
    )
    stats = export["stats"]
    if stats["decisive_rows"] < min_decisive:
        _fail(f"decisive rows {stats['decisive_rows']} below required {min_decisive}")
    declared_stats = manifest.get("provenance", {}).get("stats")
    if not isinstance(declared_stats, dict):
        _fail("manifest provenance has no stats mapping")
    for field in ("rows", "decisive_rows", "counterweight_rows"):
        _same(int(declared_stats.get(field, -1)), stats[field], f"manifest stats {field}")
    for field in (
        "surviving_action_density",
        "rows_with_boost_survivor_fraction",
        "surviving_boost_share",
        "b8_agree",
    ):
        if not math.isclose(
            float(declared_stats.get(field, -1)), stats[field], rel_tol=1e-7, abs_tol=1e-7
        ):
            _fail(f"manifest stats {field} mismatch")
    declared_keys = manifest["provenance"].get("keys_done")
    if sorted(str(value) for value in declared_keys or []) != sorted(receipts):
        _fail("manifest provenance keys_done differs from world receipts")
    declared_cycles = manifest["provenance"].get("cycles_done")
    actual_cycles = sorted({int(value["cycle"]) for value in receipts.values()})
    if sorted(int(value) for value in declared_cycles or []) != actual_cycles:
        _fail("manifest provenance cycles_done differs from world receipts")
    # Construct through the real learner loader after all shard checks.  This
    # validates packed bytes against the trainer's exact layout and exercises
    # concatenation/sampling-compatible dtypes.
    from research.redesign_m3c_20261008.obs_pack import train_layout
    from research.t3_rollout_labels_20261015.bank import LabelBank

    loaded = LabelBank(bank, nscal=60, pack=train_layout())
    _same(loaded.size, export["train_shard_rows"], "LabelBank size")
    report = {
        "schema": "t3-label-bank-audit/v1",
        "status": "PASS",
        "bank": str(bank),
        "manifest_sha256": sha256(bank / "manifest.json"),
        "namespace": {
            "value": namespace["namespace"],
            "jobs": len(namespace["jobs"]),
            "seeds": len(set(int(seed) for seed in namespace["seeds"])),
            "completed_worlds": len(receipts),
            "completed_cycles": sorted({int(value["cycle"]) for value in receipts.values()}),
        },
        "legacy": {"worlds": len(legacy_keys), "immutable": True},
        "prefilter": prefilter,
        "export": export,
        "labelbank_rows": loaded.size,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_bank(args.bank, args.out)
    except (AuditError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"BANK_AUDIT_FAIL: {exc}") from exc
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
