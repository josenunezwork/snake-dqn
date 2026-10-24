"""Run the frozen F1 root bank repeatedly for RPC re-qualification.

This module deliberately exports only operational integrity facts.  It is not a
labeller, analyser, or placement decision maker.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from research.bl_labels import f1_ruler
from research.bl_labels.exports import _materialize_triplet, _verify, publish_json
from research.rpc_qualification.injection import locator_for_unit

SCHEMA = "rpc-qualification-loop/v1"
LEDGER_SCHEMA = "rpc-qualification-verified-ledger/v1"
F1_SOURCE_COMMIT = "2af48cd0b98a2063f46871cf4d320a5541b789af"
ROOTS = 64


def _hash(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unit_id(namespace: str, allocation: str, index: int) -> str:
    if not isinstance(index, int) or index < 0:
        raise ValueError("foreign allocation index")
    return f"{namespace}/{allocation}/{index:08d}"


def empty_ledger(
    namespace: str,
    allocation: str,
    bank_sha256: str,
    source_commit: str = F1_SOURCE_COMMIT,
    source_closure_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": namespace,
        "allocation": allocation,
        "bank_sha256": bank_sha256,
        "source_commit": source_commit,
        "source_closure_sha256": source_closure_sha256,
        "units": [],
    }


def _load_ledger(
    path: Path,
    namespace: str,
    allocation: str,
    bank_sha256: str,
    source_closure_sha256: str | None = None,
) -> dict[int, dict[str, Any]]:
    _verify(path)
    value = json.loads(path.read_text())
    expected = empty_ledger(
        namespace, allocation, bank_sha256, F1_SOURCE_COMMIT, source_closure_sha256
    )
    if {key: value.get(key) for key in expected if key != "units"} != {
        key: expected[key] for key in expected if key != "units"
    } or not isinstance(value.get("units"), list):
        raise ValueError("foreign namespace/allocation/bank ledger")
    seen: dict[int, dict[str, Any]] = {}
    for row in value["units"]:
        index = row.get("index")
        if (
            not isinstance(index, int)
            or index in seen
            or row.get("unit_id") != unit_id(namespace, allocation, index)
        ):
            raise ValueError("duplicate or foreign ledger unit")
        if row.get("path") != f"units/unit-{index:08d}.json":
            raise ValueError("foreign ledger unit path")
        payload = path.parent / row["path"]
        if _verify(payload) != row.get("sha256") or payload.stat().st_size != row.get("size"):
            raise ValueError("ledger unit differs from immutable payload")
        identity = json.loads(payload.read_text())
        if (
            identity.get("schema") != SCHEMA
            or identity.get("namespace") != namespace
            or identity.get("allocation") != allocation
            or identity.get("index") != index
            or identity.get("unit_id") != row["unit_id"]
        ):
            raise ValueError("foreign ledger payload identity")
        seen[index] = row
    return seen


def _attest_closure(
    manifest: Path | None,
    expected_archive_sha256: str | None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Verify every frozen-source member before any gameplay work begins."""
    if manifest is None or expected_archive_sha256 is None:
        raise ValueError("frozen source closure manifest and sha256 are required")
    manifest_sha256 = _hash(manifest)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("frozen source closure manifest hash differs")
    value = json.loads(manifest.read_text())
    if (
        value.get("commit") != F1_SOURCE_COMMIT
        or value.get("archive_sha256") != expected_archive_sha256
    ):
        raise ValueError("frozen source closure archive/source pin differs")
    root = manifest.parent
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("frozen source closure has no file hashes")
    for relative, expected in files.items():
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or _hash(path) != expected:
            raise ValueError("frozen source closure member differs")
    ruler_path = "research/bl_labels/f1_ruler.py"
    if ruler_path in files and Path(f1_ruler.__file__).resolve() != (root / ruler_path).resolve():
        raise ValueError("F1 worker was imported outside the frozen source closure")
    return {
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": manifest_sha256,
        "members": len(files),
    }


def _label_root_with_hash(job: tuple[str, float]) -> dict[str, Any]:
    """Use the frozen labeller while retaining only an opaque canonical row hash."""
    from research.bl_labels import factory

    captured: list[dict[str, Any]] = []
    original = factory._row

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original(*args, **kwargs)
        captured.append(row)
        return row

    factory._row = capture
    try:
        record = f1_ruler._label_root(job)
    finally:
        factory._row = original
    if len(captured) != 1:
        raise ValueError("F1 root did not produce exactly one label row")
    record["label_row_sha256"] = hashlib.sha256(
        json.dumps(captured[0], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return record


def _prepare(
    bank: dict[str, Any], b8: Path, frp3: Path, cache: Path, deadline: float
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for source in bank["source_sessions"]:
        roots = [row for row in bank["roots"] if row["source_session"] == source["name"]]
        paths.update(f1_ruler._prepare_source(source, roots, b8, frp3, cache, deadline))
    return paths


def _integrity(
    record: dict[str, Any], expected_root: str, known_digest: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    if record.get("root_id") != expected_root:
        raise ValueError("worker returned a foreign root")
    if record.get("bare_audit_mismatches") != 0 or record.get("digest_checks") != 1:
        raise ValueError("F1 root integrity failure")
    digest = record.get("digest")
    label_hash = record.get("label_row_sha256")
    if not isinstance(digest, str) or not isinstance(label_hash, str):
        raise ValueError("F1 root lacks live digest or label-row hash")
    prior = known_digest.setdefault(expected_root, (digest, label_hash))
    if prior != (digest, label_hash):
        raise ValueError("F1 live digest or label-row hash differs across repeats")
    return {
        "root_id": expected_root,
        "digest": digest,
        "label_row_sha256": label_hash,
        "digest_checks": 1,
        "bare_audit_mismatches": 0,
        "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
    }


def _write_once(path: Path, payload: bytes) -> None:
    """Create one marker without allowing a reconnect to replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _claim_empty_output(path: Path) -> None:
    """The supervisor may pre-create the fresh output root, never its contents."""
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("qualification output path is reused or nonempty")
    else:
        path.mkdir(parents=True)


def publish_injection_locator(
    locator: Path,
    *,
    namespace: str,
    allocation: str,
    index: int,
    unit: str,
    request_id: str,
    descriptor_path: Path,
    runtime_root: Path,
) -> None:
    """Publish one immutable runtime-to-descriptor binding for an injected unit."""
    if unit != unit_id(namespace, allocation, index):
        raise ValueError("injection locator unit identity differs")
    if (
        not isinstance(request_id, str)
        or len(request_id) != 32
        or any(character not in "0123456789abcdef" for character in request_id)
    ):
        raise ValueError("injection locator request identity differs")
    descriptor_value = json.loads(descriptor_path.read_text())
    if any(
        descriptor_value.get(key) != value
        for key, value in {
            "namespace": namespace,
            "allocation": allocation,
            "index": index,
            "unit_id": unit,
            "request_id": request_id,
        }.items()
    ):
        raise ValueError("injection locator descriptor identity differs")
    descriptor_sha256 = _hash(descriptor_path)
    generation = runtime_root.name
    if not generation or not runtime_root.is_absolute():
        raise ValueError("injection locator runtime root is not an absolute generation root")
    _write_once(
        locator,
        (
            json.dumps(
                {
                    "schema": "rpc-qualification-injection-locator/v1",
                    "namespace": namespace,
                    "allocation": allocation,
                    "index": index,
                    "unit_id": unit,
                    "request_id": request_id,
                    "descriptor_path": str(descriptor_path),
                    "descriptor_sha256": descriptor_sha256,
                    "runtime_root": str(runtime_root),
                    "generation": generation,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )


def qualified_finish_deadline(finalize_seconds: float = 300.0) -> float:
    """Reserve finalization time from the supervisor's actual qualified session cap."""
    root = Path(os.environ["COLAB_CLI_ROOT"])
    spec = json.loads((root / "spec.json").read_text())
    state = json.loads((root / "state.json").read_text())
    elapsed = time.time() - float(state["started_at"])
    timeout = float(spec["timeout_seconds"])
    if not 0 <= elapsed < timeout - finalize_seconds:
        raise ValueError("qualified session lacks remaining finalization time")
    return time.monotonic() + timeout - elapsed - finalize_seconds


def prepare_injection(
    *,
    bank_path: Path,
    ledger_path: Path,
    b8: Path,
    frp3: Path,
    output: Path,
    namespace: str,
    allocation: str,
    source_closure_manifest: Path | None,
    source_closure_sha256: str | None,
    source_manifest_sha256: str | None = None,
    finish_deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    prepare: Callable[[dict[str, Any], Path, Path, Path, float], dict[str, Path]] | None = None,
) -> Path:
    """Publish three ready-to-RPC actual-root descriptors after full-bank preparation.

    The RPC client invokes :func:`execute_injected_descriptor`; it must never
    execute the label then merely fetch an already-produced result.
    """
    closure = _attest_closure(
        source_closure_manifest, source_closure_sha256, source_manifest_sha256
    )
    if os.environ.get("F1_COMMIT", F1_SOURCE_COMMIT) != F1_SOURCE_COMMIT:
        raise ValueError("F1 frozen source pin differs")
    bank_sha256 = f1_ruler.bank_digest(bank_path)
    bank = f1_ruler.load_bank(bank_path)
    if bank_sha256 != "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d":
        raise ValueError("injection requires the frozen F1 bank")
    _load_ledger(ledger_path, namespace, allocation, bank_sha256, source_closure_sha256)
    _claim_empty_output(output)
    cache = output / "root-cache"
    cache.mkdir()
    deadline = finish_deadline if finish_deadline is not None else clock() + 1800
    paths = (prepare or _prepare)(bank, b8, frp3, cache, deadline)
    descriptors = []
    for index, root in enumerate(bank["roots"][:3]):
        identity = unit_id(namespace, allocation, index)
        descriptor = {
            "schema": "rpc-qualification-injection-unit/v1",
            "namespace": namespace,
            "allocation": allocation,
            "bank_sha256": bank_sha256,
            "source_commit": F1_SOURCE_COMMIT,
            "source_closure": closure,
            "index": index,
            "unit_id": identity,
            "request_id": hashlib.sha256(("rpc-injection/" + identity).encode()).hexdigest()[:32],
            "root_id": root["root_id"],
            "cache_path": str(paths[root["root_id"]].resolve()),
            "unit_path": str((output / "units" / f"unit-{index:08d}.json").resolve()),
            "execution_dir": str((output / "execution").resolve()),
            "deadline_monotonic": deadline,
            "python_executable": sys.executable,
            "package_root": str(Path(__file__).resolve().parents[2]),
        }
        path = output / "descriptors" / f"unit-{index:08d}.json"
        publish_json(path, descriptor)
        locator = Path(locator_for_unit(identity))
        publish_injection_locator(
            locator,
            namespace=namespace,
            allocation=allocation,
            index=index,
            unit=identity,
            request_id=descriptor["request_id"],
            descriptor_path=path,
            runtime_root=Path(os.environ["COLAB_CLI_ROOT"]),
        )
        descriptors.append(
            {
                "index": index,
                "unit_id": descriptor["unit_id"],
                "request_id": descriptor["request_id"],
                "locator": str(locator),
                "namespace": namespace,
                "allocation": allocation,
            }
        )
    ready = {
        "schema": "rpc-qualification-injection-ready/v1",
        "status": "READY",
        "namespace": namespace,
        "allocation": allocation,
        "bank_sha256": bank_sha256,
        "source_commit": F1_SOURCE_COMMIT,
        "descriptors": descriptors,
        "outcomes": "not read or exported",
    }
    path = output / "injection-ready.json"
    publish_json(path, ready)
    # The runtime remains alive while the broker sends the real RPC commands.
    # A broker-written immutable proof is only a wake-up signal; all unit and
    # execution evidence below is checked independently before terminal export.
    proof = output / "injection-proof.json"
    while not Path(str(proof) + ".complete").is_file():
        if clock() >= deadline:
            raise TimeoutError("injected RPC proof did not arrive before its cap")
        wait(min(1.0, max(0.001, deadline - clock())))
    _verify(proof)
    units = []
    for index in range(3):
        unit = output / "units" / f"unit-{index:08d}.json"
        started = output / "execution" / f"unit-{index:08d}.started"
        done = output / "execution" / f"unit-{index:08d}.done"
        if not started.is_file() or not done.is_file():
            raise ValueError("injected unit lacks exactly-once execution markers")
        started_value, done_value = json.loads(started.read_text()), json.loads(done.read_text())
        if (
            started_value != done_value
            or started_value.get("request_id") != descriptors[index]["request_id"]
        ):
            raise ValueError("injected marker request identity differs")
        unit_sha = _verify(unit)
        payload = json.loads(unit.read_text())
        if (
            payload.get("unit_id") != unit_id(namespace, allocation, index)
            or payload.get("execution_count") != 1
        ):
            raise ValueError("injected unit identity/execution count differs")
        units.append(
            {
                "index": index,
                "unit_id": payload["unit_id"],
                "path": f"units/{unit.name}",
                "sha256": unit_sha,
                "size": unit.stat().st_size,
            }
        )
    ledger = empty_ledger(
        namespace, allocation, bank_sha256, F1_SOURCE_COMMIT, source_closure_sha256
    )
    ledger["units"] = units
    publish_json(output / "verified-ledger.json", ledger)
    publish_json(
        output / "runtime-result.json",
        {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "mode": "injection",
            "namespace": namespace,
            "allocation": allocation,
            "bank_sha256": bank_sha256,
            "source_commit": F1_SOURCE_COMMIT,
            "source_closure": closure,
            "injected_units": 3,
            "verified_ledger": "verified-ledger.json",
            "proof": "injection-proof.json",
            "outcomes": "not read or exported",
        },
    )
    return path


def execute_injected_descriptor(
    descriptor_path: Path, request_id: str | None = None
) -> dict[str, Any]:
    """Execute exactly one injected unit; safe reconnects only read a completed unit."""
    _verify(descriptor_path)
    spec = json.loads(Path(descriptor_path).read_text())
    if (
        spec.get("schema") != "rpc-qualification-injection-unit/v1"
        or spec.get("source_commit") != F1_SOURCE_COMMIT
    ):
        raise ValueError("foreign injected unit descriptor")
    index = spec.get("index")
    if not isinstance(index, int) or spec.get("unit_id") != unit_id(
        spec.get("namespace", ""), spec.get("allocation", ""), index
    ):
        raise ValueError("foreign injected unit identity")
    if request_id is not None and request_id != spec.get("request_id"):
        raise ValueError("injected request identity differs")
    execution = Path(spec["execution_dir"])
    started, done = execution / f"unit-{index:08d}.started", execution / f"unit-{index:08d}.done"
    unit = Path(spec["unit_path"])
    if done.exists():
        _verify(unit)
        payload = json.loads(unit.read_text())
        started_value = json.loads(started.read_text()) if started.is_file() else {}
        done_value = json.loads(done.read_text())
        if (
            started_value != done_value
            or started_value.get("request_id") != spec["request_id"]
            or started_value.get("execution_count") != 1
            or payload.get("execution_count") != 1
        ):
            raise ValueError("injected unit duplicate execution")
        return payload
    try:
        _write_once(
            started,
            (
                json.dumps(
                    {
                        "unit_id": spec["unit_id"],
                        "request_id": spec["request_id"],
                        "execution_count": 1,
                    }
                )
                + "\n"
            ).encode(),
        )
    except FileExistsError as error:
        raise RuntimeError("injected request is already running; do not relaunch") from error
    record = _label_root_with_hash((spec["cache_path"], float(spec["deadline_monotonic"])))
    integrity = _integrity(record, spec["root_id"], {})
    started_value = json.loads(started.read_text())
    if started_value != {
        "unit_id": spec["unit_id"],
        "request_id": spec["request_id"],
        "execution_count": 1,
    }:
        raise ValueError("injected execution marker identity differs")
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "namespace": spec["namespace"],
        "allocation": spec["allocation"],
        "index": index,
        "unit_id": spec["unit_id"],
        "request_id": spec["request_id"],
        "root_id": spec["root_id"],
        "execution_count": started_value["execution_count"],
        "integrity": integrity,
    }
    publish_json(unit, payload)
    _write_once(done, (json.dumps(started_value, sort_keys=True) + "\n").encode())
    return payload


def run(
    *,
    bank_path: Path,
    ledger_path: Path,
    b8: Path,
    frp3: Path,
    output: Path,
    platform: str,
    workers: int,
    namespace: str,
    allocation: str,
    loop_seconds: float,
    source_closure_manifest: Path | None,
    source_closure_sha256: str | None,
    source_manifest_sha256: str | None = None,
    finish_deadline: float | None = None,
    injection: bool = False,
    clock: Callable[[], float] = time.monotonic,
    prepare: Callable[[dict[str, Any], Path, Path, Path, float], dict[str, Path]] | None = None,
    execute: Callable[[tuple[str, float]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare the whole frozen bank, then repeatedly label roots until the deadline.

    ``execute`` is deliberately injectable for deterministic no-gameplay tests.
    Production uses an initialized spawn pool so the configured W2/W12 shape is
    the actual root execution shape.
    """
    if platform not in {"cpu", "l4"} or workers != {"cpu": 2, "l4": 12}.get(platform):
        raise ValueError("qualification requires CPU W2 or L4 W12")
    if not injection and loop_seconds < 3600:
        raise ValueError("qualification loop must be at least 60 minutes")
    if injection and not 0 < loop_seconds <= 1800:
        raise ValueError("injection loop must be at most 30 minutes")
    closure = _attest_closure(
        source_closure_manifest, source_closure_sha256, source_manifest_sha256
    )
    if os.environ.get("F1_COMMIT", F1_SOURCE_COMMIT) != F1_SOURCE_COMMIT:
        raise ValueError("F1 frozen source pin differs")
    bank_sha256 = f1_ruler.bank_digest(bank_path)
    bank = f1_ruler.load_bank(bank_path)
    if bank_sha256 != "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d":
        raise ValueError("qualification requires the frozen F1 bank")
    if len(bank["roots"]) != ROOTS:
        raise ValueError("qualification requires all 64 F1 roots")
    verified = _load_ledger(ledger_path, namespace, allocation, bank_sha256, source_closure_sha256)
    _claim_empty_output(output)
    cache = output / "root-cache"
    cache.mkdir()
    started = clock()
    # Preparation is setup, outside the registered 60-minute bank-loop clock.
    # A caller with a qualified whole-session deadline supplies it explicitly.
    work_deadline = (
        finish_deadline if finish_deadline is not None else started + loop_seconds + 7200
    )
    paths = (prepare or _prepare)(bank, b8, frp3, cache, work_deadline)
    if set(paths) != {root["root_id"] for root in bank["roots"]}:
        raise ValueError("frozen F1 preparation did not produce the complete bank")
    records: list[dict[str, Any]] = []
    known_digests: dict[str, tuple[str, str]] = {}
    for row in verified.values():
        source = ledger_path.parent / row["path"]
        payload = json.loads(source.read_text())
        expected_root = bank["roots"][row["index"] % ROOTS]["root_id"]
        if (
            payload.get("root_id") != expected_root
            or payload.get("platform") != platform
            or payload.get("workers") != workers
        ):
            raise ValueError("verified unit has foreign root/platform/worker allocation")
        _integrity(payload["integrity"], expected_root, known_digests)
        target = output / row["path"]
        _materialize_triplet(source, target)
        records.append({"index": row["index"], "path": target, "root_id": expected_root})
    index = 0
    executor = execute or _label_root_with_hash
    pool: Any = None
    if execute is None:
        context = mp.get_context("spawn")
        barrier = context.Barrier(workers + 1)
        pool = context.Pool(workers, initializer=f1_ruler._warmup, initargs=(barrier,))
        barrier.wait(timeout=max(0.001, work_deadline - clock()))
    # Pool spawn/import/numba warmup is setup, not F1-bank collection time.
    loop_started = clock()
    deadline = loop_started + loop_seconds
    if work_deadline <= deadline:
        raise ValueError("finish deadline must leave time to drain admitted work")
    next_progress = [loop_started + 110]

    def commit(current: int, root: dict[str, Any], record: dict[str, Any]) -> None:
        """Publish one completed root after it has actually finished."""
        integrity = _integrity(record, root["root_id"], known_digests)
        payload = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "namespace": namespace,
            "allocation": allocation,
            "index": current,
            "unit_id": unit_id(namespace, allocation, current),
            "platform": platform,
            "workers": workers,
            "root_id": root["root_id"],
            "integrity": integrity,
        }
        path = output / "units" / f"unit-{current:08d}.json"
        publish_json(path, payload)
        records.append({"index": current, "path": path, "root_id": root["root_id"]})
        now = clock()
        if now >= next_progress[0]:
            print(
                json.dumps(
                    {
                        "rpc_qualification_progress": True,
                        "namespace": namespace,
                        "completed": len(records),
                        "elapsed_seconds": now - loop_started,
                    }
                ),
                flush=True,
            )
            next_progress[0] = now + 110

    try:
        if pool is None:
            while clock() < deadline:
                root = bank["roots"][index % ROOTS]
                current = index
                index += 1
                if current not in verified:
                    commit(current, root, executor((str(paths[root["root_id"]]), work_deadline)))
        else:
            pending: list[tuple[int, dict[str, Any], Any]] = []
            admitting = True
            while admitting or pending:
                while admitting and len(pending) < workers and clock() < deadline:
                    root = bank["roots"][index % ROOTS]
                    current = index
                    index += 1
                    if current not in verified:
                        pending.append(
                            (
                                current,
                                root,
                                pool.apply_async(
                                    _label_root_with_hash,
                                    ((str(paths[root["root_id"]]), work_deadline),),
                                ),
                            )
                        )
                if clock() >= deadline:
                    admitting = False
                if pending:
                    current, root, future = pending.pop(0)
                    # Once admitted, finish this root even after the collection deadline.
                    commit(current, root, future.get())
                elif not admitting:
                    break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    units = []
    for item in records:
        path = item["path"]
        units.append(
            {
                "index": item["index"],
                "unit_id": unit_id(namespace, allocation, item["index"]),
                "path": f"units/{path.name}",
                "sha256": _verify(path),
                "size": path.stat().st_size,
            }
        )
    ledger_value = empty_ledger(
        namespace, allocation, bank_sha256, F1_SOURCE_COMMIT, source_closure_sha256
    )
    ledger_value["units"] = units
    # The ledger is the final mutable-collection publication and must precede a verdict/read.
    publish_json(output / "verified-ledger.json", ledger_value)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "namespace": namespace,
        "allocation": allocation,
        "platform": platform,
        "workers": workers,
        "bank_sha256": bank_sha256,
        "source_commit": F1_SOURCE_COMMIT,
        "source_closure": closure,
        "loop_seconds": loop_seconds,
        "measured_loop_seconds": clock() - loop_started,
        "elapsed_seconds": clock() - started,
        "units_completed": len(units),
        "root_repeat_counts": {
            root["root_id"]: sum(item["root_id"] == root["root_id"] for item in records)
            for root in bank["roots"]
        },
        "root_order": [root["root_id"] for root in bank["roots"]],
        "integrity": {
            "all_attempted_roots_clean": True,
            "repeat_digest_consistent": True,
            "digest_checks": len(units),
            "bare_audit_mismatches": 0,
        },
        "verified_ledger": "verified-ledger.json",
        "outcomes": "not read or exported",
    }
    publish_json(output / "runtime-result.json", result)
    return result


def main() -> None:
    """Small RPC-safe entry point; normal sessions call :func:`run` from colab.sh."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: str) -> argparse.ArgumentParser:
        item = sub.add_parser(command)
        item.add_argument("--bank", type=Path, required=True)
        item.add_argument("--ledger", type=Path, required=True)
        item.add_argument("--b8", type=Path, required=True)
        item.add_argument("--frp3", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--namespace", required=True)
        item.add_argument("--allocation", required=True)
        item.add_argument("--source-closure-manifest", type=Path, required=True)
        item.add_argument("--source-closure-sha256", required=True)
        item.add_argument("--source-manifest-sha256")
        return item

    normal = common("run")
    normal.add_argument("--platform", choices=("cpu", "l4"), required=True)
    normal.add_argument("--workers", type=int, required=True)
    normal.add_argument("--loop-seconds", type=float, required=True)
    normal.add_argument("--finish-deadline", type=float)
    injection = common("prepare-injection")
    # Accepted so colab.sh can pass one normalized runtime argument list.
    injection.add_argument("--platform", choices=("cpu", "l4"))
    injection.add_argument("--workers", type=int)
    injection.add_argument("--loop-seconds", type=float)
    unit = sub.add_parser("unit", help="execute one prepared injected F1 unit")
    unit.add_argument("--descriptor", type=Path, required=True)
    unit.add_argument("--request-id", required=True)
    args = parser.parse_args()
    if args.command == "run":
        print(
            json.dumps(
                run(
                    bank_path=args.bank,
                    ledger_path=args.ledger,
                    b8=args.b8,
                    frp3=args.frp3,
                    output=args.output,
                    platform=args.platform,
                    workers=args.workers,
                    namespace=args.namespace,
                    allocation=args.allocation,
                    loop_seconds=args.loop_seconds,
                    source_closure_manifest=args.source_closure_manifest,
                    source_closure_sha256=args.source_closure_sha256,
                    source_manifest_sha256=args.source_manifest_sha256,
                    finish_deadline=args.finish_deadline or qualified_finish_deadline(),
                ),
                sort_keys=True,
            )
        )
    elif args.command == "prepare-injection":
        print(
            prepare_injection(
                bank_path=args.bank,
                ledger_path=args.ledger,
                b8=args.b8,
                frp3=args.frp3,
                output=args.output,
                namespace=args.namespace,
                allocation=args.allocation,
                source_closure_manifest=args.source_closure_manifest,
                source_closure_sha256=args.source_closure_sha256,
                source_manifest_sha256=args.source_manifest_sha256,
                finish_deadline=qualified_finish_deadline(),
            )
        )
    else:
        value = execute_injected_descriptor(args.descriptor, args.request_id)
        print(
            json.dumps({"unit_id": value["unit_id"], "execution_count": value["execution_count"]})
        )


if __name__ == "__main__":
    main()
