"""Generate immutable session units using the adopted Watch6 exact-stop engine."""

from __future__ import annotations

import argparse
import faulthandler
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tools.ops.package_spec import publish_immutable, publish_json, read_ledger, unit_id
from web.backend.watch_replay import (
    B8_SHA,
    CONTROLLERS,
    FRP3_SHA,
    MAX_DECODED,
    MAX_LINE,
    SCHEMA,
    canonical,
    verify_payload,
)

ROOT = Path(__file__).resolve().parents[2]
_STOP = None
MAX_WORKERS = 48
DEFAULT_CHECKPOINT_INTERVAL = 0
SESSION_CHECKPOINT_SCHEMA = "watch-replay-session-checkpoint/v1"


def initialize_worker(stop_event: object) -> None:
    """Share the stop fence across queued and currently running session units."""
    global _STOP
    _STOP = stop_event


def allocation(namespace: str, count: int, controller: str, horizon: int = 6000) -> dict:
    """Freeze identities before execution; no outcome-dependent session choice."""
    unit_id(namespace, "sessions", 0)
    if not 1 <= count <= 128 or controller not in CONTROLLERS:
        raise ValueError("invalid session allocation")
    if not 1 <= horizon <= 6000:
        raise ValueError("invalid horizon")
    return {
        "namespace": namespace,
        "count": count,
        "controller": controller,
        "horizon": horizon,
        "rows": [
            {
                "index": i,
                "seed": int.from_bytes(
                    hashlib.sha256(f"{namespace}/{i}".encode()).digest()[:4], "big"
                ),
            }
            for i in range(count)
        ],
    }


def identity(contract: dict) -> dict:
    """Bind every allocation dimension and the executable commit in the ledger."""
    return {"contract_sha256": hashlib.sha256(canonical(contract)).hexdigest()}


def completed(ledger: Path, contract: dict) -> list[dict]:
    rows = read_ledger(ledger, contract["namespace"], "sessions", identity(contract))
    if any(not 0 <= r["index"] < contract["count"] for r in rows):
        raise ValueError("foreign allocation index")
    return rows


def _load_session_checkpoint(
    contract: dict,
    row: dict,
    candidate_dirs: list[Path],
) -> dict[str, Any] | None:
    """Find and verify the newest intra-session checkpoint for `row['index']` across candidates."""
    from research.world_checkpoint.checkpoint import (
        CheckpointIdentityError,
        find_latest_checkpoint,
        load_checkpoint,
        verify_checkpoint_identity,
    )

    best_key: tuple[int, int] | None = None
    best_ckpt: dict[str, Any] | None = None
    seen_dirs: set[Path] = set()
    for cdir in candidate_dirs:
        if not cdir.is_dir():
            continue
        resolved = cdir.resolve()
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        latest = find_latest_checkpoint(cdir, verify_identity=False)
        if latest is None:
            continue
        ckpt = load_checkpoint(latest, verify_identity=True)
        verify_checkpoint_identity(ckpt)
        if (
            ckpt.get("session_checkpoint_schema") != SESSION_CHECKPOINT_SCHEMA
            or ckpt.get("allocation") != identity(contract)
            or ckpt.get("namespace") != contract["namespace"]
            or int(ckpt.get("index", -1)) != int(row["index"])
            or int(ckpt.get("seed", -1)) != int(row["seed"])
            or ckpt.get("controller") != contract["controller"]
            or int(ckpt.get("horizon", -1)) != int(contract["horizon"])
        ):
            raise CheckpointIdentityError(
                f"foreign or mismatched watch-replay session checkpoint: {latest}"
            )
        key = (int(ckpt["frame"]), int(ckpt.get("checkpoint_seq", 0)))
        if best_key is None or key > best_key:
            best_key = key
            best_ckpt = ckpt
    return best_ckpt


def generate(task: tuple) -> dict:
    """Publish one all-or-nothing session; partial units checkpoint and never mark complete."""
    from research.watch_replay.engine import run_arm

    if len(task) == 3:
        contract, row, directory = task
        checkpoint_in_dir = None
        checkpoint_interval = DEFAULT_CHECKPOINT_INTERVAL
    elif len(task) == 5:
        contract, row, directory, checkpoint_in_dir, checkpoint_interval = task
    else:
        raise ValueError("invalid generate task tuple")

    out = Path(directory)
    path = out / f"{row['index']:06d}.watch.gz"
    if path.exists() or Path(str(path) + ".complete").exists():
        raise FileExistsError("never overwrite or recompute a completed unit")
    started, last = time.monotonic(), time.monotonic()
    faulthandler.dump_traceback_later(900, exit=True)
    stop = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def stop_requested() -> bool:
        return bool(stop or (_STOP is not None and _STOP.is_set()))

    def check() -> None:
        nonlocal last
        if stop_requested():
            raise InterruptedError("mechanical stop; partial session discarded")
        if time.monotonic() - last >= 60:
            print(
                f"watch-replay index={row['index']} working "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
            last = time.monotonic()

    check.stop_requested = stop_requested  # type: ignore[attr-defined]

    from research.world_checkpoint.checkpoint import (
        highest_checkpoint_seq_on_disk,
        quarantine_incomplete_checkpoints,
    )

    unit_tag = f"{row['index']:06d}"
    out_ckpt_dir = out / "checkpoints" / unit_tag
    if out_ckpt_dir.is_dir():
        quarantine_incomplete_checkpoints(out_ckpt_dir)
    candidate_ckpt_dirs: list[Path] = []
    if checkpoint_in_dir:
        in_root = Path(checkpoint_in_dir)
        candidate_ckpt_dirs.extend(
            [
                in_root / unit_tag,
                in_root / "checkpoints" / unit_tag,
                in_root / "sessions" / "checkpoints" / unit_tag,
            ]
        )
    candidate_ckpt_dirs.append(out_ckpt_dir)
    loaded_ckpt = _load_session_checkpoint(contract, row, candidate_ckpt_dirs)

    decoded_bytes = 0
    peak_line_bytes = 0
    ckpt_seq = highest_checkpoint_seq_on_disk(out_ckpt_dir, *candidate_ckpt_dirs)
    resume_state: dict[str, Any] | None = None
    with tempfile.TemporaryFile() as frames:
        if loaded_ckpt is not None:
            raw_prefix = gzip.decompress(loaded_ckpt["recorded_jsonl_gz"])
            if hashlib.sha256(raw_prefix).hexdigest() != loaded_ckpt[
                "recorded_jsonl_sha256"
            ] or len(raw_prefix) != int(loaded_ckpt["decoded_bytes"]):
                raise ValueError("corrupted recorded_jsonl_gz in session checkpoint")
            frames.write(raw_prefix)
            decoded_bytes = int(loaded_ckpt["decoded_bytes"])
            peak_line_bytes = int(loaded_ckpt["peak_line_bytes"])
            ckpt_seq = max(ckpt_seq, int(loaded_ckpt["checkpoint_seq"]))
            resume_state = dict(loaded_ckpt["engine_state"])

        def record(frame: dict) -> None:
            nonlocal decoded_bytes, peak_line_bytes
            # Keep gameplay/inspector/veto counters byte-stable across workers.
            # Diagnostic v8 wall times describe generation hardware, not gameplay,
            # and are not fields consumed by the browser's protocol contract.
            veto = frame["session"].get("safety_veto")
            if isinstance(veto, dict):
                veto.pop("diagnostics", None)
            line = canonical(frame)
            decoded_bytes += len(line)
            peak_line_bytes = max(peak_line_bytes, len(line))
            # Reserve one maximum-size header before spending more search compute.
            if len(line) > MAX_LINE or decoded_bytes + MAX_LINE > MAX_DECODED:
                raise ValueError("incremental replay size limit; partial session discarded")
            frames.write(line)
            # Completed real-frame progress, never mere heartbeat, renews the watchdog.
            faulthandler.dump_traceback_later(900, exit=True)

        def on_checkpoint(frame_idx: int, engine_state: dict[str, Any]) -> None:
            nonlocal ckpt_seq
            from research.world_checkpoint.checkpoint import (
                WORLD_CHECKPOINT_SCHEMA,
                capture_runtime_identity,
                save_checkpoint,
            )

            frames.seek(0)
            raw_jsonl = frames.read()
            frames.seek(0, io.SEEK_END)
            if len(raw_jsonl) != decoded_bytes:
                raise ValueError("recorded frame stream size mismatch during checkpoint")
            ckpt_seq = (
                max(ckpt_seq, highest_checkpoint_seq_on_disk(out_ckpt_dir, *candidate_ckpt_dirs))
                + 1
            )
            out_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_name = (
                f"ckpt-seq-{ckpt_seq:04d}-arm-{contract['controller']}-frame-{frame_idx:06d}.ckpt"
            )
            ckpt_path = out_ckpt_dir / ckpt_name
            payload_state = {
                "checkpoint_schema": WORLD_CHECKPOINT_SCHEMA,
                "session_checkpoint_schema": SESSION_CHECKPOINT_SCHEMA,
                "checkpoint_seq": ckpt_seq,
                "identity": capture_runtime_identity(),
                "allocation": identity(contract),
                "namespace": contract["namespace"],
                "index": int(row["index"]),
                "seed": int(row["seed"]),
                "controller": contract["controller"],
                "horizon": int(contract["horizon"]),
                "frame": int(frame_idx),
                "decoded_bytes": int(decoded_bytes),
                "peak_line_bytes": int(peak_line_bytes),
                "recorded_jsonl_sha256": hashlib.sha256(raw_jsonl).hexdigest(),
                "recorded_jsonl_gz": gzip.compress(raw_jsonl, compresslevel=6, mtime=0),
                "engine_state": engine_state,
            }
            # Publish create-only under $COLAB_OUTPUT_DIR/sessions/checkpoints/<unit_tag>;
            # never unlink or rmtree published .complete exports (runtime.py:335).
            save_checkpoint(payload_state, ckpt_path)
            faulthandler.dump_traceback_later(900, exit=True)

        try:
            use_ckpt = (resume_state is not None) or (
                checkpoint_interval > 0 and int(contract["horizon"]) > checkpoint_interval
            )
            if use_ckpt:
                raw = run_arm(
                    row["seed"],
                    contract["controller"],
                    check,
                    horizon=contract["horizon"],
                    record=record,
                    checkpoint_interval=checkpoint_interval,
                    on_checkpoint=on_checkpoint,
                    resume_state=resume_state,
                )
            else:
                raw = run_arm(
                    row["seed"],
                    contract["controller"],
                    check,
                    horizon=contract["horizon"],
                    record=record,
                )
            stats = {
                "hero_mass_integral": sum(f["mass"] for f in raw["frames"]) / contract["horizon"],
                "hero_deaths": sum(f["death"] for f in raw["frames"]),
                "max_hero_length": max(f["mass"] for f in raw["frames"]),
                "decisions": len(raw["decisions"]),
                "search_roots": len(raw["batches"]),
                "interventions": sum(b["chosen"] != b["bare"] for b in raw["batches"]),
                "mirror_identity_frames": raw["mirror_identity_frames"],
            }
            header = {
                "schema": SCHEMA,
                "protocol_version": 2,
                "world": "watch6-mechanics-v2-continuous-respawn",
                "unit_id": unit_id(contract["namespace"], "sessions", row["index"]),
                "index": row["index"],
                "seed": row["seed"],
                "horizon": contract["horizon"],
                "controller": contract["controller"],
                "checkpoint_shas": {"hero": B8_SHA, "opponents": FRP3_SHA},
                "source_commit": contract["commit"],
                "allocation": identity(contract),
                "fixture": contract["horizon"] != 6000,
                "stats": stats,
            }
            buffer = io.BytesIO()
            with gzip.GzipFile(
                fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=6
            ) as zipped:
                zipped.write(canonical(header))
                frames.seek(0)
                while chunk := frames.read(1024**2):
                    zipped.write(chunk)
            payload = buffer.getvalue()
            verify_payload(payload, product=not header["fixture"])
            result = publish_immutable(path, payload)
        finally:
            faulthandler.cancel_dump_traceback_later()
    seconds = time.monotonic() - started
    print(f"watch-replay index={row['index']} complete seconds={seconds:.3f}", flush=True)
    return {
        **result,
        "index": row["index"],
        "seconds": seconds,
        "decoded_bytes": decoded_bytes + len(canonical(header)),
        "peak_line_bytes": max(peak_line_bytes, len(canonical(header))),
        "stats": stats,
    }


def run(
    contract: dict,
    ledger: Path,
    out: Path,
    workers: int,
    *,
    checkpoint_in: Path | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> list[dict]:
    """Skip only uploaded Mac-verified units; merge in frozen allocation order."""
    if type(workers) is not int or not 1 <= workers <= min(os.cpu_count() or 1, MAX_WORKERS):
        raise ValueError("invalid worker count")
    verified = completed(ledger, contract)
    skip = {r["index"] for r in verified}
    out.mkdir(parents=True, exist_ok=False)
    publish_json(out / "contract.json", contract)
    if checkpoint_in is None and checkpoint_interval == DEFAULT_CHECKPOINT_INTERVAL:
        tasks: list[tuple] = [
            (contract, row, str(out)) for row in contract["rows"] if row["index"] not in skip
        ]
    else:
        tasks = [
            (
                contract,
                row,
                str(out),
                None if checkpoint_in is None else str(checkpoint_in),
                int(checkpoint_interval),
            )
            for row in contract["rows"]
            if row["index"] not in skip
        ]
    rows, exclusions = [], []

    def collect(task: tuple, result: dict | None = None, error: Exception | None = None) -> None:
        if error is not None:
            exclusions.append(
                {"index": task[1]["index"], "error": str(error), "kind": type(error).__name__}
            )
            print(f"watch-replay excluded index={task[1]['index']}: {error}", flush=True)
        else:
            rows.append(result)

    try:
        if workers == 1:
            for task in tasks:
                try:
                    collect(task, generate(task))
                except Exception as exc:
                    collect(task, error=exc)
                    if isinstance(exc, InterruptedError):
                        break
        else:
            context = multiprocessing.get_context("spawn")
            stop_event = context.Event()
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=initialize_worker,
                initargs=(stop_event,),
            ) as pool:

                def stop_pool(signum: int, frame: object) -> None:
                    stop_event.set()

                signal.signal(signal.SIGUSR1, stop_pool)
                signal.signal(signal.SIGTERM, stop_pool)
                futures = {pool.submit(generate, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        collect(task, future.result())
                    except Exception as exc:
                        collect(task, error=exc)
    finally:
        rows.sort(key=lambda row: row["index"])
        publish_json(
            out / "operational.json",
            {
                "allocation": identity(contract),
                "completed": len(rows),
                "skipped": sorted(skip),
                "total": contract["count"],
                "exclusions": sorted(exclusions, key=lambda row: row["index"]),
                "units": rows,
            },
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    expected = allocation(
        contract["namespace"], contract["count"], contract["controller"], contract["horizon"]
    )
    if contract != {**expected, "commit": contract["commit"]}:
        raise ValueError("malformed frozen contract")
    print(
        json.dumps(
            run(
                contract,
                args.ledger,
                args.out,
                args.workers,
                checkpoint_in=args.checkpoint_in,
                checkpoint_interval=args.checkpoint_interval,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
