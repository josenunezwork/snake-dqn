"""Claim A end-to-end driver: student labels -> payload -> cache -> fit -> probe.

One entrypoint for the Mac smoke and the (CUDA) L4 qualification.  Phases:

1. ``label``  - optional student-policy label chunk (CPU process fan-out) via
   :mod:`research.t34_claim_a.label`, wall-bounded by ``--label-seconds``.
2. ``payload`` - hash-bound world index over the new chunk plus any existing
   exports (``--extra-export``), with an explicit train/held-out split and a
   minimum held-out decisive-row precondition.
3. ``fit``    - the ratified ``research.t34_exit_b8_fit.runner.run`` (mode
   ``fit``) with this package's production ``source_factory``/``probe_factory``.
   It builds the observation cache once, then fits on ``--device`` (cpu|cuda)
   with the frozen loss, held-out decisive agreement, probe and tripwires.

Receipts.  ``pipeline.json`` (+ ``.sha256``/``.complete``) is rewritten
atomically in ``--receipt-dir`` after every phase and every probe epoch, and on
SIGUSR1/SIGTERM (the runtime's wall-cap stop), so a killed run still leaves
its timings.  ``final: true`` marks the last write.  Statuses:
``FIT_COMPLETED`` / ``TIMING_ONLY_FIT_COMPLETED`` / ``FIT_STOPPED`` (ratified
tripwire) / ``PRECONDITION_FAILED`` (data cannot support the fit; never a
tripwire) / ``LABEL_INCOMPLETE`` / ``FAILED`` (exception) / ``TERMINATED``.

Timing-only mode (``--timing-only-no-tripwires``) records rather than enforces
the ratified tripwires and **exports no weights**: the selected ``.pth`` files
are deleted after hashing and replaced by a ``TIMING_ONLY_NOT_A_CANDIDATE``
marker.  Smoke-sized probes are stamped ``SMOKE_PROBE_SHAPE``.  Nothing here
carries a scientific or promotion claim.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)
FINISHED = {"FIT_COMPLETED", "FIT_STOPPED", "TIMING_ONLY_FIT_COMPLETED"}
EMPTY_HELDOUT = "held-out decisive stream is empty"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


class Receipt:
    """``pipeline.json`` with sidecars, atomically rewritten as the run progresses."""

    def __init__(self, directory: Path, report: dict[str, Any]) -> None:
        self.path = directory / "pipeline.json"
        self.report = report
        self.started = time.monotonic()
        directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"receipt already exists: {self.path}")

    def write(self, *, final: bool = False, **updates: Any) -> None:
        self.report.update(updates)
        self.report["timings"]["total_seconds"] = time.monotonic() - self.started
        self.report["final"] = final
        self.report["written_unix"] = time.time()
        encoded = (
            json.dumps(self.report, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
        ).encode()
        _atomic_write(self.path, encoded)
        _atomic_write(
            self.path.with_name(self.path.name + ".sha256"),
            f"{hashlib.sha256(encoded).hexdigest()}  {self.path.name}\n".encode(),
        )
        _atomic_write(self.path.with_name(self.path.name + ".complete"), b"complete\n")


@contextlib.contextmanager
def stop_signals(receipt: Receipt) -> Iterator[None]:
    """Persist the receipt on the runtime's SIGUSR1 (stop requested) / SIGTERM."""

    def requested(signum: int, _frame: Any) -> None:
        receipt.report.setdefault("signals", []).append(signal.Signals(signum).name)
        receipt.write()

    def terminated(signum: int, _frame: Any) -> None:
        receipt.report.setdefault("signals", []).append(signal.Signals(signum).name)
        receipt.write(final=True, status="TERMINATED")
        raise SystemExit(143)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    signal.signal(signal.SIGUSR1, requested)
    signal.signal(signal.SIGTERM, terminated)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def smoke_probe_shape(worlds: int, horizon: int) -> Iterator[None]:
    """Temporarily validate the probe at a declared smoke shape (local wiring only)."""
    from research.t34_claim_a.probe import (
        PRODUCTION_HORIZON,
        PRODUCTION_WORLDS,
        validate_smoke_probe,
    )
    from research.t34_exit_b8_fit import train

    if (worlds, horizon) == (PRODUCTION_WORLDS, PRODUCTION_HORIZON):
        yield
        return
    original = train.validate_probe
    train.validate_probe = lambda probe: validate_smoke_probe(  # type: ignore[assignment]
        probe, worlds=worlds, horizon=horizon
    )
    try:
        yield
    finally:
        train.validate_probe = original


@contextlib.contextmanager
def record_tripwires(enabled: bool, sink: list[dict[str, Any]]) -> Iterator[None]:
    """Timing-only mode: evaluate the ratified tripwires but record instead of stopping.

    The ratified ``enforce_tripwires`` still runs on every epoch; its verdict is
    kept in ``sink``.  A run in this mode is a throughput/plumbing receipt and
    exports no weights.
    """
    from research.t34_exit_b8_fit import train
    from research.t34_exit_b8_fit.diagnostics import FitTripwire

    if not enabled:
        yield
        return
    original = train.enforce_tripwires

    def recorder(**kwargs: Any) -> None:
        try:
            original(**kwargs)
            sink.append({"fired": None, "heldout": kwargs["heldout_decisive_agreement"]})
        except FitTripwire as exc:
            sink.append({"fired": str(exc), "heldout": kwargs["heldout_decisive_agreement"]})

    train.enforce_tripwires = recorder  # type: ignore[assignment]
    try:
        yield
    finally:
        train.enforce_tripwires = original


@contextlib.contextmanager
def probe_callback(callback: Any) -> Iterator[None]:
    from research.t34_claim_a import probe

    probe._ON_CALL.append(callback)
    try:
        yield
    finally:
        probe._ON_CALL.remove(callback)


def cuda_numerics(device: str) -> dict[str, Any]:
    """Full-fp32 CUDA for the whole fit (labels were recorded without TF32)."""
    import torch

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "device": device,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def previous_heldout_agreement(cache_dir: Path, model: Any, device: str) -> dict[str, Any] | None:
    """Raw-argmax agreement of the *previous* student on cached held-out decisive rows."""
    from research.t34_exit_b8_fit import train
    from research.t34_exit_b8_fit.cache import CachedReplaySource

    manifest = cache_dir / "manifest.json"
    if not manifest.is_file() or not json.loads(manifest.read_text()).get("heldout"):
        return None
    identity = json.loads(manifest.read_text())["identity"]
    import torch

    source = CachedReplaySource(cache_dir, identity=identity)
    return train._heldout_metrics(model.to(device), source, torch.device(device), 0)


def withhold_weights(fit_output: Path) -> dict[str, Any]:
    """Timing-only: hash then delete selected weights; leave an ineligibility marker."""
    removed = {}
    for path in sorted(fit_output.glob("*.pth")):
        removed[path.name] = _sha256(path)
        for sidecar in (path, path.with_name(path.name + ".sha256")):
            sidecar.unlink(missing_ok=True)
        path.with_name(path.name + ".complete").unlink(missing_ok=True)
    marker = {
        "status": "TIMING_ONLY_NOT_A_CANDIDATE",
        "reason": "tripwires recorded, not enforced; weights withheld (deleted after hashing)",
        "withheld_sha256": removed,
    }
    if fit_output.is_dir():
        with (fit_output / "TIMING_ONLY_NOT_A_CANDIDATE").open("x") as stream:
            json.dump(marker, stream, indent=2, sort_keys=True)
    return marker


def _parse_extra(value: str) -> tuple[Path, int, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--extra-export takes PATH:ROUND_AGE:CHUNK_ID")
    return Path(parts[0]), int(parts[1]), parts[2]


def _label_phase(args: argparse.Namespace, receipt: Receipt, carrier: Path, previous: Path):
    from research.t34_claim_a import label
    from research.t34_exit_b8.label import _write_once

    out: Path = args.out
    started = time.monotonic()
    label_config = {
        "checkpoint": carrier,
        "student_checkpoint": previous,
        "policy_factory": args.policy_factory,
        "output_dir": out / "label" / "worlds",
        "round": args.round,
        "shard": args.shard,
        "attempt": args.attempt,
        "worlds": args.label_worlds,
        "max_worlds": args.label_worlds,
        "frames": args.label_frames,
        "seconds": args.label_seconds,
        "horizon": 300,
        "threads": 1,
        "nproc": args.label_nproc,
        "device": "cpu",
        "kernel": "v2",
        "namespace": args.label_namespace,
    }
    if args.label_mixes:
        label_config["mixes"] = tuple(args.label_mixes.split(","))
    chunk = label.run_label_chunk(label_config)
    receipt.report["timings"]["label_seconds"] = time.monotonic() - started
    _write_once(out / "label" / "label.json", chunk)
    if args.receipt_dir.resolve() != out.resolve():
        _write_once(args.receipt_dir / "label" / "label.json", chunk)
    completed = sorted((out / "label" / "worlds").rglob("world-*.json.complete"))
    summary = {
        key: chunk.get(key)
        for key in (
            "complete",
            "worlds",
            "requested_worlds",
            "decisions",
            "decisions_per_second",
            "worlds_per_hour",
            "export_bytes",
            "mixes",
            "recipe",
            "student",
            "elapsed_seconds",
            "invalid_reason",
        )
    }
    summary["completed_world_files"] = len(completed)
    receipt.write(phase="label", label=summary)
    return chunk, len(completed)


def run_pipeline(args: argparse.Namespace, receipt: Receipt) -> None:
    out: Path = args.out
    carrier = args.carrier.resolve()
    previous = (args.previous or args.carrier).resolve()
    timings = receipt.report["timings"]
    receipt.report.update(
        carrier_sha256=_sha256(carrier), previous_sha256=_sha256(previous), phase="start"
    )
    receipt.write()

    # 1. label (wall-bounded; completed worlds are kept even if the chunk is not)
    sources: list[tuple[Path, int, str]] = []
    if args.label_worlds:
        chunk, completed = _label_phase(args, receipt, carrier, previous)
        if not chunk.get("complete") and completed < 2:
            receipt.write(final=True, status="LABEL_INCOMPLETE")
            return
        if not chunk.get("complete"):
            receipt.report["label_incomplete_continued_with_completed_worlds"] = completed
        sources.append((out / "label" / "worlds", 0, f"{args.round}-s{args.shard}-a{args.attempt}"))
    sources.extend(args.extra_export or [])
    if not sources:
        raise SystemExit("nothing to fit: pass --label-worlds and/or --extra-export")

    # 2. payload (explicit split + held-out decisive precondition) -----------
    from research.t34_claim_a.replay_source import (
        SOURCE_FACTORY,
        PreconditionFailed,
        build_payload,
        last_source,
        load_payload,
    )

    started = time.monotonic()
    exports = out / "engine-export"
    try:
        build_payload(
            sources,
            exports,
            heldout_fraction=args.heldout_fraction,
            heldout_ids=args.heldout_world_id or (),
            heldout_decisive=args.heldout_decisive,
            min_heldout_decisive_rows=args.min_heldout_decisive_rows,
        )
    except PreconditionFailed as exc:
        timings["payload_seconds"] = time.monotonic() - started
        receipt.write(final=True, status="PRECONDITION_FAILED", precondition=str(exc))
        return
    payload = load_payload(exports)
    timings["payload_seconds"] = time.monotonic() - started
    receipt.write(
        phase="payload",
        payload={
            "sha256": _sha256(exports / "payload.json"),
            "worlds": {
                split: sum(1 for w in payload["worlds"] if w["split"] == split)
                for split in ("train", "heldout")
            },
            "heldout_world_ids": [
                w["world_id"] for w in payload["worlds"] if w["split"] == "heldout"
            ],
            "split_rule": payload["split_rule"],
            "heldout_decisive_rows": payload["heldout_decisive_rows"],
            "min_heldout_decisive_rows": payload["min_heldout_decisive_rows"],
            "rollout_policy_sha256s": payload["rollout_policy_sha256s"],
        },
    )

    # 3. cache + fit + probe (ratified runner) -----------------------------
    from research.t34_claim_a.probe import PROBE_FACTORY, last_probe
    from research.t34_exit_b8_fit import runner

    binding = {
        "approved": False,
        "arm": args.arm,
        "seed": args.seed,
        "source_factory": SOURCE_FACTORY,
        "probe_factory": PROBE_FACTORY,
        "claim_a": {
            "carrier_checkpoint": str(carrier),
            "previous_sha256": receipt.report["previous_sha256"],
            "probe": {
                "namespace": args.probe_namespace,
                "worlds": args.probe_worlds,
                "horizon": args.probe_horizon,
                "workers": args.probe_workers,
                **({"mixes": args.probe_mixes.split(",")} if args.probe_mixes else {}),
            },
        },
    }
    cache_dir = args.cache_dir or out / "observation-cache"
    tripwire_log: list[dict[str, Any]] = []
    receipt.report["tripwires"] = {
        "mode": (
            "RECORDED_NOT_ENFORCED_TIMING_ONLY"
            if args.timing_only_no_tripwires
            else "RATIFIED_ENFORCED"
        ),
        "per_epoch": tripwire_log,
    }
    receipt.report["probe_calls"] = []

    def cache_stats() -> dict[str, Any] | None:
        manifest = cache_dir / "manifest.json"
        if not manifest.is_file():
            return None
        stats = json.loads(manifest.read_text())["stats"]
        timings["cache_build_seconds"] = float(stats["regen_seconds"])
        keep = ("regen_seconds", "rows", "training_rows", "heldout_rows", "bytes")
        return {
            **{key: stats.get(key) for key in keep},
            "bytes_per_row": stats.get("bytes_per_row"),
            "regen_seconds_per_row": stats.get("regen_seconds_per_row"),
            "training": stats.get("training"),
            "heldout": stats.get("heldout"),
        }

    def on_probe(call: dict[str, Any]) -> None:
        receipt.report["probe_calls"].append(call)
        source = last_source()
        receipt.write(
            phase=f"fit-epoch-{call['epoch']}",
            cache=cache_stats(),
            source=source.timings.as_dict() if source else None,
        )

    started = time.monotonic()
    receipt.write(phase="fit", binding=binding)
    with (
        smoke_probe_shape(args.probe_worlds, args.probe_horizon),
        record_tripwires(args.timing_only_no_tripwires, tripwire_log),
        probe_callback(on_probe),
    ):
        result = runner.run(
            exports=exports,
            checkpoint=previous,
            output_dir=out / "fit-output",
            worlds=receipt.report["payload"]["worlds"]["train"],
            updates=0,
            device=args.device,
            mode="fit",
            epochs=args.epochs,
            fake_engine=False,
            binding=binding,
            cache_dir=cache_dir,
        )
    timings["fit_runner_seconds"] = time.monotonic() - started
    source, probe = last_source(), last_probe()
    stats = cache_stats()
    probe_seconds = sum(call["seconds"] for call in (probe.calls if probe else []))
    timings["probe_seconds"] = probe_seconds
    timings["fit_and_heldout_seconds"] = max(
        0.0,
        timings["fit_runner_seconds"] - timings.get("cache_build_seconds", 0.0) - probe_seconds,
    )
    baseline = None
    if source is not None:
        try:
            baseline = previous_heldout_agreement(cache_dir, source.previous_model, args.device)
        except ValueError as exc:
            baseline = {"unavailable": str(exc)}
    fit = {
        key: result.get(key)
        for key in (
            "completed",
            "tripwire",
            "probes_ok",
            "history",
            "selected",
            "selected_checkpoint",
            "updates",
            "fit_timing",
            "seed",
        )
    }
    if str(result.get("tripwire") or "").startswith(EMPTY_HELDOUT):
        status = "PRECONDITION_FAILED"  # ratified runner reports it as a tripwire
    elif not result.get("completed") or result.get("tripwire"):
        status = "FIT_STOPPED"
    elif args.timing_only_no_tripwires:
        status = "TIMING_ONLY_FIT_COMPLETED"
    else:
        status = "FIT_COMPLETED"
    if args.timing_only_no_tripwires:
        receipt.report["claim"] = "TIMING_ONLY_TRIPWIRES_RECORDED_NO_WEIGHTS_EXPORTED"
        fit["selected_checkpoint"] = None
        fit["weights_withheld"] = withhold_weights(out / "fit-output")
    receipt.write(
        final=True,
        phase="done",
        status=status,
        previous_student_heldout=baseline,
        source=source.timings.as_dict() if source else None,
        cache=stats,
        probe_calls=probe.calls if probe else [],
        probe_shape=(
            "PRODUCTION_32xH5000"
            if (args.probe_worlds, args.probe_horizon) == (32, 5000)
            else f"SMOKE_PROBE_SHAPE_{args.probe_worlds}xH{args.probe_horizon}"
        ),
        fit=fit,
    )


def _default_probe_workers() -> int:
    count = max(1, int(os.cpu_count() or 1))
    if os.uname().sysname == "Darwin":
        return min(2, count)  # shared Mac: at most two CPU threads
    return max(1, count - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument(
        "--receipt-dir", type=Path, help="where pipeline.json is kept current (default: --out)"
    )
    parser.add_argument("--carrier", type=Path, required=True, help="B8 world carrier checkpoint")
    parser.add_argument("--previous", type=Path, help="previous student (default: carrier)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arm", choices=("S1b", "S1a"), default="S1b")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path)
    # label phase
    parser.add_argument("--label-worlds", type=int, default=0)
    parser.add_argument("--label-frames", type=int, default=5000)
    parser.add_argument("--label-seconds", type=float, default=2400)
    parser.add_argument("--label-nproc", type=int, default=2)
    parser.add_argument("--label-namespace", default="t34-claim-a/qualification/label")
    parser.add_argument("--label-mixes")
    parser.add_argument("--policy-factory", default="research.t34_claim_a.student:student_policy")
    parser.add_argument("--round", default="r1")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    # payload
    parser.add_argument("--extra-export", type=_parse_extra, action="append")
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--heldout-world-id", action="append")
    parser.add_argument(
        "--heldout-decisive", type=int, default=0, help="hold out K decisive-bearing worlds"
    )
    parser.add_argument("--min-heldout-decisive-rows", type=int, default=5)
    # probe
    parser.add_argument("--probe-namespace", default="t34-claim-a/qualification/probe")
    parser.add_argument("--probe-worlds", type=int, default=32)
    parser.add_argument("--probe-horizon", type=int, default=5000)
    parser.add_argument("--probe-mixes")
    parser.add_argument("--probe-workers", type=int, default=_default_probe_workers())
    parser.add_argument(
        "--timing-only-no-tripwires",
        action="store_true",
        help="record (not enforce) the ratified tripwires; exports no weights",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for key in THREAD_ENV:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
    if os.environ.get("COLAB_CLI_ROOT") and (args.probe_worlds, args.probe_horizon) != (32, 5000):
        raise SystemExit("runtime qualification runs the production 32 x H5000 probe")
    args.out.mkdir(parents=True, exist_ok=False)
    args.receipt_dir = args.receipt_dir or args.out
    receipt = Receipt(
        args.receipt_dir,
        {
            "schema": "t34-claim-a-pipeline/v2",
            "argv": sys.argv[1:],
            "device": args.device,
            "arm": args.arm,
            "status": "RUNNING",
            "timings": {},
            "host": {"sysname": os.uname().sysname, "cpu_count": os.cpu_count()},
            "threads": {key: os.environ.get(key) for key in THREAD_ENV},
            "probe_workers": args.probe_workers,
            "claim": "PIPELINE_QUALIFICATION_ONLY_NO_RESEARCH_OR_PROMOTION_CLAIM",
            "cuda_numerics": cuda_numerics(args.device),
        },
    )
    with stop_signals(receipt):
        try:
            run_pipeline(args, receipt)
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 - every failure must leave a receipt
            receipt.write(
                final=True,
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-8000:],
            )
            print(json.dumps({"status": "FAILED", "error": str(exc)}), flush=True)
            return 2
    status = receipt.report["status"]
    print(json.dumps({"status": status, "timings": receipt.report["timings"]}, sort_keys=True))
    return 0 if status in FINISHED else 2


if __name__ == "__main__":
    raise SystemExit(main())
