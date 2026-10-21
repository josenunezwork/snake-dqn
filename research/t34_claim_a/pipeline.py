"""Claim A end-to-end driver: student labels -> payload -> cache -> fit -> probe.

One entrypoint for the Mac smoke and the (CUDA) L4 qualification.  Phases:

1. ``label``  - optional student-policy label chunk (CPU process fan-out) via
   :mod:`research.t34_claim_a.label`.
2. ``payload`` - hash-bound world index over the new chunk plus any existing
   exports (``--extra-export``), with a deterministic train/held-out split.
3. ``fit``    - the ratified ``research.t34_exit_b8_fit.runner.run`` (mode
   ``fit``) with this package's production ``source_factory``/``probe_factory``.
   It builds the observation cache once, then fits on ``--device`` (cpu|cuda)
   with the frozen loss, held-out decisive agreement, probe and tripwires.

Every phase is timed; ``pipeline.json`` (create-only, sha256 + complete
sidecars) records timings, cache statistics, per-epoch probe calls and the fit
outcome (``FIT_COMPLETED`` with a selected candidate, or ``FIT_STOPPED`` with the
ratified tripwire reason).  Smoke-sized probes are stamped
``SMOKE_PROBE_SHAPE``; they carry no scientific or promotion claim.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_receipt(path: Path, report: dict[str, Any]) -> None:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
    )
    path.with_name(path.name + ".complete").write_text("complete\n")


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
    its candidate is never eligible for evaluation (the receipt says so).
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


def _parse_extra(value: str) -> tuple[Path, int, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--extra-export takes PATH:ROUND_AGE:CHUNK_ID")
    return Path(parts[0]), int(parts[1]), parts[2]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=False)
    carrier = args.carrier.resolve()
    previous = (args.previous or args.carrier).resolve()
    timings: dict[str, float] = {}
    report: dict[str, Any] = {
        "schema": "t34-claim-a-pipeline/v1",
        "argv": sys.argv[1:],
        "device": args.device,
        "arm": args.arm,
        "carrier_sha256": _sha256(carrier),
        "previous_sha256": _sha256(previous),
        "host": {"sysname": os.uname().sysname, "cpu_count": os.cpu_count()},
        "threads": {key: os.environ.get(key) for key in THREAD_ENV},
        "claim": "PIPELINE_QUALIFICATION_ONLY_NO_RESEARCH_OR_PROMOTION_CLAIM",
    }
    started_total = time.monotonic()

    # 1. label -------------------------------------------------------------
    sources: list[tuple[Path, int, str]] = []
    if args.label_worlds:
        from research.t34_claim_a import label

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
        timings["label_seconds"] = time.monotonic() - started
        from research.t34_exit_b8.label import _write_once

        _write_once(out / "label" / "label.json", chunk)
        report["label"] = {
            key: chunk.get(key)
            for key in (
                "complete",
                "worlds",
                "decisions",
                "decisions_per_second",
                "worlds_per_hour",
                "export_bytes",
                "mixes",
                "recipe",
                "student",
                "elapsed_seconds",
            )
        }
        if not chunk.get("complete"):
            report["status"] = "LABEL_INCOMPLETE"
            timings["total_seconds"] = time.monotonic() - started_total
            report["timings"] = timings
            return report
        sources.append((out / "label" / "worlds", 0, f"{args.round}-s{args.shard}-a{args.attempt}"))
    sources.extend(args.extra_export or [])
    if not sources:
        raise SystemExit("nothing to fit: pass --label-worlds and/or --extra-export")

    # 2. payload -----------------------------------------------------------
    from research.t34_claim_a.replay_source import (
        SOURCE_FACTORY,
        build_payload,
        last_source,
        load_payload,
    )

    started = time.monotonic()
    exports = out / "engine-export"
    build_payload(
        sources,
        exports,
        heldout_fraction=args.heldout_fraction,
        heldout_ids=args.heldout_world_id or (),
        heldout_decisive=args.heldout_decisive,
    )
    payload = load_payload(exports)
    timings["payload_seconds"] = time.monotonic() - started
    report["payload"] = {
        "sha256": _sha256(exports / "payload.json"),
        "worlds": {
            split: sum(1 for w in payload["worlds"] if w["split"] == split)
            for split in ("train", "heldout")
        },
        "split_rule": payload["split_rule"],
        "rollout_policy_sha256s": payload["rollout_policy_sha256s"],
    }

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
            "previous_sha256": report["previous_sha256"],
            "probe": {
                "namespace": args.probe_namespace,
                "worlds": args.probe_worlds,
                "horizon": args.probe_horizon,
                **({"mixes": args.probe_mixes.split(",")} if args.probe_mixes else {}),
            },
        },
    }
    cache_dir = args.cache_dir or out / "observation-cache"
    started = time.monotonic()
    tripwire_log: list[dict[str, Any]] = []
    with (
        smoke_probe_shape(args.probe_worlds, args.probe_horizon),
        record_tripwires(args.timing_only_no_tripwires, tripwire_log),
    ):
        result = runner.run(
            exports=exports,
            checkpoint=previous,
            output_dir=out / "fit-output",
            worlds=report["payload"]["worlds"]["train"],
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
    cache_stats: dict[str, Any] | None = None
    manifest = cache_dir / "manifest.json"
    if manifest.is_file():
        stats = json.loads(manifest.read_text())["stats"]
        cache_stats = {
            key: stats.get(key)
            for key in (
                "regen_seconds",
                "rows",
                "training_rows",
                "heldout_rows",
                "bytes",
                "bytes_per_row",
                "regen_seconds_per_row",
                "training",
                "heldout",
            )
        }
        timings["cache_build_seconds"] = float(stats["regen_seconds"])
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
    timings["total_seconds"] = time.monotonic() - started_total
    report.update(
        {
            "previous_student_heldout": baseline,
            "tripwires": {
                "mode": (
                    "RECORDED_NOT_ENFORCED_TIMING_ONLY"
                    if args.timing_only_no_tripwires
                    else "RATIFIED_ENFORCED"
                ),
                "per_epoch": tripwire_log,
            },
        }
    )
    if args.timing_only_no_tripwires:
        report["claim"] = "TIMING_ONLY_TRIPWIRES_RECORDED_CANDIDATE_NOT_ELIGIBLE"
    report.update(
        {
            "binding": binding,
            "timings": timings,
            "source": source.timings.as_dict() if source else None,
            "cache": cache_stats,
            "probe_calls": probe.calls if probe else [],
            "probe_shape": (
                "PRODUCTION_32xH5000"
                if (args.probe_worlds, args.probe_horizon) == (32, 5000)
                else f"SMOKE_PROBE_SHAPE_{args.probe_worlds}xH{args.probe_horizon}"
            ),
            "fit": {
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
            },
            "status": (
                "FIT_STOPPED"
                if not result.get("completed") or result.get("tripwire")
                else (
                    "TIMING_ONLY_FIT_COMPLETED"
                    if args.timing_only_no_tripwires
                    else "FIT_COMPLETED"
                )
            ),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
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
    parser.add_argument("--label-seconds", type=float, default=10200)
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
    # probe
    parser.add_argument("--probe-namespace", default="t34-claim-a/qualification/probe")
    parser.add_argument("--probe-worlds", type=int, default=32)
    parser.add_argument("--probe-horizon", type=int, default=5000)
    parser.add_argument("--probe-mixes")
    parser.add_argument(
        "--timing-only-no-tripwires",
        action="store_true",
        help="record (not enforce) the ratified tripwires; receipt-only, candidate ineligible",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for key in THREAD_ENV:
        os.environ.setdefault(key, "1")
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
    if os.environ.get("COLAB_CLI_ROOT") and (args.probe_worlds, args.probe_horizon) != (32, 5000):
        raise SystemExit("runtime qualification runs the production 32 x H5000 probe")
    report = run_pipeline(args)
    _write_receipt(args.out / "pipeline.json", report)
    print(
        json.dumps({"status": report["status"], "timings": report.get("timings")}, sort_keys=True)
    )
    finished = {"FIT_COMPLETED", "FIT_STOPPED", "TIMING_ONLY_FIT_COMPLETED"}
    return 0 if report["status"] in finished else 2


if __name__ == "__main__":
    raise SystemExit(main())
