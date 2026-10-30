"""Real pilot/ruler/Mac preflight entrypoint; never allocates or dispatches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from research.sandbox_a_resident.ct_run import B8_S3_SHA256, _load_vector61
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    training_rosters,
)
from research.spatial_student_pilot.collect import collect_batch, select_dataset
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.fit import fit_student
from research.spatial_student_pilot.model import export_checkpoint
from research.spatial_student_pilot.resume import (
    UnitStore,
    _publish_bytes,
    empty_ledger,
    validate_ledger,
)
from research.spatial_student_pilot.search import assess_roots, label_roots, paired_gameplay
from src.model.ego2s_network import load_ego2s_checkpoint


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal_file(path: Path) -> None:
    """Seal an already immutable actor/summary, completion last."""
    _publish_bytes(Path(str(path) + ".sha256"), (digest(path) + "  " + path.name + "\n").encode())
    _publish_bytes(Path(str(path) + ".complete"), b"complete\n")


def publish_json(path: Path, payload: dict[str, Any]) -> None:
    _publish_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    seal_file(path)


class Control:
    """Signal/deadline fence and completed-work-only watchdog."""

    def __init__(self, output: Path, seconds: float) -> None:
        self.output = output
        self.store: UnitStore | None = None
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.signal = False
        self.last_print = 0.0
        self.watchdog = StallWatchdog(output / "diagnostics", 900)
        self.watchdog.start("startup")
        self.previous = signal.signal(signal.SIGUSR1, self.on_signal)

    def on_signal(self, _signum: int, _frame: Any) -> None:
        self.signal = True
        marker = self.output / "sigusr1-marker.json"
        if not marker.exists():
            publish_json(
                marker,
                {
                    "schema": "spatial-stop/v1",
                    "partial_unit_complete": False,
                    "latest_verified_units": self.store.list_ids() if self.store else [],
                },
            )

    def stop(self) -> bool:
        return self.signal or time.monotonic() >= self.deadline

    def progress(self, counters: dict[str, Any]) -> None:
        self.watchdog.progress(str(counters.get("stage", counters.get("phase", "work"))), counters)
        now = time.monotonic()
        if now - self.last_print >= 30:
            print(json.dumps({"event": "progress", **counters}, sort_keys=True), flush=True)
            self.last_print = now

    def close(self) -> None:
        self.watchdog.close()
        signal.signal(signal.SIGUSR1, self.previous)


def _worlds(cfg: PilotConfig) -> list[dict[str, Any]]:
    return [
        {
            "world_id": i,
            "roster_index": i,
            "mix": roster.mix,
            "seed": cfg.world_seed(i, "gameplay"),
            "opp_seed": cfg.opp_seed(i, "gameplay"),
        }
        for i, roster in enumerate(training_rosters(cfg.gameplay_worlds))
    ]


def pipeline(
    cfg: PilotConfig,
    store: UnitStore,
    *,
    b8: Any,
    networks: dict[str, Any],
    device: str,
    control: Control,
) -> dict[str, Any]:
    batches = []
    for index, start in enumerate(range(0, cfg.worlds, cfg.world_batch_size)):
        name = f"collect-{index:04d}"
        unit = store.get(name)
        if unit is None:
            unit = collect_batch(
                cfg,
                range(start, min(cfg.worlds, start + cfg.world_batch_size)),
                b8=b8,
                networks=networks,
                device=device,
                stop=control.stop,
                progress=control.progress,
            )
            if control.stop():
                raise InterruptedError("collection finished after evidence deadline")
            store.publish(name, unit)
        batches.append(unit)
    dataset = select_dataset(cfg, batches)
    train_roots = [r for r in dataset["roots"] if r["partition"] == "train"]
    assessment_roots = [r for r in dataset["roots"] if r["partition"] == "assessment"]
    labels = []
    for index, start in enumerate(range(0, len(train_roots), cfg.search_chunk)):
        name = f"search-{index:04d}"
        unit = store.get(name)
        if unit is None:
            unit = label_roots(
                train_roots[start : start + cfg.search_chunk],
                cfg=cfg,
                b8=b8,
                networks=networks,
                device=device,
                stop=control.stop,
                progress=control.progress,
            )
            if control.stop():
                raise InterruptedError("search finished after evidence deadline")
            store.publish(name, unit)
        labels.extend(unit)
    student, fit_timing = fit_student(
        cfg,
        dataset["demos"],
        labels,
        device=device,
        store=store,
        stop=control.stop,
        progress=control.progress,
    )
    if control.stop():
        raise InterruptedError("fit finished after evidence deadline")
    actor = store.output / "final-actor.pth"
    export_checkpoint(
        student,
        actor,
        metadata={
            "config_sha256": cfg.sha256,
            "seed": cfg.seed,
            "updates": cfg.bc_updates + cfg.improvement_updates,
            "namespace": cfg.namespace,
            "source_commit": os.environ.get("SPATIAL_SOURCE_COMMIT", "local"),
        },
    )
    seal_file(actor)
    for index, start in enumerate(range(0, len(assessment_roots), cfg.search_chunk)):
        name = f"assess-{index:04d}"
        if not store.has(name):
            unit = assess_roots(
                assessment_roots[start : start + cfg.search_chunk],
                student,
                cfg=cfg,
                b8=b8,
                networks=networks,
                device=device,
                stop=control.stop,
                progress=control.progress,
            )
            if control.stop():
                raise InterruptedError("assessment finished after evidence deadline")
            store.publish(name, unit)
    worlds = _worlds(cfg)
    for index, start in enumerate(range(0, len(worlds), cfg.world_batch_size)):
        name = f"gameplay-{index:04d}"
        if not store.has(name):
            unit = paired_gameplay(
                worlds[start : start + cfg.world_batch_size],
                student,
                cfg=cfg,
                b8=b8,
                networks=networks,
                device=device,
                stop=control.stop,
                progress=control.progress,
            )
            if control.stop():
                raise InterruptedError("gameplay finished after evidence deadline")
            store.publish(name, unit)
    return {
        "status": "ENDPOINT_EXPORTED_NO_REMOTE_LOOK",
        "decisions": dataset["decisions"],
        "units": store.list_ids(),
        "candidate_sha256": digest(actor),
        "fit_timing": fit_timing,
        "scientific_result": False,
        "preflight_only": cfg.mode == "preflight",
    }


def ruler(
    cfg: PilotConfig,
    store: UnitStore,
    *,
    b8: Any,
    networks: dict[str, Any],
    device: str,
    control: Control,
) -> dict[str, Any]:
    """Real integrated stages in a disjoint 900-second development domain.

    Stop collection after at most 420 seconds; natural mature support is a
    qualification condition, never manufactured to improve the forecast.
    """
    started = time.monotonic()
    phases = {}
    batch = collect_batch(
        cfg,
        range(cfg.world_batch_size),
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
        ruler_seconds=420,
    )
    store.publish("collect-0000", batch)
    phases["collection"] = {"seconds": batch["elapsed_seconds"], "rows": batch["world_ticks"]}
    train = [r for r in batch["roots"] if r["partition"] == "train"]
    late = [r for r in train if r["hero_length"] >= 600]
    mature_mixes = {r["mix"] for r in late}
    roots = [
        r
        for mix in ("scripted", "frozen", "mixed")
        for r in [row for row in (late or train) if row["mix"] == mix][:2]
    ]
    if not roots or not batch["demos"]:
        raise RuntimeError("INCOMPLETE_RULER no natural roots/demonstrations")
    mark = time.monotonic()
    labels = label_roots(
        roots,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
    )
    store.publish("search-0000", labels)
    phases["search"] = {
        "seconds": time.monotonic() - mark,
        "logical_steps": sum(int(r["mask"].sum()) * 2 * cfg.branch_horizon for r in roots),
    }
    student, fit_timing = fit_student(
        cfg,
        batch["demos"],
        labels,
        device=device,
        store=store,
        stop=control.stop,
        progress=control.progress,
    )
    phases["fit"] = fit_timing
    mark = time.monotonic()
    worlds = _worlds(cfg)[:3]
    rows = paired_gameplay(
        worlds,
        student,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
    )
    store.publish("gameplay-0000", rows)
    phases["gameplay"] = {
        "seconds": time.monotonic() - mark,
        "logical_world_steps": 2 * len(worlds) * cfg.horizon,
    }
    mark = time.monotonic()
    actor = store.output / "ruler-actor.pth"
    export_checkpoint(student, actor, metadata={"namespace": cfg.namespace, "ruler_only": True})
    seal_file(actor)
    phases["export"] = {"seconds": time.monotonic() - mark, "bytes": actor.stat().st_size}
    # Match actual represented steps, not 10x extrapolation from another actor.
    collection_seconds = (
        1024 * 5000 * phases["collection"]["seconds"] / phases["collection"]["rows"]
    )
    search_seconds = (
        (1024 * 6 * 2 * 300) * phases["search"]["seconds"] / phases["search"]["logical_steps"]
    )
    fit_seconds = 8000 * fit_timing["elapsed_seconds"] / (cfg.bc_updates + cfg.improvement_updates)
    assessment_seconds = (
        (384 * (6 * 2 + 4 * 2) * 300)
        * phases["search"]["seconds"]
        / phases["search"]["logical_steps"]
    )
    gameplay_seconds = 192 / 3 * phases["gameplay"]["seconds"]
    forecast = sum(
        (collection_seconds, search_seconds, fit_seconds, assessment_seconds, gameplay_seconds)
    )
    return {
        "status": "RULER_COMPLETE",
        "scientific_result": False,
        "phases": phases,
        "natural_mature_support": bool(late),
        "mature_mixes": sorted(mature_mixes),
        "forecast_qualified": len(mature_mixes) == 3,
        "forecast_compute_seconds": forecast,
        "forecast_with_setup_export_seconds": forecast + 3600,
        "within_six_hours": len(mature_mixes) == 3 and forecast + 3600 <= 21600,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "ruler", "preflight"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = PilotConfig.from_dict(json.loads(args.config.read_text()))
    ledger = json.loads(args.ledger.read_text())
    validate_ledger(ledger, original)
    cfg = original if args.mode == "pilot" else original.development(args.mode)
    if args.mode != "pilot":
        # A ruler can transport a completion ledger to qualify its exact input
        # inventory; the short-canary exemption never decodes/imports its units.
        if args.mode == "ruler":
            cfg = replace(cfg, bc_updates=25, improvement_updates=25, checkpoint_every=25)
        ledger = empty_ledger(cfg)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.mode == "pilot" and args.device != "cuda":
        raise ValueError("scientific pilot requires registered CUDA hardware")
    if args.output.exists():
        raise FileExistsError("every attempt/preflight requires a fresh output")
    args.output.mkdir(parents=True)
    store = UnitStore(args.output, cfg, ledger, args.input_dir)
    # Runtime admission hashes every immutable comparator, even on tiny preflight.
    checks = {
        "b8-s3.pth": B8_S3_SHA256,
        "frp3-s12.pth": FRP3_S12_SHA256,
        "champion.pth": CHAMPION_SHA256,
    }
    for name, expected in checks.items():
        if digest(args.input_dir / name) != expected:
            raise ValueError(f"checkpoint digest mismatch: {name}")
    b8 = load_ego2s_checkpoint(str(args.input_dir / "b8-s3.pth"), device=args.device)
    networks = {
        checks[name]: _load_vector61(args.input_dir / name, torch.device(args.device))
        for name in ("frp3-s12.pth", "champion.pth")
    }
    # Never authorize scientific data after the cumulative allocated evidence deadline.
    allowance = (900 if args.mode == "ruler" else 21600) - ledger["consumed_seconds"]
    if args.mode == "pilot":
        root = Path(os.environ["COLAB_CLI_ROOT"])
        runtime_spec = json.loads((root / "spec.json").read_text())
        runtime_state = json.loads((root / "state.json").read_text())
        # The supervisor freezes remaining whole-session seconds before VM
        # launch. Subtract its loss from the 720min envelope, plus VM age.
        # This conservatively includes its fixed 70s host/release margins.
        already_allocated = max(0, 43200 - runtime_spec["timeout_seconds"])
        already_allocated += max(0, time.time() - runtime_state["started_at"])
        allowance -= already_allocated
    control = Control(args.output, max(0, allowance))
    control.store = store
    try:
        result = (ruler if args.mode == "ruler" else pipeline)(
            cfg, store, b8=b8, networks=networks, device=args.device, control=control
        )
        if control.stop():
            raise InterruptedError("deadline/signal reached before complete export")
        publish_json(
            args.output / "summary.json",
            {
                "schema": "spatial-run/v1",
                **result,
                "namespace": cfg.namespace,
                "config_sha256": cfg.sha256,
                "elapsed_seconds": time.monotonic() - control.started,
            },
        )
        print("SPATIAL_ENTRYPOINT_COMPLETE " + args.mode, flush=True)
        return 0
    except Exception as error:
        publish_json(
            args.output / "summary.json",
            {
                "schema": "spatial-run/v1",
                "status": "INCOMPLETE",
                "reason": str(error),
                "units": store.list_ids(),
                "scientific_result": False,
                "elapsed_seconds": time.monotonic() - control.started,
            },
        )
        return 138 if control.signal else 2
    finally:
        control.close()


if __name__ == "__main__":
    raise SystemExit(main())
