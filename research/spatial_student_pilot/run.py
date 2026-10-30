"""Real pilot/ruler/Mac preflight entrypoint; never allocates or dispatches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
from research.spatial_student_pilot.collection_resume import (
    publish_collection,
    restore_collection,
    split_collection,
)
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.fit import fit_student
from research.spatial_student_pilot.forecast import (
    export_projection,
    historical_support_projection,
    late_collection_rate,
    tick_forecast,
)
from research.spatial_student_pilot.model import export_checkpoint
from research.spatial_student_pilot.resume import (
    UnitStore,
    _publish_bytes,
    empty_ledger,
    validate_ledger,
)
from research.spatial_student_pilot.ruler_fixtures import (
    BANK_SHA256,
    FIXTURE_NAME,
    load_fixtures,
    thread_evidence,
    warm_native,
)
from research.spatial_student_pilot.search import (
    _Job,
    _rollouts,
    assess_roots,
    label_roots,
    paired_gameplay,
    prepare_assessment,
    student_actions,
)
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
        self.progress_lock = threading.Lock()
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
        with self.progress_lock:
            self.watchdog.progress(
                str(counters.get("stage", counters.get("phase", "work"))), counters
            )
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
    count = math.ceil(cfg.worlds / cfg.collection_unit_worlds)
    if not all(store.has(f"collect-{i:04d}") for i in range(count)):
        unit = collect_batch(
            cfg,
            range(cfg.worlds),
            b8=b8,
            networks=networks,
            device=device,
            stop=control.stop,
            progress=control.progress,
            resume_state=restore_collection(store, cfg),
            checkpoint=lambda state: publish_collection(store, cfg, state),
        )
        if control.stop():
            raise InterruptedError("collection finished after evidence deadline")
        for i, shard in enumerate(split_collection(cfg, unit)):
            if not store.has(f"collect-{i:04d}"):
                store.publish(f"collect-{i:04d}", shard)
    batches = [store.get(f"collect-{i:04d}") for i in range(count)]
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

    cancel_independent = threading.Event()

    def independent_stop() -> bool:
        return control.stop() or cancel_independent.is_set()

    def independent_assessment() -> None:
        # Thread-local inference mode; no RNG draws, fit tensors or optimizer.
        # The single default CUDA stream provides kernel ordering; useful host
        # work and serialization can overlap fit. Forecast never assumes a gain.
        for index, start in enumerate(range(0, len(assessment_roots), cfg.search_chunk)):
            chunk = assessment_roots[start : start + cfg.search_chunk]
            selected = store.get(f"aselect-{index:04d}")
            if selected is None:
                selected = label_roots(
                    chunk,
                    cfg=cfg,
                    b8=b8,
                    networks=networks,
                    device=device,
                    stop=independent_stop,
                    progress=control.progress,
                )
                store.publish(f"aselect-{index:04d}", selected)
            if not store.has(f"ascore-{index:04d}"):
                prepared = prepare_assessment(
                    chunk,
                    cfg=cfg,
                    b8=b8,
                    networks=networks,
                    device=device,
                    stop=independent_stop,
                    progress=control.progress,
                    labels=selected,
                )
                store.publish(f"ascore-{index:04d}", prepared)

    # UnitStore publication is serialized by its own lock. Preflight uses the
    # same overlap, proving optimizer/RNG state is independent of this lane.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="spatial-independent") as pool:
        future = pool.submit(independent_assessment)
        try:
            student, fit_timing = fit_student(
                cfg,
                dataset["demos"],
                labels,
                device=device,
                store=store,
                stop=lambda: control.stop() or (future.done() and future.exception() is not None),
                progress=control.progress,
            )
        except BaseException:
            cancel_independent.set()
            if future.done():
                future.result()
            raise
        future.result()
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
                prepared=store.get(f"ascore-{index:04d}"),
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
    fixtures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Real integrated stages in a disjoint 900-second development domain.

    Historical natural mature fixtures measure cost only. Fresh early growth
    and frozen historical fractions supply a conditional support estimate.
    """
    started = time.monotonic()
    fixtures = [
        dict(fixtures[i % len(fixtures)], fixture_replica=i // len(fixtures))
        for i in range(cfg.world_batch_size)
    ]
    phases = {"native_startup": warm_native(control.progress) if device.startswith("cuda") else {}}
    if device.startswith("cuda"):
        from research.spatial_student_pilot.native_cache import export_cache

        phases["native_cache_export"] = export_cache(store.output)
        from research.spatial_student_pilot.parity import mapped_mature_proof

        phases["mapped_parity"] = mapped_mature_proof(
            cfg, fixtures, b8=b8, networks=networks, device=device
        )
        publish_json(store.output / "mapped-parity.json", phases["mapped_parity"])
        if phases["mapped_parity"]["exact"] is not True:
            raise ValueError("E128/E1024 mature mapped exact-digest proof failed")
    early = collect_batch(
        cfg,
        range(cfg.world_batch_size),
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
        ruler_seconds=90,
        ruler_ticks=80,
    )
    store.publish("early-0000", early)
    checkpoint_probe = {}

    def measure_checkpoint(state: dict[str, Any]) -> None:
        from research.spatial_student_pilot.forecast import _sizes, checkpoint_components

        class ProbeStore:
            def has(self, name: str) -> bool:
                return False

            def publish(self, name: str, payload: Any) -> None:
                checkpoint_probe.setdefault("sizes", {})[name] = _sizes(payload)
                if name.startswith("cstate-"):
                    checkpoint_probe.setdefault("components", []).append(
                        checkpoint_components(payload)
                    )

        mark = time.monotonic()
        publish_collection(ProbeStore(), cfg, state)
        scale = max(1.0, cfg.demos / max(1, sum(len(v) for v in state["demos"].values())))
        checkpoint_probe.update(
            capture_seconds=state["capture_seconds"],
            serialization_seconds=time.monotonic() - mark,
            record_growth_factor=scale,
        )

    batch = collect_batch(
        cfg,
        range(cfg.world_batch_size),
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
        ruler_ticks=80,
        ruler_fixtures=fixtures,
        checkpoint=measure_checkpoint,
    )
    store.publish("mature-0000", batch)
    collection_rate = late_collection_rate(batch)
    support = historical_support_projection(early)
    phases["early_growth"] = {
        "seconds": early["elapsed_seconds"],
        "ticks": len(early["tick_samples"]),
        "samples": early["tick_samples"],
    }
    phases["collection_checkpoint_probe"] = checkpoint_probe
    phases["collection"] = {
        "seconds": batch["elapsed_seconds"],
        "rows": batch["world_ticks"],
        "late_window": collection_rate,
        "support_projection": support,
    }
    train = [r for r in batch["roots"] if r["partition"] == "train"]
    late = [r for r in train if r["hero_length"] >= 600]
    mature_mixes = {r["mix"] for r in late}
    # Round-robin mixes, at least22 roots AND enough legal jobs for a full128 slab.
    pools = {mix: [r for r in late if r["mix"] == mix] for mix in ("scripted", "frozen", "mixed")}
    roots = []
    for index in range(max(map(len, pools.values()), default=0)):
        for pool in pools.values():
            if index < len(pool):
                roots.append(pool[index])
        if (
            len(roots) >= 22
            and sum(int(r["mask"].sum()) * 2 for r in roots) >= cfg.branch_batch_size
        ):
            break
    if not roots or not batch["demos"]:
        raise RuntimeError("INCOMPLETE_RULER missing mature fixture roots or demonstrations")
    branch_timings = []

    def ruler_progress(event: dict[str, Any]) -> None:
        if event.get("phase") == "branch_timing":
            branch_timings.append(event)
        control.progress(event)

    mark = time.monotonic()
    labels = label_roots(
        roots,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=ruler_progress,
    )
    store.publish("search-0000", labels)
    phases["search"] = {
        "seconds": time.monotonic() - mark,
        "roots": len(roots),
        "slabs": list(branch_timings),
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
    # Assessment mixed scoring slabs execute both B8 and the student forward.
    closed_jobs = [
        _Job(i % len(roots), int(roots[i % len(roots)]["b8_action"]), 4 + i % 2, True)
        for i in range(cfg.branch_batch_size)
    ]
    _rollouts(
        roots,
        closed_jobs,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        student=student,
        stop=control.stop,
        progress=ruler_progress,
    )
    phases["closed_loop"] = {"slabs": [r for r in branch_timings if r["closed_loop"]]}
    # Serialize real assessment records, including the action-count-bearing row.
    assessment_rows = assess_roots(
        [dict(r, partition="assessment") for r in roots[:2]],
        student,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
    )
    store.publish("assess-0000", assessment_rows)
    # Measure E128 student extraction/forward separately on mature states. This
    # is added to the collection floor even though that floor already pays B8.
    from research.sandbox_a_resident.ct_snapshots import (
        CompactWorldSnapshot,
        restore_world_snapshots_into,
    )
    from research.spatial_student_pilot.batching import BatchedOpponents
    from research.spatial_student_pilot.search import _new_sim, _sync

    forward_fixtures = fixtures[: cfg.gameplay_worlds]
    sim = _new_sim(cfg, len(forward_fixtures), device)
    opp = BatchedOpponents(
        sim,
        [training_rosters(cfg.worlds)[r["roster_index"]] for r in forward_fixtures],
        b8_network=b8,
        checkpoint_networks=networks,
    )
    restore_world_snapshots_into(
        [CompactWorldSnapshot.from_state_dict(r["snapshot"]) for r in forward_fixtures],
        sim,
        opp,
    )
    forward_times = []
    for index in range(23):
        _sync(device)
        before = time.monotonic()
        student_actions(sim, student)
        _sync(device)
        if index >= 3:
            forward_times.append(time.monotonic() - before)
    phases["student_forward"] = {"slab_size": len(forward_fixtures), "seconds": forward_times}
    del sim, opp
    mark = time.monotonic()
    worlds = _worlds(cfg)[: cfg.world_batch_size]
    # Keep E128 even after early deaths; exclude setup and cap each arm to45s/300ticks.
    arm_seconds = max(0.001, min(45.0, (control.deadline - time.monotonic() - 30) / 2))
    rows = paired_gameplay(
        worlds,
        student,
        cfg=cfg,
        b8=b8,
        networks=networks,
        device=device,
        stop=control.stop,
        progress=control.progress,
        ruler_ticks=300,
        ruler_arm_seconds=arm_seconds,
    )
    store.publish("gameplay-0000", rows)
    phases["gameplay"] = {
        "seconds": time.monotonic() - mark,
        "logical_world_steps": 2 * len(worlds) * cfg.horizon,
        "executed_world_steps": {
            arm: rows[0][arm + "_executed_world_steps"] for arm in ("b8", "student")
        },
        "ticks": {arm: rows[0][arm + "_ticks"] for arm in ("b8", "student")},
        "slab_size": len(worlds),
        "arm_seconds": {arm: rows[0][arm + "_loop_seconds"] for arm in ("b8", "student")},
        "setup_inclusive_arm_seconds": {
            arm: rows[0][arm + "_wall_seconds"] for arm in ("b8", "student")
        },
        "mature_tick_floors": {
            "b8": collection_rate["seconds_per_tick"],
            "student": collection_rate["seconds_per_tick"] + max(forward_times),
        },
    }
    mark = time.monotonic()
    actor = store.output / "ruler-actor.pth"
    export_checkpoint(student, actor, metadata={"namespace": cfg.namespace, "ruler_only": True})
    seal_file(actor)
    phases["export"] = {"seconds": time.monotonic() - mark, "bytes": actor.stat().st_size}
    full_slabs = [
        r for r in branch_timings if r["slab_size"] == cfg.branch_batch_size and r["ticks"] == 80
    ]
    b8_rates = [r["loop_seconds"] / r["ticks"] for r in full_slabs if not r["closed_loop"]]
    student_rates = [r["loop_seconds"] / r["ticks"] for r in full_slabs if r["closed_loop"]]
    gameplay_qualified = len(worlds) == cfg.gameplay_worlds and all(
        phases["gameplay"]["ticks"][a] >= 20 for a in ("b8", "student")
    )
    rates_qualified = (
        len(roots) >= 22
        and phases.get("mapped_parity", {}).get("exact", False)
        and bool(b8_rates)
        and bool(student_rates)
        and gameplay_qualified
        and collection_rate["qualified"]
    )
    components = tick_forecast(
        collection_tick=collection_rate["seconds_per_tick"],
        b8_branch_tick=max(b8_rates, default=900.0),
        student_branch_tick=max(student_rates, default=900.0),
        gameplay_ticks={
            a: phases["gameplay"]["arm_seconds"][a] / max(1, phases["gameplay"]["ticks"][a])
            for a in ("b8", "student")
        },
        fit_update=fit_timing["elapsed_seconds"] / (cfg.bc_updates + cfg.improvement_updates),
        student_forward_tick=max(forward_times),
        cfg=PilotConfig(),
    )
    # Nine complete collection trajectory boundaries, conservatively scaling
    # the actual mature capture+torch.save+gzip measurement to a full reservoir.
    components["collection_checkpoints"] = (
        9
        * (
            checkpoint_probe.get("capture_seconds", 900)
            + checkpoint_probe.get("serialization_seconds", 900)
        )
        * checkpoint_probe.get("record_growth_factor", 1)
        * 1.25
    )
    sizes = export_projection(
        batch,
        labels,
        [store.get(n) for n in store.list_ids("fit-")],
        actor.stat().st_size,
        assessment_rows=assessment_rows,
        gameplay_rows=rows,
        checkpoint_samples=checkpoint_probe.get("components", []),
    )
    qualified = (
        len(mature_mixes) == 3 and rates_qualified and support["qualified"] and sizes["qualified"]
    )
    forecast = sum(components.values())
    setup_allowance = max(3600, phases["native_startup"].get("seconds", 0))
    return {
        "status": "RULER_COMPLETE",
        "scientific_result": False,
        "development_fixtures_only": True,
        "fixture_bank_sha256": BANK_SHA256,
        "phases": phases,
        "natural_mature_support": False,
        "historical_natural_mature_fixtures": bool(late),
        "mature_mixes": sorted(mature_mixes),
        "forecast_qualified": qualified,
        "rate_qualified": rates_qualified,
        "support_projection": support,
        "export_projection": sizes,
        "forecast_components_seconds": components,
        "forecast_compute_seconds": forecast,
        "forecast_setup_export_seconds": setup_allowance,
        "forecast_with_setup_export_seconds": forecast + setup_allowance,
        "within_six_hours": qualified and forecast + setup_allowance <= 21600,
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
    threads = thread_evidence(enforce=args.device == "cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        from research.spatial_student_pilot.native_cache import preload

        preload(args.input_dir)
    original = PilotConfig.from_dict(json.loads(args.config.read_text()))
    ledger = json.loads(args.ledger.read_text())
    validate_ledger(ledger, original)
    cfg = original if args.mode == "pilot" else original.development(args.mode)
    if args.mode != "pilot":
        # A ruler can transport a completion ledger to qualify its exact input
        # inventory; the short-canary exemption never decodes/imports its units.
        if args.mode == "ruler":
            cfg = replace(
                cfg, bc_updates=25, improvement_updates=25, checkpoint_every=25, branch_horizon=80
            )
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
        kwargs = dict(b8=b8, networks=networks, device=args.device, control=control)
        if args.mode == "ruler":
            result = ruler(
                cfg,
                store,
                **kwargs,
                fixtures=load_fixtures(args.input_dir / FIXTURE_NAME, cfg.mode),
            )
        else:
            result = pipeline(cfg, store, **kwargs)
        if control.stop():
            raise InterruptedError("deadline/signal reached before complete export")
        publish_json(
            args.output / "summary.json",
            {
                "schema": "spatial-run/v1",
                "threads": threads,
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
                "threads": threads,
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
