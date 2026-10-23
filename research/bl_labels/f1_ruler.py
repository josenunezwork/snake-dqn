"""F1 placement ruler: a frozen 64-root F0 labeller workload, without outcome exports."""

from __future__ import annotations

import argparse
import copyreg
import hashlib
import json
import lzma
import math
import multiprocessing as mp
import os
import pickle
import platform
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .exports import _verify, publish_json
from .f1_units import LABELS, load_ledger

SCHEMA = "bl-labels-f1-root-bank/v1"
RECEIPT_SCHEMA = "bl-labels-f1-ruler-receipt/v1"
ROOTS = 64


class WorkerStall(RuntimeError):
    """The frozen no-completed-root watchdog fired."""


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip() or "UNKNOWN"
    return (platform.processor() or "UNKNOWN") if platform.system() == "Darwin" else "UNKNOWN"


def build_bank(sessions: list[Path], out: Path) -> dict[str, Any]:
    """Select metadata by root-ID hash; never inspect F0 label/outcome rows."""
    candidates, sources = [], {}
    for path in sorted(sessions):
        digest = _verify(path)
        value = json.loads(path.read_text())
        if value.get("schema") != "bl-labels-f0-session/v1" or value.get("status") != "COMPLETE":
            raise ValueError("F1 requires verified complete F0 sessions")
        sources[path.name] = {
            "name": path.name,
            "sha256": digest,
            "namespace": value["namespace"],
            "allocation": value["allocation"],
            "tape": value["audit_tape"],
        }
        for root in value["audit_roots"]:
            if root.get("initial_allocation") != value["allocation"]:
                raise ValueError("F1 audit root belongs to a foreign parent allocation")
            end = root.get("tape_end")
            if type(end) is not int or not 0 < end <= len(value["audit_tape"]):
                raise ValueError("F1 audit root has invalid tape_end")
            candidates.append(
                {**root, "source_session": path.name, "source_namespace": value["namespace"]}
            )
    if len({r["root_id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate F1 root ID")
    selected = sorted(candidates, key=lambda r: hashlib.sha256(r["root_id"].encode()).hexdigest())[
        :ROOTS
    ]
    if len(selected) != ROOTS:
        raise ValueError("F1 requires at least 64 frozen roots")
    selected_sources = sorted({r["source_session"] for r in selected})
    namespaces = {sources[name]["namespace"] for name in selected_sources}
    if len(namespaces) != 1:
        raise ValueError("F1 roots must come from the same frozen canary namespace")
    value = {
        "schema": SCHEMA,
        "outcome_blind": True,
        "root_count": ROOTS,
        "selection": "lowest sha256(root_id) over complete Canary A audit-root metadata",
        "source_sessions": [sources[name] for name in selected_sources],
        "roots": selected,
    }
    publish_json(out, value)
    return value


def bank_bytes(path: Path) -> bytes:
    _verify(path)
    payload = path.read_bytes()
    return lzma.decompress(payload) if path.suffix == ".xz" else payload


def bank_digest(path: Path) -> str:
    return hashlib.sha256(bank_bytes(path)).hexdigest()


def load_bank(path: Path) -> dict[str, Any]:
    value = json.loads(bank_bytes(path))
    roots = value.get("roots")
    if (
        value.get("schema") != SCHEMA
        or value.get("root_count") != ROOTS
        or not isinstance(roots, list)
        or len(roots) != ROOTS
    ):
        raise ValueError("foreign or partial F1 root bank")
    if any(not re.fullmatch("[0-9a-f]{64}", str(r.get("root_id"))) for r in roots):
        raise ValueError("F1 root IDs must be content-addressed hashes")
    if len({r["root_id"] for r in roots}) != ROOTS:
        raise ValueError("duplicate F1 bank roots")
    if roots != sorted(roots, key=lambda r: hashlib.sha256(r["root_id"].encode()).hexdigest()):
        raise ValueError("F1 root selection order differs")
    return value


def _deadline_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("INCOMPLETE F1 whole-session collection deadline")


def _prepare_source(
    source: dict, roots: list[dict], b8: Path, frp3: Path, cache: Path, deadline: float
) -> dict[str, Path]:
    """Replay a parent once, preserving F0 carry and nearest-20 snapshots.

    The original tape is an integrity oracle. No archived branch outcome or
    selected-action label is read. Natural post-step mass supplies bare truth.
    """
    from . import factory

    checkpoint = hashlib.sha256(b8.read_bytes()).hexdigest()
    driver = factory.make_driver(source["allocation"], b8, frp3)
    restore = type(driver).restore
    targets = {int(r["root_decision"]): r for r in roots}
    entries, anchors, captured, events = {}, {}, {}, {}
    next_progress = time.monotonic() + 110
    last_needed = max(int(r["tape_end"]) for r in roots)
    for row in source["tape"][:last_needed]:
        _deadline_check(deadline)
        driver.prepare()
        decision = driver.decision
        if decision != row["decision"] or not bool(driver.sim.alive[0, 0]):
            raise ValueError("F1 source tape decision/death boundary differs")
        legal, bare = driver.legal_and_bare()
        # Preserve recorded hero actions on float ties; never override opponents
        # or reuse their tape after a counterfactual divergence.
        action = int(row["actions"][0])
        if bare != action:
            raise ValueError("F1 source bare policy action differs on this platform")
        entry = factory._TapeEntry(
            decision,
            driver.life,
            int(driver.sim.frame[0]),
            int(driver.sim.length[0, 0]),
            bare,
            tuple(float(x) for x in driver.last_root_q),
            tuple(legal),
            factory._digest_driver(driver, checkpoint) if decision in targets else "",
            factory._raw_observation(driver) if decision in targets else {},
        )
        entries[decision] = entry
        if decision % factory.SNAPSHOT_STRIDE == 0 or not anchors:
            anchors[decision] = driver.snapshot()
        if decision in targets:
            root = targets[decision]
            # Source full hashes include platform-local policy/RNG carry.
            # Recorded simulator/action history is checked on every transition;
            # local snapshot restoration must equal this local live full hash.
            anchor = max(d for d in anchors if d <= decision)
            captured[decision] = (
                entry,
                anchor,
                anchors[anchor],
                {d: entries[d] for d in range(anchor, decision + 1)},
            )
        event = driver.step()
        if (
            event["actions"] != row["actions"]
            or factory._env_digest(driver.sim) != row["env_digest"]
        ):
            raise ValueError("F1 natural replay action/environment digest mismatch")
        events[decision] = event
        entries[decision] = replace(
            entry, actions=tuple(event["actions"]), post_length=int(event["hero_length"])
        )
        # Discovery snapshots are bounded like the source factory. Captured
        # target prefixes own only their nearest anchor, at most 19 replay steps.
        entries = {d: e for d, e in entries.items() if d >= decision - 420}
        anchors = {d: p for d, p in anchors.items() if d >= decision - 440}
        if time.monotonic() >= next_progress:
            print(
                f'F1_PREPARE parent={source["name"]} done={decision + 1} total={last_needed}',
                flush=True,
            )
            next_progress = time.monotonic() + 110
    result = {}
    for root in roots:
        target, end = int(root["root_decision"]), int(root["tape_end"])
        if target not in captured or end - 1 not in events:
            raise ValueError("F1 source is missing a selected root continuation")
        # Factory audit roots are doom roots; tape_end is the exclusive fatal
        # endpoint captured by remember_audit, not an inferred short suffix.
        fatal = events[end - 1]["death_cause"] != "none"
        if not fatal or not 0 < end - target <= factory.HORIZON:
            raise ValueError("F1 audit root is not an exact natural doom root")
        entry, anchor, payload, prefix = captured[target]
        post = [int(events[d]["hero_length"]) for d in range(target, end)]
        natural = {
            "survives": False,
            "frames_executed": end - target,
            "death_decision": end - 1,
            "final_length": post[-1],
            "mass_integral": sum(post) / factory.HORIZON,
        }
        cached = {
            "root": root,
            "entry": entry,
            "anchor": anchor,
            "snapshot": payload,
            "prefix": prefix,
            "natural": natural,
            "restore": restore,
            "checkpoint": checkpoint,
            "death_endpoint": end,
        }
        path = cache / (root["root_id"] + ".pickle")
        path.write_bytes(pickle.dumps(cached, protocol=pickle.HIGHEST_PROTOCOL))
        result[root["root_id"]] = path
    return result


def _prepare_source_job(job: tuple) -> dict[str, Path]:
    return _prepare_source(*job)


def _label_root(job: tuple[str, float]) -> dict[str, Any]:
    """Execute the source F0 doom-root labeller, exporting timing/integrity only."""
    from research.bl_served_search import ceiling, exact_stop

    from . import factory

    path, deadline = job
    value = pickle.loads(Path(path).read_bytes())
    root, entry, natural = value["root"], value["entry"], value["natural"]
    started = time.monotonic()

    def pulse() -> None:
        _deadline_check(deadline)

    pulse()
    restored = factory._restore_root(
        {value["anchor"]: value["snapshot"]},
        value["prefix"],
        entry,
        value["checkpoint"],
        restore=value["restore"],
    )
    snapshot = restored.snapshot()
    namespace = root["source_namespace"] + "/" + root["initial_allocation"]["allocation_id"]
    selected = (
        lambda kind: int(factory._selection_hash(namespace, kind, entry.life, entry.decision), 16)
        % 10
        == 0
    )
    complete_audit = selected("complete")
    complete = entry.decision >= value["death_endpoint"] - factory.LAST_CONTIGUOUS or complete_audit
    actions = sorted(set((*entry.legal, entry.action)))

    def evaluate(action: int) -> dict:
        pulse()
        if action == entry.action:
            return natural
        return factory._branch(
            snapshot, action, factory.HORIZON, restore=value["restore"], pulse=pulse
        )

    if complete:
        outcomes = {a: evaluate(a) for a in actions}
        chosen = ceiling._choose(outcomes, entry.action, np.asarray(entry.q))
    else:
        stop = exact_stop.run(actions, entry.action, np.asarray(entry.q), evaluate, ceiling._choose)
        outcomes = {a: v for a, v in stop["outcomes"].items() if "survives" in v}
        chosen = stop["chosen"]
    row = factory._row(
        entry,
        outcomes,
        death_decision=value["death_endpoint"] - 1,
        stream=root["initial_allocation"]["stream"],
        platform_tag=factory._platform_tag(),
        forced_complete=complete,
        audit_selected=complete_audit,
    )
    bare_audit = selected("bare-audit")
    if bare_audit:
        actual = factory._branch(
            snapshot, entry.action, factory.HORIZON, restore=value["restore"], pulse=pulse
        )
        if any(
            actual[k] != natural[k]
            for k in ("survives", "frames_executed", "death_decision", "final_length")
        ):
            raise ValueError("INVALID F1 bare audit mismatch")
    if row["decisive"] and selected("extended"):
        factory._branch(
            snapshot, int(chosen), factory.EXTENDED_HORIZON, restore=value["restore"], pulse=pulse
        )
    return {
        "root_id": root["root_id"],
        "elapsed_seconds": time.monotonic() - started,
        "digest": entry.digest,
        "source_digest": root["root_digest"],
        "source_digest_matches_local": entry.digest == root["root_digest"],
        "digest_checks": 1,
        "bare_audits": int(bare_audit),
        "bare_audit_mismatches": 0,
    }


def session_deadline(session_seconds: float, finalize_seconds: float) -> dict[str, Any]:
    root = Path(os.environ["COLAB_CLI_ROOT"])
    spec, state = (json.loads((root / name).read_text()) for name in ("spec.json", "state.json"))
    elapsed = time.time() - float(state["started_at"])
    timeout = float(spec["timeout_seconds"])
    if (
        spec["job_id"] != state["job_id"]
        or spec["generation"] != state["generation"]
        or Path(spec["root"]).resolve() != root.resolve()
        or float(spec["env"]["F1_SESSION_SECONDS"]) != session_seconds
        or float(spec["env"]["F1_FINALIZE_SECONDS"]) != finalize_seconds
        or not all(math.isfinite(x) for x in (elapsed, timeout, session_seconds, finalize_seconds))
        or not 0 <= elapsed < timeout <= session_seconds
        or not 0 < finalize_seconds < timeout - elapsed
    ):
        raise ValueError("F1 qualified whole-session budget identity/bounds differ")
    remaining = timeout - elapsed - finalize_seconds
    return {
        "schema": "bl-labels-f1-session-budget/v1",
        "session_seconds": session_seconds,
        "qualified_window_seconds": timeout,
        "measured_setup_seconds": elapsed,
        "finalize_seconds": finalize_seconds,
        "runtime_seconds": remaining,
        "collection_deadline_monotonic": time.monotonic() + remaining,
    }


def run(
    bank: dict,
    bank_sha256: str,
    ledger: Path,
    b8: Path,
    frp3: Path,
    platform_label: str,
    output: Path,
    deadline: float,
    *,
    probe: bool = False,
    local_workers: int | None = None,
) -> dict[str, Any]:
    nproc = os.cpu_count() or 1
    if platform_label == "l4" and nproc != 12:
        raise ValueError("F1 L4 host must have exactly 12 vCPU")
    source_commit = os.environ.get("F1_COMMIT", "local-functional")
    verified = load_ledger(ledger, bank_sha256, platform_label, source_commit)
    source_verified = verified
    if probe:
        # Probe exports are excluded one-root diagnostics, never 64-root
        # completed units; validate uploaded ledger but do not skip probe stages.
        verified = {}
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "root-cache"
    cache.mkdir(exist_ok=False)
    roots = bank["roots"][:1] if probe else bank["roots"]
    missing = [label for label in LABELS if label not in verified]
    paths, preparation = {}, time.monotonic()
    if missing:
        jobs = []
        for source in bank["source_sessions"]:
            selected = [r for r in roots if r["source_session"] == source["name"]]
            if selected:
                jobs.append((source, selected, b8, frp3, cache, deadline))
        prep_workers = min(local_workers or nproc, len(jobs))
        if prep_workers == 1:
            for job in jobs:
                paths.update(_prepare_source_job(job))
        elif jobs:
            with mp.get_context("spawn").Pool(prep_workers) as preparation_pool:
                prepared = preparation_pool.imap(_prepare_source_job, jobs)
                for _job in jobs:
                    paths.update(prepared.next(timeout=max(0.001, deadline - time.monotonic())))
    setup_seconds = time.monotonic() - preparation
    model = cpu_model()
    if source_verified and model == "UNKNOWN":
        raise ValueError("F1 cannot resume timings across unmeasured CPU models")
    expected_ids = [r["root_id"] for r in bank["roots"]]
    for config in source_verified.values():
        if [r.get("root_id") for r in config.get("records", [])] != expected_ids:
            raise ValueError("F1 verified schedule has a foreign or partial root allocation")
        if config.get("cpu") != {"nproc": nproc, "model": model}:
            raise ValueError("F1 resumed worker schedules have foreign host capacity/model")
    configs = []
    for label in LABELS:
        _deadline_check(deadline)
        if label in verified:
            configs.append(verified[label])
            continue
        workers = {"one": 1, "half": max(1, nproc // 2), "all": nproc}[label]
        if local_workers is not None:
            if not probe:
                raise ValueError("local worker override is functional-probe-only")
            workers = local_workers
        # Pool startup is recorded separately. The timed labeller window starts
        # only after all workers have imported their libraries and are ready.
        context = mp.get_context("spawn")
        barrier = context.Barrier(workers + 1)
        with context.Pool(workers, initializer=_warmup, initargs=(barrier,)) as pool:
            barrier.wait(timeout=max(0.001, deadline - time.monotonic()))
            started = time.monotonic()
            pending = pool.imap(_label_root, [(str(paths[r["root_id"]]), deadline) for r in roots])
            records = []
            next_progress = time.monotonic() + 110
            last_result = time.monotonic()
            while len(records) < len(roots):
                try:
                    records.append(
                        pending.next(timeout=min(20, max(0.001, deadline - time.monotonic())))
                    )
                    last_result = time.monotonic()
                except mp.TimeoutError:
                    _deadline_check(deadline)
                    if time.monotonic() - last_result >= 300:
                        raise WorkerStall("F1_WORKER_STALL: no completed root for 300 seconds")
                if time.monotonic() >= next_progress:
                    print(
                        f"F1_PROGRESS label={label} done={len(records)} total={len(roots)}",
                        flush=True,
                    )
                    next_progress = time.monotonic() + 110
            elapsed = time.monotonic() - started
        config = {
            "schema": RECEIPT_SCHEMA,
            "status": "COMPLETE",
            "platform": platform_label,
            "bank_sha256": bank_sha256,
            "source_commit": source_commit,
            "transport_probe": probe,
            "schedule_label": label,
            "workers": workers,
            "cpu": {"nproc": nproc, "model": model},
            "root_count": len(roots),
            "elapsed_seconds": elapsed,
            "roots_per_hour": len(roots) * 3600 / elapsed,
            "integrity": {
                "digest_checks": len(records),
                "bare_audit_mismatches": 0,
                "source_full_hash_mismatches": sum(
                    not r["source_digest_matches_local"] for r in records
                ),
            },
            "records": records,
        }
        publish_json(output / "units" / f"config-{label}.json", config)
        configs.append(config)
        print(
            f"F1_CONFIG platform={platform_label} label={label} roots={len(roots)} complete=1",
            flush=True,
        )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "platform": platform_label,
        "bank_sha256": bank_sha256,
        "source_commit": source_commit,
        "transport_probe": probe,
        "configs": configs,
        "root_preparation_seconds": setup_seconds,
        "cpu": {"nproc": nproc, "model": model},
        "outcome_exports": False,
        "skipped_verified": sorted(verified),
    }


def _warmup(barrier: Any) -> None:
    import torch

    from research.t1_commitment_20261015.replay import configure_cpu

    configure_cpu()
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("F1 worker numerical thread cap was not applied")
    from src.simd_env import eval_engine

    from . import factory

    cls = eval_engine._terminal_hero_sim_class("grid")
    setattr(eval_engine, cls.__name__, cls)
    copyreg.pickle(cls, factory._reduce_terminal_sim)
    barrier.wait(timeout=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-bank")
    build.add_argument("--session", action="append", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    job = sub.add_parser("run")
    for name in ("bank", "ledger", "b8", "frp3", "output"):
        job.add_argument("--" + name, type=Path, required=True)
    job.add_argument("--platform", choices=("cpu", "l4"), required=True)
    job.add_argument("--session-seconds", type=float)
    job.add_argument("--finalize-seconds", type=float)
    job.add_argument("--deadline-seconds", type=float)
    job.add_argument("--workers", type=int)
    job.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.command == "build-bank":
        build_bank(args.session, args.out)
        return
    if args.session_seconds is not None:
        budget = session_deadline(args.session_seconds, args.finalize_seconds)
        deadline = budget["collection_deadline_monotonic"]
        publish_json(args.output / "session-budget.json", budget)
    elif args.deadline_seconds and args.probe:
        deadline = time.monotonic() + args.deadline_seconds
    else:
        parser.error(
            "qualified whole-session budget required (local probes may use --deadline-seconds)"
        )
    bank = load_bank(args.bank)
    digest = bank_digest(args.bank)
    from .factory import FactoryInvalid

    try:
        result = run(
            bank,
            digest,
            args.ledger,
            args.b8,
            args.frp3,
            args.platform,
            args.output,
            deadline,
            probe=args.probe,
            local_workers=args.workers,
        )
    except (TimeoutError, mp.TimeoutError, WorkerStall) as error:
        result = {
            "schema": RECEIPT_SCHEMA,
            "status": "INCOMPLETE",
            "platform": args.platform,
            "bank_sha256": digest,
            "reason": str(error),
            "worker_stall": isinstance(error, WorkerStall),
        }
    except (ValueError, FactoryInvalid) as error:
        result = {
            "schema": RECEIPT_SCHEMA,
            "status": "INVALID",
            "platform": args.platform,
            "bank_sha256": digest,
            "reason": str(error),
        }
        publish_json(args.output / "INVALID.json", result)
    publish_json(args.output / "runtime-result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "configs"}, sort_keys=True))
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
