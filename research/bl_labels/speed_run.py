"""Prepare, collect and compare the excluded round-27 speed-identity bank.

Collection never compares outcomes. Analysis requires an origin/main-published
metadata ledger before opening any science payload. Actual runs are operator-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import speed_equiv as eq
from .exports import _verify, publish_json

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("speed_worker.py")
THREADS = {
    k: "1"
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}
THRESHOLDS = {"forward_q": 0.20, "digests": 0.08, "snapshots": 0.05, "cold_start": 0.04}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_tree(root: Path, pin: str) -> None:
    """Refuse another pin, dirty tree, or any active/live D256 worktree."""
    root = root.resolve()
    if root.name == "snake-dqn-run-bl-f0-mac-d256-02.noindex" or (root / ".IN_USE_LOCK").exists():
        raise ValueError("reference worktree is live; HOLD")
    if git(root, "rev-parse", "HEAD") != pin or git(root, "status", "--porcelain"):
        raise ValueError("T-EQ worktree must be clean at the exact requested pin")
    if git(root, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
        raise ValueError("T-EQ scientific worktree must be detached")


def prepare(out: Path, new_pin: str, *, materialize: bool = True) -> dict[str, Any]:
    """Freeze all B-EQ roster plans and inputs before a single parent runs."""
    if out.exists():
        raise FileExistsError("speed package must be fresh")
    if len(new_pin) != 40 or any(c not in "0123456789abcdef" for c in new_pin):
        raise ValueError("full candidate SHA required")
    from .mac_package import LEGACY_FACTORY_COMMIT, d256_allocation, future_control
    from .manifests import materialize_gate_plans

    allocation = eq.beq_allocation()
    if materialize:
        allocation = materialize_gate_plans(allocation)
    excluded = {
        r["seed"] for r in d256_allocation(f"f0-hindsight-labels-{LEGACY_FACTORY_COMMIT}-mac-d256")
    }
    for stage in ("d1024", "d4096"):
        excluded.update(r["seed"] for r in future_control(new_pin, stage=stage)["allocation"])
    if any(r["seed"] in excluded for r in allocation):
        raise ValueError("B-EQ/production seed collision")
    bank = {
        "schema": "bl-labels-f0-speed-equiv-bank/v1",
        "namespace": eq.NAMESPACE,
        "excluded_from_production": True,
        "allocation": allocation,
        "old_pin": eq.OLD_PIN,
        "new_pin": new_pin,
        "ruling_sha256": eq.RULING_SHA256,
        "f1_bank_sha256": eq.F1_BANK_SHA256,
        "profile_build_thresholds": THRESHOLDS,
        "collision_audit": {"d256_d1024_d4096": "NO_COLLISIONS"},
    }
    out.mkdir(parents=True)
    publish_json(out / "beq-manifest.json", bank)
    return bank


def verify_bank(path: Path, new_pin: str) -> dict[str, Any]:
    _verify(path)
    bank = json.loads(path.read_text())
    if (
        bank.get("namespace") != eq.NAMESPACE
        or bank.get("old_pin") != eq.OLD_PIN
        or bank.get("new_pin") != new_pin
        or bank.get("ruling_sha256") != eq.RULING_SHA256
        or bank.get("profile_build_thresholds") != THRESHOLDS
        or bank.get("excluded_from_production") is not True
    ):
        raise ValueError("foreign speed bank")
    expected = eq.beq_allocation()
    rows = bank["allocation"]
    if len(rows) != 12:
        raise ValueError("B-EQ must have exactly 12 parents")
    for row, wanted in zip(rows, expected, strict=True):
        if {k: row.get(k) for k in wanted} != wanted:
            raise ValueError("B-EQ allocation differs")
        if row["stream"] != "watch6" and not isinstance(row.get("plan"), dict):
            raise ValueError("B-EQ gate plan was not frozen")
    return bank


def make_tree(repo: Path, path: Path, pin: str) -> None:
    if not path.exists():
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), pin],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    verify_tree(path, pin)


def metadata(paths: list[Path], base: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(p.relative_to(base)), "size": p.stat().st_size, "sha256": _verify(p)}
        for p in sorted(paths)
    ]


def run_cell(
    tree: Path,
    pin: str,
    bank_path: Path,
    indexes: list[int],
    width: int,
    out: Path,
    b8: Path,
    frp3: Path,
    monitor: Any,
    *,
    profile: bool = False,
    f1_bank: Path | None = None,
) -> dict[str, Any]:
    """Actual bounded independent processes; never pretend a serial loop is W6."""
    verify_tree(tree, pin)
    out.mkdir(parents=True, exist_ok=False)
    waiting = list(indexes)
    active: list[tuple[subprocess.Popen[Any], Any]] = []
    env = {**os.environ, **THREADS, "SNAKE_DQN_DEVICE": "cpu"}
    env.pop("PYTHONPATH", None)
    started = time.monotonic()
    try:
        while waiting or active:
            while waiting and len(active) < width:
                index = waiting.pop(0)
                launched = time.monotonic()
                command = [
                    sys.executable,
                    str(WORKER),
                    "--worktree",
                    str(tree),
                    "--pin",
                    pin,
                    "--output",
                    str(out / f"parent-{index:02d}"),
                    "--b8",
                    str(b8),
                    "--frp3",
                    str(frp3),
                    "--deadline",
                    str(monitor.deadline),
                    "--launched",
                    str(launched),
                ]
                if f1_bank is not None:
                    command += ["--bank", str(f1_bank)]
                else:
                    command += ["--manifest", str(bank_path), "--index", str(index)]
                if profile:
                    command += ["--profile"]
                log = (out / f"parent-{index:02d}.log").open("xb")
                process = subprocess.Popen(
                    command,
                    cwd=tree,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active.append((process, log))
            monitors = [
                SimpleNamespace(pid=p.pid, is_alive=lambda p=p: p.poll() is None)
                for p, _log in active
                if p.poll() is None
            ]
            if time.monotonic() >= monitor.next_sample and monitors:
                # F1 owns spawned pool descendants in its fresh process group.
                # Include their RSS rather than sampling only the coordinator.
                groups = {p.pid for p, _log in active if p.poll() is None}
                listing = subprocess.check_output(["/bin/ps", "-axo", "pid=,pgid="], text=True)
                descendants = []
                for line in listing.splitlines():
                    fields = line.split()
                    if len(fields) == 2 and all(v.isdigit() for v in fields):
                        pid_value, group = map(int, fields)
                        if group in groups and pid_value not in groups:
                            descendants.append(
                                SimpleNamespace(pid=pid_value, is_alive=lambda: True)
                            )
                monitors.extend(descendants)
            monitor.poll(monitors)
            for process, log in list(active):
                status = process.poll()
                if status is not None:
                    log.close()
                    active.remove((process, log))
                    if status:
                        raise RuntimeError("isolated speed worker failed; inspect retained log")
            if active:
                time.sleep(0.1)
    finally:
        # All children are owned sessions; terminate/reap only those started here.
        import signal

        for process, _log in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            log.close()
    elapsed = time.monotonic() - started
    paths = list(out.glob("parent-*/payloads/*.json")) + list(out.glob("parent-*/diagnostic.json"))
    paths += list(out.glob("parent-*/runtime-result.json"))
    phase = {
        "schema": "bl-labels-f0-speed-phase/v1",
        "pin": pin,
        "workers": width,
        "elapsed_seconds": elapsed,
        "files": metadata(paths, out),
        "profile": profile,
        "parent_indexes": indexes,
    }
    publish_json(out / "phase-ledger.json", phase)
    return phase


def collect(args: argparse.Namespace) -> None:
    from research.mac_lease import held

    from . import f1_ruler, w12_calibration

    pin = git(ROOT, "rev-parse", "HEAD")
    package = args.package.resolve()
    bank = verify_bank(package / "beq-manifest.json", pin)
    if f1_ruler.bank_digest(args.f1_bank) != eq.F1_BANK_SHA256:
        raise ValueError("foreign frozen F1 bank")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("fresh collection output required")
    # Creating isolated trees is offline; no D256 worktree is opened.
    old = package / "old-reference"
    new = package / "new-reference"
    make_tree(ROOT, old, eq.OLD_PIN)
    make_tree(ROOT, new, pin)
    if not all(p.is_file() for p in (args.b8, args.frp3)):
        raise ValueError("missing pinned checkpoint")
    os.environ.update(THREADS)
    with held("bl-f0-speed-equiv", "excluded bank byte identity", seconds=args.timeout_seconds):
        output.mkdir(parents=True)
        monitor = w12_calibration.Monitor(
            output, args.timing_armed, time.monotonic() + args.timeout_seconds
        )
        phases: dict[str, Any] = {}
        if args.profile:
            chosen = [
                next(i for i, r in enumerate(bank["allocation"]) if r["stream"] == s)
                for s in ("gate-mixed", "watch6")
            ]
            phases["profile-old"] = run_cell(
                old,
                eq.OLD_PIN,
                package / "beq-manifest.json",
                chosen,
                1,
                output / "profile-old",
                args.b8,
                args.frp3,
                monitor,
                profile=True,
            )
        else:
            for label, tree, source, width in (
                ("old-w1", old, eq.OLD_PIN, 1),
                ("new-w1", new, pin, 1),
                ("old-w6", old, eq.OLD_PIN, 6),
                ("new-w6", new, pin, 6),
            ):
                phases[label] = run_cell(
                    tree,
                    source,
                    package / "beq-manifest.json",
                    list(range(12)),
                    width,
                    output / label,
                    args.b8,
                    args.frp3,
                    monitor,
                )
            for label, tree, source in (("old-f1", old, eq.OLD_PIN), ("new-f1", new, pin)):
                phases[label] = run_cell(
                    tree,
                    source,
                    package / "beq-manifest.json",
                    [0],
                    1,
                    output / label,
                    args.b8,
                    args.frp3,
                    monitor,
                    f1_bank=args.f1_bank,
                )
            # Forecast only; interleaved same-bank W6 observations.
            if args.timing:
                for i, (tree, source) in enumerate(
                    ((old, eq.OLD_PIN), (new, pin), (old, eq.OLD_PIN), (new, pin))
                ):
                    label = f"timing-{i}"
                    phases[label] = run_cell(
                        tree,
                        source,
                        package / "beq-manifest.json",
                        list(range(12)),
                        6,
                        output / label,
                        args.b8,
                        args.frp3,
                        monitor,
                    )
        publish_json(
            output / "phase-ledger.json",
            {
                "schema": "bl-labels-f0-speed-phase-ledger/v1",
                "old_pin": eq.OLD_PIN,
                "new_pin": pin,
                "bank_path": str(package / "beq-manifest.json"),
                "bank_sha256": _verify(package / "beq-manifest.json"),
                "f1_bank_path": str(args.f1_bank.resolve()),
                "f1_bank_sha256": eq.F1_BANK_SHA256,
                "b8_sha256": eq.sha_file(args.b8),
                "frp3_sha256": eq.sha_file(args.frp3),
                "phases": {
                    k: {
                        "path": str(output / k / "phase-ledger.json"),
                        "sha256": _verify(output / k / "phase-ledger.json"),
                    }
                    for k in phases
                },
                "profile": args.profile,
                "outcomes_compared": False,
                "ruling_sha256": eq.RULING_SHA256,
            },
        )
    print(
        json.dumps(
            {
                "status": "COLLECTED_NOT_COMPARED",
                "ledger": str(output / "phase-ledger.json"),
                "next": "publish metadata ledger to origin/main, then analyze",
            }
        )
    )


def verify_publication(output: Path, repo: Path, path: str) -> dict[str, Any]:
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"origin/main:{path}"])
    publication = json.loads(raw)
    ledger = output / "phase-ledger.json"
    if publication != {"ledger": str(ledger.resolve()), "sha256": _verify(ledger)}:
        raise ValueError("phase ledger has not been published unchanged on origin/main")
    return json.loads(ledger.read_text())


def load_phase(
    output: Path, name: str, ledger: dict[str, Any], *, metadata_only: bool = False
) -> tuple[list[dict], list[dict]]:
    binding = ledger["phases"][name]
    path = output / name / "phase-ledger.json"
    if str(path) != binding["path"] or _verify(path) != binding["sha256"]:
        raise ValueError("phase ledger changed")
    phase = json.loads(path.read_text())
    expected_pin = (
        ledger["old_pin"] if name.startswith("old") or name == "profile-old" else ledger["new_pin"]
    )
    expected_width = 6 if name.endswith("w6") or name.startswith("timing-") else 1
    if name.startswith("timing-"):
        expected_pin = ledger["old_pin"] if name[-1] in ("0", "2") else ledger["new_pin"]
    if phase.get("pin") != expected_pin or phase.get("workers") != expected_width:
        raise ValueError("phase source/worker identity differs")
    names = [r["path"] for r in phase["files"]]
    actual = list(path.parent.glob("parent-*/payloads/*.json"))
    actual += list(path.parent.glob("parent-*/diagnostic.json"))
    actual += list(path.parent.glob("parent-*/runtime-result.json"))
    if len(names) != len(set(names)) or set(names) != {
        str(p.relative_to(path.parent)) for p in actual
    }:
        raise ValueError("phase file inventory incomplete or duplicate")
    payloads, diagnostics = [], []
    for row in phase["files"]:
        target = path.parent / row["path"]
        if not target.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("foreign phase path")
        if target.stat().st_size != row["size"] or _verify(target) != row["sha256"]:
            raise ValueError("phase payload changed")
        if metadata_only:
            continue
        value = json.loads(target.read_text())
        if "/payloads/" in str(target):
            payloads.append(value)
        elif target.name == "diagnostic.json":
            diagnostics.append(value)
    if metadata_only:
        return [], []
    if name.endswith("w1") or name.endswith("w6"):
        bank = json.loads(Path(ledger["bank_path"]).read_text())
        expected = {r["allocation_id"]: r for r in bank["allocation"]}
        if phase.get("parent_indexes") != list(range(12)):
            raise ValueError("phase parent allocation differs")
        if {d.get("allocation_id") for d in diagnostics} != set(expected) or len(diagnostics) != 12:
            raise ValueError("phase diagnostics allocation differs")
        if any(d.get("pin") != expected_pin for d in diagnostics):
            raise ValueError("diagnostic source differs")
        seen = set()
        for payload in payloads:
            parent = payload.get(
                "allocation_id", payload.get("allocation", {}).get("allocation_id")
            )
            if parent not in expected or payload.get("allocation") != expected[parent]:
                raise ValueError("science payload allocation differs")
            schema = payload.get("schema")
            ident = (parent, schema, payload.get("life"))
            if ident in seen:
                raise ValueError("duplicate science payload")
            seen.add(ident)
        for parent in expected:
            for schema in ("bl-labels-f0-unit/v1", "bl-labels-f0-session/v1"):
                if (parent, schema, None) not in seen:
                    raise ValueError("missing science unit/session")
            session = next(
                p
                for p in payloads
                if p.get("allocation_id") == parent and p.get("schema") == "bl-labels-f0-session/v1"
            )
            deaths = session["counters"]["deaths"]
            if type(deaths) is not int or deaths < 0:
                raise ValueError("invalid death-life allocation")
            lives = {
                life for p, schema, life in seen if p == parent and schema == "bl-labels-f0-life/v1"
            }
            if lives != set(range(deaths)):
                raise ValueError("missing or foreign completed life")
    return payloads, diagnostics


def pair_inputs(payloads: list[dict], diagnostics: list[dict]) -> list[dict]:
    by_parent = {d["allocation_id"]: d["advisory"] for d in diagnostics}
    result = [dict(p, rows=[dict(r) for r in p.get("rows", [])]) for p in payloads]
    for payload in result:
        parent = payload.get("allocation_id", payload.get("allocation", {}).get("allocation_id"))
        for row in payload.get("rows", []):
            row["_speed_equiv_advisory"] = by_parent[parent][f"{row['life']}:{row['decision']}"]
    return result


def _analyze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    ledger = verify_publication(output, args.publication_repo, args.publication_path)
    verify_bank(Path(ledger["bank_path"]), ledger["new_pin"])
    if _verify(Path(ledger["bank_path"])) != ledger["bank_sha256"]:
        raise ValueError("B-EQ bank changed after collection")
    if ledger.get("profile"):
        _payloads, diagnostics = load_phase(output, "profile-old", ledger)
        if len(diagnostics) != 2:
            raise ValueError("profile requires exactly two frozen parents")
        wall = sum(d["collector_seconds"] for d in diagnostics)
        cold = sum(d["spawn_to_entry_seconds"] for d in diagnostics)
        shares = {
            name: sum(d["profile_seconds"][name] for d in diagnostics) / (wall + cold)
            for name in ("forward_q", "digests", "snapshots", "raw_observation", "v8_flood")
        }
        shares["cold_start"] = cold / (wall + cold)
        publish_json(
            output / "profile-receipt.json",
            {
                "schema": "bl-labels-f0-speed-profile/v1",
                "old_pin": eq.OLD_PIN,
                "cumulative_shares": shares,
                "thresholds": THRESHOLDS,
                "eligible_to_build": {k: shares[k] >= v for k, v in THRESHOLDS.items()},
                "builder_day_max": 1,
                "digest_plus_raw_observation_share_of_play_upper_bound": sum(
                    d["profile_seconds"]["digests"] + d["profile_seconds"]["raw_observation"]
                    for d in diagnostics
                )
                / sum(d["play_seconds"] for d in diagnostics),
                "implementation_authorized_without_separate_teq": False,
                "flood_new_round_threshold": 0.25,
                "lazy_bookkeeping_requires_new_judgment": True,
                "outcomes": "NOT_EMITTED",
                "phase_ledger_sha256": _verify(output / "phase-ledger.json"),
            },
        )
        return
    comparisons = {}
    counter_gate = True
    checked_counters = set()
    for label, left, right in (
        ("old_vs_new_w1", "old-w1", "new-w1"),
        ("old_vs_new_w6", "old-w6", "new-w6"),
        ("new_w1_vs_w6", "new-w1", "new-w6"),
    ):
        # Canonical original payloads first; diagnostics never enter this proof.
        a, ad = load_phase(output, left, ledger)
        b, bd = load_phase(output, right, ledger)
        for arm, diagnostics in ((left, ad), (right, bd)):
            if arm.startswith("new") and arm not in checked_counters:
                checked_counters.add(arm)
                for diagnostic in diagnostics:
                    counts = diagnostic["counts"]
                    if not (
                        counts["living_decisions"] > 0
                        and counts["natural_forwards"]
                        == counts["natural_rasters"]
                        == counts["living_decisions"]
                        and counts["forced_forwards"] == 0
                    ):
                        counter_gate = False
        result = eq.compare_payloads(a, b, compare_pairs=False)
        pairs_a = eq.pair_digests(pair_inputs(a, ad))
        pairs_b = eq.pair_digests(pair_inputs(b, bd))
        result["pairs"] = {"old": pairs_a, "new": pairs_b}
        if pairs_a != pairs_b:
            result.update(status="FAIL", reason="F4 pair/mass-gap identity differs")
        comparisons[label] = result
        del a, b, ad, bd
    f1_paths = []
    for name in ("old-f1", "new-f1"):
        load_phase(output, name, ledger)  # Verify metadata before opening F1 records.
        f1_paths.append(output / name / "parent-00/runtime-result.json")
    try:
        comparisons["f1"] = eq.f1_identity(
            *f1_paths, bank=Path(ledger["f1_bank_path"]), expected_new_sha=ledger["new_pin"]
        )
    except ValueError as error:
        if str(error) != "F1 worker-width identity failed":
            raise
        comparisons["f1"] = {
            "status": "FAIL",
            "reason": str(error),
            "roots": 64,
            "bank_sha256": eq.F1_BANK_SHA256,
        }
    status = (
        "PASS"
        if counter_gate and all(v["status"] == "PASS" for v in comparisons.values())
        else "FAIL"
    )
    receipt = {
        "schema": eq.SCHEMA,
        "status": status,
        "old": {"commit": ledger["old_pin"]},
        "new": {"commit": ledger["new_pin"]},
        "bank": {"sha256": ledger["bank_sha256"], "path": ledger["bank_path"]},
        "f1_bank_sha256": eq.F1_BANK_SHA256,
        "comparisons": comparisons,
        "phase_ledger_sha256": _verify(output / "phase-ledger.json"),
        "publication_path": args.publication_path,
        "ruling_sha256": eq.RULING_SHA256,
        "call_counter_gate": "PASS" if counter_gate else "FAIL",
        "checkpoints": {"b8": ledger["b8_sha256"], "frp3": ledger["frp3_sha256"]},
    }
    timing = []
    for index in range(4):
        name = f"timing-{index}"
        if name in ledger["phases"]:
            load_phase(output, name, ledger)
            phase = json.loads((output / name / "phase-ledger.json").read_text())
            timing.append(
                {
                    "order": index,
                    "pin": phase["pin"],
                    "workers": 6,
                    "elapsed_seconds": phase["elapsed_seconds"],
                    "worker_seconds": 6 * phase["elapsed_seconds"],
                    "parents": len(phase["parent_indexes"]),
                }
            )
    if timing:
        if len(timing) != 4:
            raise ValueError("interleaved timing sequence incomplete")
        publish_json(
            output / "timing-table.json",
            {
                "schema": "bl-labels-f0-speed-timing/v1",
                "forecast_only": True,
                "cells": timing,
                "phase_ledger_sha256": _verify(output / "phase-ledger.json"),
            },
        )
        receipt["timing_table_sha256"] = _verify(output / "timing-table.json")
    publish_json(output / "receipt.json", receipt)
    print(json.dumps({"status": status, "receipt": str(output / "receipt.json")}))
    if status != "PASS":
        raise SystemExit(2)


def analyze(args: argparse.Namespace) -> None:
    """Comparison is an actual Mac task and holds the same shared lease."""
    from research.mac_lease import held

    from .mac_driver import _guard_snapshot

    with held("bl-f0-speed-analyze", "published excluded-bank comparison", seconds=3600):
        _guard_snapshot(args.output.resolve())
        _analyze(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--new-pin", required=True)
    run = sub.add_parser("collect")
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--b8", type=Path, required=True)
    run.add_argument("--frp3", type=Path, required=True)
    run.add_argument("--f1-bank", type=Path, default=eq.F1_BANK)
    run.add_argument("--timeout-seconds", type=float, default=7200)
    run.add_argument(
        "--timing-armed",
        type=Path,
        default=Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed"),
    )
    run.add_argument("--profile", action="store_true")
    run.add_argument("--timing", action="store_true")
    publication = sub.add_parser("publication")
    publication.add_argument("--output", type=Path, required=True)
    publication.add_argument("--out", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--publication-repo", type=Path, default=ROOT)
    analyze_parser.add_argument("--publication-path", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.new_pin)
    elif args.command == "collect":
        if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 10800:
            parser.error("finite timeout <=10800 seconds required")
        collect(args)
    elif args.command == "publication":
        ledger = args.output.resolve() / "phase-ledger.json"
        publish_json(args.out, {"ledger": str(ledger), "sha256": _verify(ledger)})
        print("Commit and push this metadata record before analysis: " + str(args.out))
    else:
        analyze(args)


if __name__ == "__main__":
    main()
