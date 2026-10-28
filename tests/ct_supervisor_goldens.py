"""Regenerate C/T goldens using the unchanged qualified broker and runtime.

Offline only: the sole CLI binary and OAuth helper are the existing subprocess
fakes. Faults enter through provider state, a virtual clock and Broker.observer;
no supervisor/runtime encoder, classifier or receipt writer is replaced.
Run with the project's CPU wrapper and --destination pointing at a NEW directory.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from research.colab_cli import runtime, supervisor
from research.sandbox_a_resident.ct_run import run_ct_experiment
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.ct_units import empty_ledger
from research.sandbox_a_resident.test_ct_runner_and_resume import (
    _save_tiny_checkpoints,
    _write_tiny_initial_packet,
)
from tests.test_colab_cli_supervisor import _fake_env, _manifest, _patch_virtual_clock

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "clean-completion",
    "wall-cap",
    "deadline-helper",
    "deadline-cli-call",
    "unit-runaway-guard",
    "unit-cap-account-lock",
    "calibration-reserve",
    "study-reserve",
    "lease-renewal-timeout",
    "rpc-exec-poll-timeout",
    "rpc-busy-timeout",
    "child-timeout",
    "child-lease-expired",
    "child-credential-expiring",
    "child-nonmechanical-exit",
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runtime_tree(work: Path, stage_tree: Path, mode: str, job_id: str) -> Path:
    """Actual runtime.supervise child and actual completed-export scanner."""
    root = work / "runtime-producer"
    root.mkdir()
    command = (
        "import os,shutil,signal,time; from pathlib import Path; "
        "out=Path(os.environ['COLAB_OUTPUT_DIR'])/'ct'; "
        f"shutil.copytree({str(stage_tree)!r},out); "
        "signal.signal(signal.SIGUSR1,lambda *_:exit(138)); "
        "print('CT_CHILD_READY',flush=True); "
        + ("exit(7)" if mode == "child-nonmechanical-exit" else "time.sleep(60)")
    )
    spec = {
        "job_id": job_id,
        "generation": "replaced-at-provider-boundary",
        "root": str(root),
        "command": [sys.executable, "-c", command],
        "env": {},
        "timeout_seconds": 0.8 if mode != "child-lease-expired" else 60,
        "lease_seconds": 0.8 if mode == "child-lease-expired" else 180,
        "checkpoint_grace_seconds": 0.5,
        "max_export_bytes": 32 * 1024 * 1024,
        "release_endpoint": "offline-only",
    }
    _json(root / "spec.json", spec)
    # Same local process/release boundary exemptions as test_colab_cli_runtime.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runtime, "_validate_spec", lambda value: value)
        patch.setattr(runtime, "_require_release_safety", lambda *_: None)
        patch.setattr(runtime, "_unassign", lambda *_: None)
        patch.setattr(
            runtime,
            "_release_credential_expiring",
            lambda *_: mode == "child-credential-expiring"
            and (root / "output/ct/stage-0001.pt.complete").exists(),
        )
        assert runtime.supervise(spec) != 0
    return root


def build(
    destination: Path,
    scratch: Path,
    *,
    ct_manifest: bool = False,
    scenarios: tuple[str, ...] = SCENARIOS,
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if ct_manifest:
        canonical = ROOT.parent / "snake-dqn/saved_snakes"
        ckpts = tuple(
            canonical / name
            for name in (
                "b8_s3_t7_20261016.pth",
                "frp3_m3_s12_u60000_20261005.pth",
                "champion_a5_freespace_20260621.pth",
            )
        )
    else:
        ckpts = _save_tiny_checkpoints(scratch)
    control = CTControl(
        source_commit=pin,
        worlds=6,
        collection_ticks=12,
        max_body_capacity=128,
        max_deaths=2,
        lookbacks=(2, 4),
        max_ordinary_roots=4,
        anchor_rows=16,
        min_death_length=1,
        min_held_out_deaths=0,
        min_held_out_death_worlds=0,
        min_held_out_ordinary_roots=1,
        min_held_out_anchor_rows=1,
        branch_horizon=3,
        minibatches=2,
        searched_batch_size=2,
        anchor_batch_size=4,
        bootstrap_resamples=50,
        scoring_chunks=2,
        branch_batch_size=16,
        snapshot_store_max_bytes=16 * 1024 * 1024,
        tiny=True,
    )
    ledger = scratch / "empty-ledger.json"
    _json(ledger, empty_ledger(control.to_dict()))

    class StopAfterStageOne:
        def progress(self, stage: str, *_: object) -> None:
            if stage == "stage-0001-published":
                os.kill(os.getpid(), signal.SIGUSR1)

    stage_tree = scratch / "ct-output"
    with (destination / "ct-job.log").open("w") as log, contextlib.redirect_stdout(log):
        code, _ = run_ct_experiment(
            control,
            b8_checkpoint=ckpts[0],
            frp3_checkpoint=ckpts[1],
            champion_checkpoint=ckpts[2],
            output=stage_tree,
            ledger_path=ledger,
            watchdog=StopAfterStageOne(),
        )
    assert code == 138 and (stage_tree / "stage-0001.pt.complete").is_file()
    _json(destination / "control.json", control.to_dict())
    # Share the actual C/T output bytes once; each supervisor output tree retains
    # its own real transferred exports, not a hand-built stage descriptor.
    shutil.copytree(stage_tree, destination / "ct-output")

    for scenario in scenarios:
        work = scratch / scenario
        work.mkdir()
        captured = destination / scenario
        captured.mkdir()
        now = [0.0]
        active: dict = {}
        armed = [False]
        with pytest.MonkeyPatch.context() as patch:
            binary = _fake_env(patch, work)
            patch.setattr(supervisor, "ACCOUNT_ROOT", work / "private-account")
            _patch_virtual_clock(patch, now, advance_sleep=True)
            if ct_manifest:

                def monotonic() -> float:
                    now[0] += 0.001
                    return now[0]

                supervisor.time.monotonic = monotonic
            manifest = _manifest(work, rate=6.84)
            manifest["source_paths"] = ["research/colab_cli/acceptance_probe.sh"]
            manifest["commit"] = pin
            if ct_manifest:
                initial = work / "initial-packet"
                _write_tiny_initial_packet(initial, control, ckpts, work / "job-output")
                manifest = json.loads((initial / "dispatch-manifest.json").read_text())
                # A tiny offline entrypoint probe uses the qualified canary
                # admission; production C/T sandbox admission requires a
                # production control and is verified separately after resealing.
                from research.colab_cli import canary

                manifest["timeout_minutes"] = 3
                manifest["requirements"]["est_minutes"] = 1
                manifest["budget"]["unit_cap"] = 1
                manifest["canary"] = {
                    "schema": canary.SCHEMA,
                    "family": canary.family(manifest),
                    "mode": "probe",
                }
                manifest["env"].update(
                    SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/sandbox_a_resident"
                )
                _json(initial / "dispatch-manifest.json", manifest)
                # Fake's provider debit matches this frozen C/T manifest.
                patch.setenv("FAKE_COLAB_OWN_RATE", "8.0")
            if scenario in {"calibration-reserve", "study-reserve"}:
                manifest["budget"]["reserve_units"] = 100.0

            def observe(event: str, context: dict) -> None:
                if scenario == "calibration-reserve" and event == "runtime_ready":
                    state = json.loads((work / "fake-state.json").read_text())
                    state["balance"] = 99.0
                    _json(work / "fake-state.json", state)
                if event != "ready":
                    return
                active.update(context)
                cli, generation = context["cli"], context["generation"]
                state = json.loads((work / "fake-state.json").read_text())
                remote = work / "remote" / cli.session / context["root"].lstrip("/")
                if scenario.startswith("child-"):
                    actual = _runtime_tree(work, stage_tree, scenario, manifest["job_id"])
                    for name in ("state.json", "terminal.json", "lease.json"):
                        data = json.loads((actual / name).read_text())
                        data["generation"] = generation
                        _json(actual / name, data)
                    shutil.copytree(actual / "output", remote / "output", dirs_exist_ok=True)
                    value = runtime.read_state(actual, generation, renew=False, compact=True)
                else:
                    shutil.copytree(stage_tree, remote / "output/ct", dirs_exist_ok=True)
                    # Use the qualified runtime's encoder/scanner for the real
                    # nonterminal tree. A clean terminal uses the fake's native
                    # completion encoder and deliberately cannot be continued.
                    value = {
                        "state": {
                            "job_id": manifest["job_id"],
                            "generation": generation,
                            "status": "running",
                            "max_export_bytes": 32 * 1024 * 1024,
                        },
                        "terminal": None,
                        "exports": runtime._exports(remote, 32 * 1024 * 1024),
                    }
                    _json(remote / "state.json", {**value["state"], "exports": value["exports"]})
                    _json(remote / "lease.json", {"generation": generation, "renewed_at": 0})
                    value = runtime.read_state(remote, generation, renew=False, compact=True)
                    if scenario == "clean-completion":
                        term = state["runtime"][cli.session]["terminal"]
                        term["exports"] = runtime._exports(remote, 32 * 1024 * 1024)
                        value["terminal"] = term
                state["runtime"][cli.session]["ct_read_state"] = value
                _json(work / "fake-state.json", state)
                armed[0] = scenario not in {"clean-completion"} and not scenario.startswith(
                    "child-"
                )

            clock = supervisor.time
            sleep = clock.sleep

            def advance(seconds: float) -> None:
                sleep(seconds)
                if not armed[0]:
                    return
                armed[0] = False
                cli = active["cli"]
                if scenario == "wall-cap":
                    now[0] = float(manifest["timeout_minutes"]) * 60
                elif scenario == "deadline-helper":
                    cli.deadline = now[0]
                    cli.remaining_seconds(1)
                elif scenario == "deadline-cli-call":
                    cli.deadline = now[0]
                    cli._call_once("usage", timeout=1)
                elif scenario == "unit-runaway-guard":
                    cli.priced_deadline = now[0]
                    with supervisor.account_locked(cli):
                        pass
                elif scenario == "unit-cap-account-lock":
                    import fcntl

                    cli.priced_deadline = now[0]
                    with (supervisor.ACCOUNT_ROOT / "broker.lock").open("a") as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        with supervisor.account_locked(cli):
                            pass
                elif scenario == "study-reserve":
                    state = json.loads((work / "fake-state.json").read_text())
                    state["balance"] = 99.0
                    _json(work / "fake-state.json", state)
                elif scenario in {
                    "lease-renewal-timeout",
                    "rpc-exec-poll-timeout",
                    "rpc-busy-timeout",
                }:
                    call = "upload" if scenario == "lease-renewal-timeout" else "exec"
                    if scenario == "rpc-busy-timeout":
                        patch.setenv("FAKE_COLAB_MODE", "busy_exec")
                    patch.setenv(
                        "FAKE_COLAB_FAULT_PLAN",
                        json.dumps(
                            {"schedule": [{"call": call, "fault": "timeout"} for _ in range(3)]}
                        ),
                    )
                    if call == "upload":
                        cli._put_lease(active["root"], active["generation"], 1)

            clock.sleep = advance
            log = io.StringIO()
            failure = None
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    supervisor.Broker(work / "broker-state", binary, observer=observe).run(manifest)
                except supervisor.Refused as exc:
                    failure = str(exc)
            (captured / "supervisor.log").write_text(log.getvalue())
            shutil.copytree(work / "job-output", captured / "output")
            if ct_manifest:
                shutil.copytree(work / "initial-packet", captured / "initial-packet")
            shutil.copy2(work / "fake-calls.jsonl", captured / "provider-calls.jsonl")
            shutil.copy2(work / "fake-state.json", captured / "provider-state.json")
            if (work / "runtime-producer").exists():
                shutil.copytree(work / "runtime-producer", captured / "runtime-producer")
            receipt = json.loads((captured / "output/receipt.json").read_text())
            _json(
                captured / "capture.json",
                {
                    "scenario": scenario,
                    "producer_commit": pin,
                    "transport_tree": "5833e54e9c5396121cbbf99a5cd4e52006e564fd",
                    "failure": failure,
                    "status": receipt["status"],
                    "error": receipt["error"],
                    "failure_class": receipt["failure_class"],
                    "stop_verified": receipt["stop_verified"],
                    "boundary_injections": (
                        "offline provider, virtual clock, observer, "
                        "runtime local release exemptions"
                    ),
                },
            )
            print(
                scenario, receipt["status"], receipt["failure_class"], receipt["error"], flush=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--ct-manifest", action="store_true")
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    args = parser.parse_args()
    build(
        args.destination,
        args.scratch,
        ct_manifest=args.ct_manifest,
        scenarios=tuple(args.scenario) if args.scenario else SCENARIOS,
    )
