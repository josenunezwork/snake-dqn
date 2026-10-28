"""Capture offline producer evidence; never contact the official Colab binary.

This developer tool runs the unchanged Broker against the existing subprocess
fake. Only clocks, fake-provider state and subprocess fault boundaries are
injected. Captured receipts are never reconstructed or hand-written.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import supervisor

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCENARIOS = (
    "clean-completion",
    "wall-cap",
    "helper-deadline",
    "cli-deadline",
    "runaway-guard",
    "calibration-reserve",
    "lease-renewal-timeout",
    "rpc-exec-poll-timeout",
    "child-nonzero",
    "runtime-timeout",
    "runtime-lease-expiry",
    "provider-loss-no-receipt",
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_into_fake(
    root: Path, payload: dict[str, Any], scenario: str, mp: pytest.MonkeyPatch
) -> None:
    """Run pinned runtime + real train orchestration with the existing tiny CPU seams.

    Runtime validation's /content path restriction and release safety are the
    only runtime seams. The child, native scanner, state and terminal writers
    are executed. Scientific dependencies use the existing production fixture.
    """
    runtime_path = root / "pinned-runtime.py"
    runtime_path.write_bytes(
        subprocess.check_output(
            ["git", "show", f"{payload['manifest']['commit']}:research/colab_cli/runtime.py"],
            cwd=REPO,
        )
    )
    runtime = _load(runtime_path, "golden_pinned_runtime")
    owned = root / "cpu-runtime"
    owned.mkdir()
    child_code = """
import os,time,pytest
from pathlib import Path
from research.sandbox_a_resident import test_training_production as p, training_proxy
from research.sandbox_a_resident.test_training_runner import Task
class ProxyTask(Task):
    def step(self, actions):
        event = super().step(actions)
        event.mass = self.hero_length
        return event
m = pytest.MonkeyPatch()
c,_ = p.production.__wrapped__(m,Path(os.environ['COLAB_OUTPUT_DIR']))
object.__setattr__(c,'outer_iterations',16)
p.train.ResidentTrainingTask.proxy_factory = lambda *args: lambda *args: ProxyTask()
p.train.evaluate = training_proxy.evaluate
p.train.serializable = training_proxy.serializable
def boundary(stage, values):
    if (os.environ.get('GOLDEN_WAIT_AT_BOUNDARY') == '1'
            and stage == 'iteration-complete' and values.get('iteration') == 16):
        time.sleep(30)
p.execute(c,Path(os.environ['COLAB_OUTPUT_DIR'])/'training',on_progress=boundary)
"""
    spec = {
        "job_id": payload["manifest"]["job_id"],
        "generation": payload["generation"],
        "root": str(owned),
        "command": [sys.executable, "-c", child_code],
        "env": {
            "PYTHONPATH": str(REPO),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "GOLDEN_WAIT_AT_BOUNDARY": "0" if scenario == "clean-completion" else "1",
        },
        "timeout_seconds": 8 if scenario == "runtime-timeout" else 30,
        "lease_seconds": 6 if scenario == "runtime-lease-expiry" else 60,
        "checkpoint_grace_seconds": 0.1,
        "max_export_bytes": 16 * 1024 * 1024,
        "release_endpoint": "offline-fake-endpoint",
        "post_terminal_timeout_seconds": 0.2,
    }
    # Pin is loaded from the same immutable Git object as the Broker's bundle.
    with pytest.MonkeyPatch.context() as runtime_mp:
        runtime_mp.setattr(runtime, "_validate_spec", lambda raw: dict(raw))
        runtime_mp.setattr(runtime, "_require_release_safety", lambda *_args: None)
        runtime_mp.setattr(runtime, "_release_credential_expiring", lambda *_args: False)
        runtime_mp.setattr(runtime, "_unassign", lambda *_args: None)
        runtime.supervise(spec)
    terminal = json.loads((owned / "terminal.json").read_text())
    state_path = root / "fake-state.json"
    state = json.loads(state_path.read_text())
    fake_runtime = state["runtime"][payload["cli"].session]
    fake_runtime["terminal"] = terminal
    fake_runtime["state"] = json.loads((owned / "state.json").read_text())
    # Existing fake emulates the provider filesystem; Broker still transports
    # each export and verifies its native runtime scanner hash.
    remote_root = root / "remote" / payload["cli"].session / payload["root"].lstrip("/")
    shutil.copytree(owned / "output", remote_root / "output", dirs_exist_ok=True)
    state_path.write_text(json.dumps(state))


def capture_one(destination: Path, scenario: str) -> dict[str, Any]:
    """Capture one actual receipt and retained output under an isolated fake account."""
    assert scenario in SCENARIOS
    if scenario == "provider-loss-no-receipt" and not os.environ.get("GOLDEN_LOSS_CHILD"):
        env = {**os.environ, "GOLDEN_LOSS_CHILD": "1"}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "research.sandbox_a_resident.capture_supervisor_goldens",
                str(destination.parent),
                "--scenario",
                scenario,
            ],
            env=env,
            cwd=REPO,
        )
        if result.returncode != 91:
            raise RuntimeError(f"offline host loss child returned {result.returncode}")
        provenance = json.loads((destination / "provenance.json").read_text())
        scratch = Path(provenance["scratch_removed_by_parent"])
        if scratch.parent != Path("/private/tmp") or not scratch.name.startswith(
            "train01-goldens-"
        ):
            raise RuntimeError("unexpected private scratch path")
        shutil.rmtree(scratch)
        provenance["files"] = {
            str(p.relative_to(destination)): {"size": p.stat().st_size, "sha256": _sha(p)}
            for p in sorted(destination.rglob("*"))
            if p.is_file() and p.name != "provenance.json"
        }
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return provenance
    if destination.exists():
        raise FileExistsError(destination)
    helpers = _load(REPO / "tests/test_colab_cli_supervisor.py", "golden_supervisor_helpers")
    sources = [
        "research/colab_cli/supervisor.py",
        "research/colab_cli/runtime.py",
        "research/colab_cli/failures.py",
        "tests/fixtures/fake_colab.py",
        "tests/fixtures/fake_colab_helper.py",
        "tests/test_colab_cli_supervisor.py",
        "research/sandbox_a_resident/train.py",
        "research/sandbox_a_resident/training_proxy.py",
        "research/sandbox_a_resident/test_training_production.py",
        "research/sandbox_a_resident/test_training_runner.py",
    ]
    provenance: dict[str, Any] = {
        "schema": "sandbox-a-offline-supervisor-golden/v1",
        "scenario": scenario,
        "source_sha256": {p: _sha(REPO / p) for p in sources},
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "provider": "tests/fixtures/fake_colab.py (subprocess; no account access)",
        "receipt_fabricated": False,
        "admission_baseline_sha256": _sha(HERE / "training_resume.py"),
        "injections": [],
    }
    with tempfile.TemporaryDirectory(prefix="train01-goldens-", dir="/private/tmp") as scratch:
        root = Path(scratch)
        with pytest.MonkeyPatch.context() as mp:
            # This fixture uses a private alternate Git index; never the checkout index.
            helpers._isolate_account.__wrapped__(mp, root)
            binary = helpers._fake_env(mp, root)
            now = [0.0]
            helpers._patch_virtual_clock(mp, now, advance_sleep=True)
            manifest = helpers._manifest(root)
            if scenario == "calibration-reserve":
                original_call = supervisor.CLI.call

                def allocation_debit(self: Any, *args: Any, **kwargs: Any) -> Any:
                    value = original_call(self, *args, **kwargs)
                    if args and args[0] == "new":
                        state = json.loads((root / "fake-state.json").read_text())
                        state["balance"] = 0
                        (root / "fake-state.json").write_text(json.dumps(state))
                    return value

                mp.setattr(supervisor.CLI, "call", allocation_debit)
                provenance["injections"].append(
                    "fake account balance becomes zero immediately after allocation"
                )

            def observer(event: str, payload: dict[str, Any]) -> None:
                if event != "ready":
                    return
                cli = payload["cli"]
                if scenario in {"clean-completion", "runtime-timeout", "runtime-lease-expiry"}:
                    _runtime_into_fake(root, payload, scenario, mp)
                elif scenario == "wall-cap":
                    now[0] = 200
                elif scenario == "runaway-guard":
                    original_sessions = cli.sessions

                    def advance_during_read(*args: Any, **kwargs: Any) -> Any:
                        value = original_sessions(*args, **kwargs)
                        now[0] = 600
                        return value

                    cli.sessions = advance_during_read
                elif scenario in {"helper-deadline", "cli-deadline"}:
                    cli.deadline = -1
                    if scenario == "helper-deadline":
                        cli.remaining_seconds(1)
                    else:
                        cli.call("usage")
                elif scenario in {"lease-renewal-timeout", "rpc-exec-poll-timeout"}:
                    original_run = supervisor.subprocess.run

                    def timeout_boundary(args: Any, **kwargs: Any) -> Any:
                        if "-f" in args:
                            code = Path(args[args.index("-f") + 1]).read_text()
                            if (
                                "renew_lease("
                                if scenario == "lease-renewal-timeout"
                                else "read_state("
                            ) in code:
                                now[0] += float(kwargs.get("timeout", 1))
                                raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 1))
                        return original_run(args, **kwargs)

                    mp.setattr(supervisor.subprocess, "run", timeout_boundary)
                    if scenario == "lease-renewal-timeout":
                        cli.renew_runtime_lease(payload["root"], payload["generation"], force=False)
                elif scenario == "child-nonzero":
                    state_path = root / "fake-state.json"
                    state = json.loads(state_path.read_text())
                    runtime = state["runtime"][cli.session]
                    runtime["terminal"].update(status="failed", returncode=23)
                    state_path.write_text(json.dumps(state))
                elif scenario == "provider-loss-no-receipt":
                    # Host loss before finally is reproduced in a separate subprocess below.
                    state_path = root / "fake-state.json"
                    state = json.loads(state_path.read_text())
                    state["sessions"] = {}
                    state_path.write_text(json.dumps(state))
                    destination.mkdir(parents=True)
                    (destination / "pinned-commit.raw").write_bytes(
                        subprocess.check_output(
                            ["git", "cat-file", "commit", manifest["commit"]], cwd=REPO
                        )
                    )
                    shutil.copytree(root / "job-output", destination / "output")
                    (destination / "provider-sessions.json").write_text(
                        json.dumps(state["sessions"])
                    )
                    (destination / "supervisor.log").write_text(
                        stdout.getvalue() + stderr.getvalue()
                    )
                    provenance.update(
                        receipt_sha256=None,
                        receipt_error=None,
                        failure_class=None,
                        broker_exception="host os._exit(91) before finally",
                        scratch_removed_by_parent=str(root),
                    )
                    (destination / "provenance.json").write_text(
                        json.dumps(provenance, indent=2) + "\n"
                    )
                    os._exit(91)

            if scenario != "calibration-reserve":
                provenance["injections"].append(
                    {
                        "clean-completion": (
                            "pinned runtime CPU child running real train.run via existing "
                            "production fixture seams; offline path/release I/O seams only"
                        ),
                        "wall-cap": (
                            "at ready, virtual monotonic advances to 200 seconds; actual loop "
                            "wall guard fires"
                        ),
                        "runaway-guard": (
                            "first post-ready provider sessions read advances virtual clock "
                            "to 600s; AccountTracker.check priced cap fires after loop wall "
                            "check"
                        ),
                        "helper-deadline": (
                            "at ready, expire CLI.deadline and invoke actual " "remaining_seconds"
                        ),
                        "cli-deadline": "at ready, expire CLI.deadline and invoke actual CLI.call",
                        "lease-renewal-timeout": (
                            "at ready, invoke actual renew_runtime_lease(force=False); "
                            "repeated subprocess.TimeoutExpired only at renew_lease kernel "
                            "boundary"
                        ),
                        "rpc-exec-poll-timeout": (
                            "at ready, repeated subprocess.TimeoutExpired at telemetry "
                            "read_state RPC boundary"
                        ),
                        "child-nonzero": (
                            "existing fake runtime child terminal changed to failed/23, no "
                            "mechanical reason"
                        ),
                        "provider-loss-no-receipt": (
                            "offline fake provider sessions removed, host process "
                            "os._exit(91) before Broker finally"
                        ),
                        "runtime-timeout": (
                            "pinned runtime CPU child; validation path/release I/O replaced "
                            "for offline private temp root"
                        ),
                        "runtime-lease-expiry": (
                            "pinned runtime CPU child; short real lease; validation "
                            "path/release I/O replaced offline"
                        ),
                    }[scenario]
                )
            stdout = io.StringIO()
            stderr = io.StringIO()
            error = None
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    supervisor.Broker(root / "lanes", binary, observer=observer).run(manifest)
            except Exception as exc:
                error = repr(exc)
            destination.mkdir(parents=True)
            (destination / "pinned-commit.raw").write_bytes(
                subprocess.check_output(["git", "cat-file", "commit", manifest["commit"]], cwd=REPO)
            )
            shutil.copytree(root / "job-output", destination / "output")
            (destination / "supervisor.log").write_text(stdout.getvalue() + stderr.getvalue())
            receipt_path = destination / "output/receipt.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            provenance.update(
                broker_exception=error,
                receipt_sha256=_sha(receipt_path) if receipt else None,
                receipt_error=receipt.get("error") if receipt else None,
                failure_class=receipt.get("failure_class") if receipt else None,
            )
            provenance["files"] = {
                str(p.relative_to(destination)): {"size": p.stat().st_size, "sha256": _sha(p)}
                for p in sorted(destination.rglob("*"))
                if p.is_file() and p.name != "provenance.json"
            }
            (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS)
    args = parser.parse_args()
    selected = [args.scenario] if args.scenario else SCENARIOS
    for scenario in selected:
        result = capture_one(args.destination / scenario, scenario)
        print(json.dumps({k: result[k] for k in ("scenario", "receipt_error", "failure_class")}))


if __name__ == "__main__":
    main()
