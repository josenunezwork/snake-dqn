"""CPU probe and learner tests for TRAIN-01's <15 s SIGUSR1 checkpoint path."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident.test_training_learner import make_rollout
from research.sandbox_a_resident.train import (
    SIGUSR1_MARKER_NAME,
    SIGUSR1_MARKER_SCHEMA,
    TrainControl,
)
from research.sandbox_a_resident.training_learner import (
    GracefulSignalStop,
    PPOConfig,
    PPOTrainer,
)
from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic
from research.sandbox_a_resident.training_resume import build_ledger
from research.sandbox_a_resident.training_units import empty_ledger
from research.sandbox_a_resident.units import digest
from src.model.ego2s_network import Ego2sNet

ROOT = Path(__file__).resolve().parents[2]
SINGLE_THREAD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "SNAKE_DQN_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}


def _verify_completion_last(path: Path) -> dict:
    sidecar = Path(f"{path}.sha256")
    complete = Path(f"{path}.complete")
    assert path.is_file() and sidecar.is_file() and complete.is_file()
    assert sidecar.read_text().strip().split()[0] == digest(path)
    assert complete.read_text() == "complete\n"
    return json.loads(path.read_text())


def test_ppo_trainer_finishes_current_minibatch_on_sigusr1() -> None:
    torch.set_num_threads(1)
    b8 = Ego2sNet(local_channels=7, n_scalars=60)
    actor, critic = B8Actor(b8_model=b8), ClockedPopArtCritic(b8)
    trainer = PPOTrainer(actor, critic, PPOConfig(minibatch=3))
    rollout = make_rollout(actor, critic, time=2, worlds=3)
    before_actor = actor.adv.weight.detach().clone()
    before_critic = critic.output.weight.detach().clone()

    checks = 0

    def stop_after_first_minibatch() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 1

    with pytest.raises(GracefulSignalStop) as caught:
        trainer.update(rollout, should_stop=stop_after_first_minibatch)
    assert caught.value.stage == "learner-minibatch"
    assert caught.value.minibatch_completed is True
    assert caught.value.epoch == 1
    assert caught.value.minibatch == 1
    # Minibatch 1 optimizer steps completed cleanly, while update_count/actor_rows did not advance.
    assert not torch.equal(before_actor, actor.adv.weight)
    assert not torch.equal(before_critic, critic.output.weight)
    assert trainer.update_count == 0
    assert trainer.actor_rows == 0
    for param in (*actor.parameters(), *critic.parameters()):
        assert torch.isfinite(param).all()

    # If SIGUSR1 arrives on the final minibatch of the final epoch (4 epochs * 2 minibatches = 8),
    # update() completes normally.
    final_checks = 0

    def stop_on_final_minibatch() -> bool:
        nonlocal final_checks
        final_checks += 1
        return final_checks >= 8

    metrics = trainer.update(rollout, should_stop=stop_on_final_minibatch)
    assert trainer.update_count == 1
    assert trainer.actor_rows == 6
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def _write_probe_driver(script_path: Path, *, mode: str) -> None:
    """Write a subprocess driver running actual train.main with lightweight CPU workload."""
    script_path.write_text(textwrap.dedent(f"""\
            import json
            import sys
            import time
            from pathlib import Path
            from types import SimpleNamespace
            from dataclasses import dataclass

            import torch

            from research.sandbox_a_resident import train
            from research.sandbox_a_resident.test_training_runner import Actor, Critic, Task
            from research.sandbox_a_resident.training_learner import GracefulSignalStop

            torch.set_num_threads(1)
            MODE = {mode!r}

            hashes = {{
                "b8-s3.pth": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
                "frp3-s12.pth": train.FRP3_S12_SHA256,
                "champion.pth": train.CHAMPION_SHA256,
            }}
            train.digest = lambda p: hashes[Path(p).name]

            @dataclass
            class SimpleCfg:
                fixture: bool = True

            class ProbeActor(Actor):
                def __init__(self, checkpoint):
                    super().__init__()
                    self.parameter = torch.nn.Parameter(torch.ones(64))
                    self.reference = Actor()
                    self.temperature = torch.tensor(1.0)

                def export_ego2s(self, path, metadata):
                    path.write_bytes(b"probe-actor-export")

            class ProbeCritic(Critic):
                def __init__(self, reference):
                    super().__init__()

            class ProbeTrainer:
                def __init__(self, *args):
                    self.actor_rows = 0
                    self.critic_optimizer = None

                def update(self, rollout, *, diagnostics=True, should_stop=None):
                    if MODE == "mid_minibatch":
                        print("READY_FOR_SIGUSR1", flush=True)
                        deadline = time.monotonic() + 5.0
                        while time.monotonic() < deadline:
                            if should_stop is not None and should_stop():
                                raise GracefulSignalStop(
                                    "ppo_update",
                                    minibatch_completed=True,
                                    epoch=0,
                                    minibatch=0,
                                )
                            time.sleep(0.02)
                    self.actor_rows += int(rollout.actions.numel())
                    return {{"actor_rows": self.actor_rows}}

                def state_dict(self):
                    return {{"actor_rows": self.actor_rows}}

                def load_state_dict(self, state):
                    self.actor_rows = state["actor_rows"]

            class ProbeTask(Task):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.sim = SimpleNamespace(cfg=SimpleCfg())
                    self.rosters = [SimpleNamespace(identity="probe")]

                def attach_opponents(self, opponents):
                    pass

                def proxy_factory(self, opponents, allocations):
                    return lambda *args: None

                def state_dict(self):
                    return {{"age": self.age.clone(), "tick": self.tick}}

                def load_state_dict(self, state):
                    self.age, self.tick = state["age"].clone(), state["tick"]

            class ProbeCompile:
                def __init__(self, actor, task, **kwargs):
                    self.task, self.report = task, {{}}

                def runtime_self_check(self, rows):
                    return {{"rows": rows}}

                def observe(self):
                    return self.task.observe()

                def actor_forward(self, local, glob, scalars):
                    return Actor()(local, glob, scalars)

            proxy_calls = 0

            def probe_evaluate(actor, factory, cfg, on_progress=None, *, should_stop=None):
                global proxy_calls
                proxy_calls += 1
                if MODE == "during_boundary_proxy" and proxy_calls >= 2:
                    print("READY_FOR_SIGUSR1", flush=True)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        if should_stop is not None and should_stop():
                            raise GracefulSignalStop("proxy_eval", minibatch_completed=False)
                        time.sleep(0.02)
                from research.sandbox_a_resident.training_proxy import MixMetrics
                return {{
                    mix: MixMetrics(
                        mix=mix,
                        mass_mean=500.0,
                        survival_mean=1.0,
                        alive_fraction=1.0,
                        boost_rate_under_700=0.0,
                        mass=[500.0, 500.0, 500.0],
                        survival=[1.0, 1.0, 1.0],
                    )
                    for mix in ("scripted", "frozen", "mixed")
                }}

            def probe_prepare(*args, **kwargs):
                return SimpleNamespace(
                    state=SimpleNamespace(
                        state_dict=lambda: {{
                            "stage": "ready",
                            "temperature": 1.0,
                            "quality": {{"mse": 1.0}},
                        }}
                    ),
                    critic_optimizer=None,
                )

            train.B8Actor = ProbeActor
            train.ClockedPopArtCritic = ProbeCritic
            train.PPOTrainer = ProbeTrainer
            train.ResidentTrainingTask = ProbeTask
            train.DeviceRosterRuntime = lambda *a, **k: None
            train.TrainingCompile = ProbeCompile
            train._load_vector61 = lambda *a: None
            train.cpu_distribution_factory = lambda *a, **k: None
            train.run_distribution_screen = lambda **k: {{"material_gaps": []}}
            train.prepare = probe_prepare
            train.evaluate = probe_evaluate
            raise SystemExit(train.main())
            """))


def _probe_control_dict() -> dict:
    return asdict(
        TrainControl(
            study="sandbox-a-train01",
            source_commit="a" * 40,
            allocation_index=0,
            outer_iterations=2,
            worlds=3,
            rollout_horizon=2,
            snapshot_interval=1,
            proxy_interval=1,
            proxy_worlds_per_mix=3,
            proxy_horizon=2,
            test_only=True,
            compile_mode="none",
        )
    )


def _wait_ready_and_signal(
    proc: subprocess.Popen[str], *, use_pgid: bool = False
) -> tuple[str, float]:
    assert proc.stdout is not None
    deadline = time.monotonic() + 15.0
    lines: list[str] = []
    ready = False
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "READY_FOR_SIGUSR1" in line:
            ready = True
            break
    assert ready, f"probe never reached READY_FOR_SIGUSR1; output={''.join(lines)!r}"
    signaled_at = time.monotonic()
    if use_pgid:
        os.killpg(proc.pid, signal.SIGUSR1)
    else:
        proc.send_signal(signal.SIGUSR1)
    rest, _ = proc.communicate(timeout=15.0)
    elapsed = time.monotonic() - signaled_at
    lines.append(rest)
    return "".join(lines), elapsed


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX SIGUSR1 required")
def test_cpu_probe_train_main_sigusr1_mid_minibatch_writes_marker_and_exits_0(
    tmp_path: Path,
) -> None:
    work = tmp_path.resolve()
    inputs = work / "inputs"
    inputs.mkdir()
    for name in ("b8-s3.pth", "frp3-s12.pth", "champion.pth"):
        (inputs / name).write_bytes(b"stub")
    control_path = work / "control.json"
    control_payload = _probe_control_dict()
    control_path.write_text(json.dumps(control_payload))
    driver = work / "probe_driver.py"
    _write_probe_driver(driver, mode="mid_minibatch")
    output = work / "output"

    env = {**os.environ, **SINGLE_THREAD_ENV, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [
            "nice",
            "-n",
            "15",
            sys.executable,
            str(driver),
            "--checkpoint",
            str(inputs / "b8-s3.pth"),
            "--frp3",
            str(inputs / "frp3-s12.pth"),
            "--champion",
            str(inputs / "champion.pth"),
            "--control",
            str(control_path),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, elapsed = _wait_ready_and_signal(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 0, stdout
    assert elapsed < 15.0
    assert "STOP_SANDBOX_A_TRAIN_SIGUSR1" in stdout
    marker = _verify_completion_last(output / SIGUSR1_MARKER_NAME)
    assert marker["schema"] == SIGUSR1_MARKER_SCHEMA
    assert marker["action"] == "last_known_good_marker"
    assert marker["minibatch_completed"] is True
    assert marker["last_known_good_iteration"] == 0
    assert marker["grace_elapsed_seconds"] < 15.0
    assert not (output / "summary.json").exists()
    assert not (output / "iteration-0001.pt").exists()
    _, _, latest = build_ledger(output, control_payload)
    assert latest == 0


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX SIGUSR1 required")
@pytest.mark.parametrize(
    ("max_bytes", "expected_action", "expected_latest"),
    [
        (256 * 1024 * 1024, "full_snapshot_published", 1),
        (16, "last_known_good_marker", 0),
    ],
)
def test_cpu_probe_train_main_sigusr1_at_boundary_saves_or_falls_back_by_15s_byte_budget(
    tmp_path: Path, max_bytes: int, expected_action: str, expected_latest: int
) -> None:
    work = tmp_path.resolve()
    inputs = work / "inputs"
    inputs.mkdir()
    for name in ("b8-s3.pth", "frp3-s12.pth", "champion.pth"):
        (inputs / name).write_bytes(b"stub")
    control_path = work / "control.json"
    control_payload = _probe_control_dict()
    control_path.write_text(json.dumps(control_payload))
    driver = work / "probe_driver.py"
    _write_probe_driver(driver, mode="during_boundary_proxy")
    output = work / "output"

    env = {**os.environ, **SINGLE_THREAD_ENV, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [
            "nice",
            "-n",
            "15",
            sys.executable,
            str(driver),
            "--checkpoint",
            str(inputs / "b8-s3.pth"),
            "--frp3",
            str(inputs / "frp3-s12.pth"),
            "--champion",
            str(inputs / "champion.pth"),
            "--control",
            str(control_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--sigusr1-max-snapshot-bytes",
            str(max_bytes),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, elapsed = _wait_ready_and_signal(proc)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert proc.returncode == 0, stdout
    assert elapsed < 15.0
    marker = _verify_completion_last(output / SIGUSR1_MARKER_NAME)
    assert marker["action"] == expected_action
    assert marker["last_known_good_iteration"] == expected_latest
    assert not (output / "summary.json").exists()
    _, _, latest = build_ledger(output, control_payload)
    assert latest == expected_latest
    if expected_action == "full_snapshot_published":
        info = _verify_completion_last(output / "iteration-0001.json")
        assert info["metadata"]["sigusr1_boundary_save"] is True
        assert info["metadata"]["deferred_proxy"] == "regular"
    else:
        assert marker["reason"] == "snapshot_exceeds_15s_byte_budget"
        assert not (output / "iteration-0001.pt").exists()


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX SIGUSR1 required")
def test_train_job_sh_forwards_process_group_sigusr1_and_exits_cleanly(tmp_path: Path) -> None:
    work = tmp_path.resolve()
    colab_root = work / "colab_root"
    colab_root.mkdir()
    input_dir = work / "inputs"
    input_dir.mkdir()
    for name in ("b8-s3.pth", "frp3-s12.pth", "champion.pth"):
        (input_dir / name).write_bytes(b"stub")
    control_payload = _probe_control_dict()
    (input_dir / "verified-ledger.json").write_text(json.dumps(empty_ledger(control_payload)))

    driver = work / "probe_driver.py"
    _write_probe_driver(driver, mode="mid_minibatch")
    bin_dir = work / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "python"
    shim.write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "research.sandbox_a_resident.train" ]]; then
              shift 2
              exec {sys.executable} {driver} "$@" --device cpu
            fi
            exec {sys.executable} "$@"
            """))
    shim.chmod(0o755)

    control_json = json.dumps(control_payload)
    env = {
        **os.environ,
        **SINGLE_THREAD_ENV,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(ROOT),
        "COLAB_CLI_ROOT": str(colab_root),
        "COLAB_INPUT_DIR": str(input_dir),
        "SANDBOX_A_SOURCE_COMMIT": "a" * 40,
        "SANDBOX_A_CONTROL_JSON": control_json,
    }
    proc = subprocess.Popen(
        ["nice", "-n", "15", "bash", str(ROOT / "research/sandbox_a_resident/train_job.sh")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, elapsed = _wait_ready_and_signal(proc, use_pgid=True)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

    assert proc.returncode == 0, stdout
    assert elapsed < 15.0
    marker = _verify_completion_last(colab_root / "output/training" / SIGUSR1_MARKER_NAME)
    assert marker["action"] == "last_known_good_marker"
    assert marker["last_known_good_iteration"] == 0
    with pytest.raises(ProcessLookupError):
        os.killpg(proc.pid, 0)
