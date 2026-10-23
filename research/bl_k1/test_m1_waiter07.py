"""Exercise the attempt-07 waiter without touching host flags or dispatch."""

import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("m1_attempt07_rearm.sh")


def classifier() -> str:
    source = SCRIPT.read_text()
    return source.split("/usr/bin/awk '\n", 1)[1].split("\n  '", 1)[0]


def test_busy_classifier_requires_new_quiet_conditions() -> None:
    for line, busy in [
        ("0.0 /usr/local/bin/codex exec -C /some/repo task", True),
        ("0.0 bash /Users/josenunez/.local/bin/codex-handoff.sh -C repo", True),
        ("0.0 python3 -m research.colab_cli.supervisor run manifest", True),
        ("0.0 python3 -m research.colab_cli run manifest", True),
        ("0.0 /Users/josenunez/.local/bin/colab run manifest.json", True),
        ("49.9 /Applications/Claude.app/Contents/MacOS/Claude", False),
        ("50.0 Claude Helper (Renderer)", True),
        ("55.9 /Applications/Claude.app/Contents/MacOS/Claude", True),
        ("75.0 /System/some/daemon", False),
        ("75.0 python3 unrelated.py", False),
        ("0.0 bash /repo/research/bl_k1/m1_attempt07_rearm.sh --proof proof", False),
        ("0.0 codex app-server", False),
        ("0.0 unknown-codex exec", False),
        ("not-a-cpu /usr/bin/python3 idle.py", True),
        ("0.0", True),
    ]:
        result = subprocess.run(
            ["/usr/bin/awk", classifier()], input=line + "\n", text=True, capture_output=True
        )
        assert result.returncode == (0 if busy else 1), (line, result.stderr)


def test_check_config_has_no_host_or_dispatch_side_effects(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    runner = worktree / "research/bl_k1/run_m1.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/bash\nexit 90\n")
    runner.chmod(0o755)
    proof, ledger = tmp_path / "proof", tmp_path / "ledger"
    proof.write_text("{}")
    ledger.write_text("{}")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--worktree",
            str(worktree),
            "--proof",
            str(proof),
            "--ledger",
            str(ledger),
            "--out",
            str(tmp_path / "attempt"),
            "--check-config",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "CHECK_ONLY", "timed_cells_run": False}
    assert not list(tmp_path.glob("attempt*"))


def test_waiter_starts_once_and_clears_only_its_flag(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    runner = worktree / "research/bl_k1/run_m1.sh"
    runner.parent.mkdir(parents=True)
    flag, calls, chain = tmp_path / "mac-timing-armed", tmp_path / "calls", tmp_path / "chain"
    runner.write_text(
        f'#!/bin/bash\n[[ -f "{flag}" ]] || exit 90\nprintf "%s\\n" "$*" >> "{calls}"\nexit 7\n'
    )
    runner.chmod(0o755)
    proof, ledger = tmp_path / "proof", tmp_path / "ledger"
    proof.write_text("{}")
    ledger.write_text("{}")
    source = SCRIPT.read_text()
    source = source.replace('FLAG="$OPS/mac-timing-armed"', f'FLAG="{flag}"')
    source = source.replace('CHAIN="$EXP/bl-m1-chain.log"', f'CHAIN="{chain}"')
    source = source.replace(
        "/bin/ps -ww -Ao %cpu=,args=", "printf '%s\\n' '0.0 /usr/bin/python3 idle.py'"
    )
    source = source.replace("/usr/bin/pmset -g batt", "printf '%s\\n' 'AC Power'")
    source = source.replace(
        "/usr/sbin/ioreg -r -k AppleClamshellState", "printf '%s\\n' '\"AppleClamshellState\" = No'"
    )
    waiter = tmp_path / "waiter.sh"
    waiter.write_text(source)
    out = tmp_path / "attempt"
    args = [
        "bash",
        str(waiter),
        "--worktree",
        str(worktree),
        "--proof",
        str(proof),
        "--ledger",
        str(ledger),
        "--out",
        str(out),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    assert len(calls.read_text().splitlines()) == 1
    assert not flag.exists()
    assert (tmp_path / "attempt.started").exists()
    assert chain.read_text().endswith("EXIT=7 m1-07-once\n")
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2
    assert "already started" in result.stderr
    assert len(calls.read_text().splitlines()) == 1


def test_admission_failures_are_fail_closed_without_starting(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    runner = worktree / "research/bl_k1/run_m1.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/bash\nexit 90\n")
    runner.chmod(0o755)
    proof, ledger = tmp_path / "proof", tmp_path / "ledger"
    proof.write_text("{}")
    ledger.write_text("{}")
    flag = tmp_path / "mac-timing-armed"
    out = tmp_path / "attempt"
    base = SCRIPT.read_text()
    base = base.replace('FLAG="$OPS/mac-timing-armed"', f'FLAG="{flag}"')
    base = base.replace("sleep 10", "exit 3")
    base = base.replace("/usr/bin/pmset -g batt", "printf '%s\\n' 'AC Power'")
    base = base.replace(
        "/usr/sbin/ioreg -r -k AppleClamshellState", "printf '%s\\n' '\"AppleClamshellState\" = No'"
    )
    args = [
        "bash",
        str(tmp_path / "waiter.sh"),
        "--worktree",
        str(worktree),
        "--proof",
        str(proof),
        "--ledger",
        str(ledger),
        "--out",
        str(out),
    ]
    for process_command in ("false", "printf ''"):
        waiter = Path(args[1])
        waiter.write_text(base.replace("/bin/ps -ww -Ao %cpu=,args=", process_command))
        result = subprocess.run(args, capture_output=True, text=True)
        assert result.returncode == 3, result.stderr
        assert not flag.exists() and not (tmp_path / "attempt.started").exists()

    waiter.write_text(base.replace("printf '%s\\n' 'AC Power'", "printf '%s\\n' 'Battery Power'"))
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert not flag.exists() and not (tmp_path / "attempt.started").exists()

    waiter.write_text(
        base.replace(
            "printf '%s\\n' '\"AppleClamshellState\" = No'",
            "printf '%s\\n' '\"AppleClamshellState\" = Yes'",
        )
    )
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert not flag.exists() and not (tmp_path / "attempt.started").exists()


def test_arm_race_refuses_before_start_and_preserves_existing_flag(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    runner = worktree / "research/bl_k1/run_m1.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/bash\nexit 90\n")
    runner.chmod(0o755)
    proof, ledger = tmp_path / "proof", tmp_path / "ledger"
    proof.write_text("{}")
    ledger.write_text("{}")
    flag = tmp_path / "mac-timing-armed"
    out = tmp_path / "attempt"
    source = SCRIPT.read_text()
    source = source.replace('FLAG="$OPS/mac-timing-armed"', f'FLAG="{flag}"')
    source = source.replace(
        "/bin/ps -ww -Ao %cpu=,args=", "printf '%s\\n' '0.0 /usr/bin/python3 idle.py'"
    )
    source = source.replace("/usr/bin/pmset -g batt", "printf '%s\\n' 'AC Power'")
    source = source.replace(
        "/usr/sbin/ioreg -r -k AppleClamshellState", "printf '%s\\n' '\"AppleClamshellState\" = No'"
    )
    waiter = tmp_path / "waiter.sh"
    args = [
        "bash",
        str(waiter),
        "--worktree",
        str(worktree),
        "--proof",
        str(proof),
        "--ledger",
        str(ledger),
        "--out",
        str(out),
    ]
    # The post-arm recheck models a process arriving in the admission/arm gap.
    waiter.write_text(source.replace("if busy || ! power_and_lid_ok; then", "if true; then", 1))
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2
    assert not flag.exists() and not (tmp_path / "attempt.started").exists()

    flag.write_text("another-operation\n")
    waiter.write_text(source)
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2
    assert flag.read_text() == "another-operation\n"
    assert not (tmp_path / "attempt.started").exists()
