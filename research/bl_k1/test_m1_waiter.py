"""Exercise the owner-authorized waiter without touching host flags or dispatch."""

import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("m1_attempt06_rearm.sh")


def test_help_and_missing_proof() -> None:
    result = subprocess.run([str(SCRIPT), "--help"], check=True, capture_output=True, text=True)
    assert "--proof FILE" in result.stdout
    assert "--confirmation" not in result.stdout
    result = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--proof" in result.stderr


def test_busy_classifier_rejects_only_active_codex_commands() -> None:
    source = SCRIPT.read_text()
    classifier = source.split("/usr/bin/awk '\n", 1)[1].split("\n  '", 1)[0]
    for line, busy in [
        ("/usr/local/bin/codex exec -C /some/repo task", True),
        ("bash /Users/josenunez/.local/bin/codex-handoff.sh -C repo", True),
        ("codex-handoff task", True),
        ("codex exec", True),
        ("/usr/local/bin/codex --model gpt-6.1-sol exec -C /some/repo task", True),
        ("codex --yolo e task", True),
        ("python3 -m research.colab_cli.supervisor run manifest", False),
        ("/System/some/daemon", False),
        ("/usr/bin/python3 unrelated.py", False),
        ("bash /repo/research/bl_k1/m1_attempt06_rearm.sh --proof proof", False),
        ("codex app-server", False),
        ("unknown-codex exec", False),
    ]:
        result = subprocess.run(
            ["/usr/bin/awk", classifier], input=line + "\n", text=True, capture_output=True
        )
        assert result.returncode == (0 if busy else 1), (line, result.stderr)


def test_waiter_dispatches_once_then_clears_only_its_flag(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    runner = worktree / "research/bl_k1/run_m1.sh"
    runner.parent.mkdir(parents=True)
    flag = tmp_path / "mac-timing-armed"
    calls = tmp_path / "calls"
    runner.write_text(
        f'#!/bin/bash\n[[ -f "{flag}" ]] || exit 90\nprintf "%s\\n" "$*" >> "{calls}"\nexit 7\n'
    )
    runner.chmod(0o755)
    proof, ledger = tmp_path / "proof", tmp_path / "ledger"
    proof.write_text("{}")
    ledger.write_text("{}")
    # Stub every host-dependent input. Real waiter control flow and runner remain.
    source = SCRIPT.read_text()
    source = source.replace('FLAG="$OPS/mac-timing-armed"', f'FLAG="{flag}"')
    source = source.replace('CHAIN="$EXP/bl-m1-chain.log"', f'CHAIN="{tmp_path / "chain"}"')
    source = source.replace(
        "/bin/ps -ww -Ao args=", "printf '%s\\n' 'python3 -m research.colab_cli.supervisor'"
    )
    source = source.replace("/usr/bin/pmset -g batt", "printf '%s\\n' 'AC Power'")
    source = source.replace(
        "/usr/sbin/ioreg -r -k AppleClamshellState", "printf '%s\\n' '\"AppleClamshellState\" = No'"
    )
    waiter = tmp_path / "waiter.sh"
    waiter.write_text(source)
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
        str(tmp_path / "attempt"),
    ]
    result = subprocess.run(args + ["--check-config"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "CHECK_ONLY"
    assert not flag.exists() and not calls.exists()
    # A failed/empty ps must wait, never interpret absence of data as admission.
    for listing in ("false", "printf ''"):
        failed_screen = source.replace(
            "printf '%s\\n' 'python3 -m research.colab_cli.supervisor'", listing
        ).replace("sleep 10", "exit 3")
        waiter.write_text(failed_screen)
        result = subprocess.run(args, capture_output=True, text=True)
        assert result.returncode == 3, result.stderr
        assert not flag.exists() and not calls.exists()
    waiter.write_text(source)
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    assert len(calls.read_text().splitlines()) == 1
    assert not flag.exists()
    # Failed-before-start runner is never retried; its create-only log blocks replay.
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2
    assert len(calls.read_text().splitlines()) == 1
    assert not flag.exists()
