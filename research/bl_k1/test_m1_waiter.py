"""Focused contract checks for the attempt-06 shell waiter."""

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("m1_attempt06_rearm.sh")


def test_waiter_has_safe_help_and_requires_owner_confirmation() -> None:
    help_result = subprocess.run(
        [str(SCRIPT), "--help"], check=True, capture_output=True, text=True
    )
    assert "--confirmation FILE" in help_result.stdout

    result = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "confirmation" in result.stderr


def test_waiter_encodes_the_registered_measured_window_and_single_dispatch() -> None:
    source = SCRIPT.read_text()
    assert "consecutive -lt 60" in source
    assert "sleep 10" in source
    assert "consecutive -eq 30" in source
    assert "5242880" in source
    assert "mac-timing-armed" in source
    assert source.index('> "$FLAG"') < source.index("consecutive=0")
    assert "$1 > 20" in source
    assert "Started: never retry" in source
    assert "--run --fixtures" in source
    assert "codex-handoff|codex exec|colab_cli|colab.*dispatch|[Pp]ython|pytest" in source


def test_embedded_classifier_preserves_space_paths_and_agent_rules(tmp_path: Path) -> None:
    """Exercise the waiter's own Perl classifier with synthetic ps snapshots."""
    source = SCRIPT.read_text()
    classifier = source.split("<<'PERL'\n", 1)[1].split("\nPERL\n", 1)[0]

    def classify(line: str) -> tuple[int, dict]:
        name = line.split()[0] if line else "empty"
        raw = tmp_path / f"snapshot-{name}.ps"
        out = tmp_path / f"sample-{name}.json"
        raw.write_text(line + "\n")
        environment = os.environ | {
            "M1_POWER": "Now drawing from 'AC Power'",
            "M1_LID": '"AppleClamshellState" = No',
        }
        result = subprocess.run(
            ["/usr/bin/perl", "-", str(raw), str(out)],
            input=classifier,
            text=True,
            capture_output=True,
            env=environment,
        )
        return result.returncode, json.loads(out.read_text())

    code, google = classify("101 1 0.0 /Applications/Google Drive.app/Contents/MacOS/Google Drive")
    assert code == 0
    assert google["quiet"] is True
    assert google["processes"][0]["comm"].endswith("Google Drive")

    code, codex = classify("102 1 0.0 /usr/local/bin/codex")
    assert code == 1
    assert codex["violations"][0]["rule"] == "agent_alive"

    code, claude_node = classify("103 1 0.0 node Claude Code")
    assert code == 0
    assert claude_node["quiet"] is True

    code, empty = classify("")
    assert code == 1
    assert empty["quiet"] is False

    code, malformed = classify("unreadable process row")
    assert code == 1
    assert malformed["quiet"] is False

    code, indexing = classify(
        "104 1 5.0 /System/Library/Frameworks/CoreSpotlight.framework/corespotlightd"
    )
    assert code == 1
    assert indexing["violations"][0]["rule"] == "daemon_ge_5"
