"""Run only the pre-registered 200k T11 MPS implementation smoke."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone

from research.t11_local30_20261017 import launch


def validate_summary(summary: dict, exit_code: int) -> None:
    """Require a complete MPS run with no halt and clean owned actor exits."""
    if exit_code != 0:
        raise ValueError(f"Smoke exited {exit_code}")
    if not 200000 <= summary.get("transitions", 0) < 201024:
        raise ValueError("Smoke did not complete the bounded transition target")
    if summary.get("halt") is not None:
        raise ValueError(f"Smoke halted: {summary['halt']}")
    if summary.get("actor_exitcodes") != [0, 0]:
        raise ValueError("Smoke actors did not exit cleanly")
    if summary.get("args", {}).get("device") != "mps":
        raise ValueError("Smoke did not use MPS")
    if summary.get("rate", 0) <= 0 or summary.get("wall_seconds", 0) <= 0:
        raise ValueError("Smoke is missing positive timing evidence")


def main() -> int:
    """Verify inputs, execute one isolated smoke, preserve its complete receipt."""
    launch.verify_inputs()
    cmd = launch.command(10, smoke=True)
    out = launch.output_path(10, smoke=True)
    log = out.with_suffix(".log")
    receipt = out.parent / "smoke_receipt.json"
    if any(path.exists() for path in (out, log, receipt)):
        raise SystemExit("Refusing existing smoke output, log or receipt")
    out.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(
            ["caffeinate", "-dims", *cmd],
            cwd=launch.REPO,
            env=launch.environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall = time.monotonic() - before
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    error = None
    try:
        launch.verify_inputs()
        validate_summary(summary, result.returncode)
        if not (out / "final.pth").is_file():
            raise ValueError("Smoke final checkpoint missing")
    except (ValueError, OSError) as exc:
        error = str(exc)
    files = [log, *sorted(out.glob("*.json")), *sorted(out.glob("*.jsonl")), *out.glob("*.pth")]
    payload = {
        "schema": "t11-mps-smoke-receipt/v1",
        "status": "PASS" if error is None else "FAIL",
        "error": error,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=launch.REPO, text=True
        ).strip(),
        "launch_binding_sha256": launch.sha256(launch.BINDING),
        "exit_code": result.returncode,
        "process_wall_seconds": wall,
        "training_summary": summary,
        "files": {str(path): launch.sha256(path) for path in files},
        "full_training_started": False,
    }
    with receipt.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt),
                "status": payload["status"],
                "rate": summary.get("rate"),
                "halt": summary.get("halt"),
                "transitions": summary.get("transitions"),
                "error": error,
            }
        )
    )
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
