#!/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"""Host-only admission front door; delegate unchanged CLI through the broker.

Only `new` receives an additional refusal gate. All watchdog/stop/export/read
operations keep the broker's existing executable and argument boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import small_bench_operator as op  # noqa: E402


def guard(config_path: Path, argv: list[str]) -> None:
    """Refuse stale/changed exposure immediately before the provider request."""
    if "new" not in argv:
        return
    config = op.configuration(config_path)
    admission = op.current_admission(config)
    now = datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(admission["capture"]["started_utc"])).total_seconds()
    if not 0 <= age <= 600:
        raise ValueError("stale capture at allocation request; refresh same reservation")
    rows = op.register.read(Path(config["register"]))
    if rows[-1]["row_hash"] != admission["register_row_hash"]:
        raise ValueError("financial revision changed before allocation")
    if admission["config"] != op.binding(config_path):
        raise ValueError("wrong allocation config")
    session = os.environ.get("SNAKE_COLAB_SESSION", "")
    account = Path(config["account_root"])
    pending = account / f"pending-{session}.json"
    if not session or not pending.is_file():
        raise ValueError("allocation is not owned by the qualified supervisor")
    own = json.loads(pending.read_text())
    original = json.loads(op.bound(config["manifest"]).read_text())
    retry = dict(
        original,
        output_dir=original["output_dir"] + "-retry-1",
        job_id=original["job_id"] + "-retry-1",
    )
    if len(retry["job_id"]) > 64:
        digest = hashlib.sha256(original["job_id"].encode()).hexdigest()[:8]
        retry["job_id"] = f"{original['job_id'][:47]}-{digest}-retry-1"
    is_retry = own.get("output") == retry["output_dir"]
    if own.get("session") != session or own.get("output") not in {
        original["output_dir"],
        retry["output_dir"],
    }:
        raise ValueError("wrong pending allocation identity")
    if is_retry:
        receipt = json.loads((Path(original["output_dir"]) / "receipt.json").read_text())
        log = Path(config["reservation_lock"]).parent / "logs" / f"{original['job_id']}.log"
        if not (
            receipt.get("job_id") == original["job_id"]
            and receipt.get("status") == "INCOMPLETE"
            and receipt.get("failure_class") == "ACCOUNTING"
            and receipt.get("error")
            == "AccountingError('assignment endpoints changed during allocation')"
            and receipt.get("cleanup_error") is None
            and receipt.get("stop_verified") is True
            and re.search(r"^EXIT=\d+$", log.read_text(), re.M)
        ):
            raise ValueError("retry lacks exact proven wrapper mechanical failure")
    if set(account.glob("pending-*.json")) != {pending}:
        raise ValueError("competing pending allocation")
    # Expected ops changes are the wrapper's exact job copy and assignments-only
    # live refresh. Every other watched identity must retain the captured bytes.
    before = admission["operational_fingerprint"]
    after = op.fingerprint(config)
    after.pop(str(pending), None)
    for path in list(after):
        if path not in before:
            value = json.loads(Path(path).read_text())
            expected = retry if is_retry and value.get("job_id") == retry["job_id"] else original
            if value.get("job_id") != expected["job_id"]:
                raise ValueError("intervening assignment or liability")
            value["budget"]["other_assignments"] = original["budget"].get("other_assignments", {})
            if value != expected:
                raise ValueError("changed allocation manifest")
            after.pop(path)
    if after != before:
        raise ValueError("intervening assignment or liability")
    package = dict(
        config,
        manifest_sha256=config["manifest"]["sha256"],
        capture_utc=admission["capture"]["started_utc"],
        live_balance_units=admission["live_checks"]["balance"],
        operator=admission["row"]["package"]["operator"],
    )
    op.register.reserve(rows, package, config["reservation_id"], now)


def main() -> None:
    try:
        guard(Path(os.environ["SNAKE_SMALL_BENCH_CONFIG"]), sys.argv[1:])
        # This is the original CLI; OAuth2, config hook, and supervisor ownership
        # remain supplied by the qualified supervisor. Never allocate ourselves.
        executable = "/Users/josenunez/.local/bin/colab"
        os.execv(executable, [executable, *sys.argv[1:]])
    except (OSError, ValueError, KeyError) as error:
        print(f"REFUSE small-bench allocation: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
