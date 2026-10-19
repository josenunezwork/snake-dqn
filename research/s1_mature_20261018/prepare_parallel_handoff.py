"""Record an operator-observed quiescent handoff; never allocate or dispatch.

The explicit exit attestation is required because a settled job or empty account
cannot establish that a legacy serial dispatcher has exited. Account reads use
the existing supervisor wrapper. No running job or existing evidence is changed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from research.s1_mature_20261018 import admission, ledger, run


def prepare(args: argparse.Namespace) -> Path:
    """Bind one fresh zero-assignment boundary after an observed dispatcher exit."""
    if not args.legacy_dispatcher_exited:
        raise admission.AdmissionRefused(
            "operator must independently verify legacy dispatcher exit"
        )
    for target in (args.out, args.zero_snapshot):
        if target.exists():
            raise admission.AdmissionRefused(f"create-only handoff output already exists: {target}")
    plan = run._verify_plan(args.plan.resolve(), args.repo.resolve())
    lanes = plan.get("run_rules", {}).get("lanes", 1)
    amendment = run._parallel_amendment(plan, lanes)
    with run._dispatch_lock(args.ledger.resolve()):
        worker = ledger.Ledger(args.ledger.resolve())
        if worker.preview_parallel(amendment, lanes=lanes)["outstanding_jobs"]:
            raise admission.AdmissionRefused("handoff requires all reservations to be reconciled")
        snapshot = run._snapshot(
            args.zero_snapshot.resolve(),
            colab=args.colab,
            account_config=args.account_config,
            include_active_records=True,
        )
        worker._snapshot(snapshot, require_zero=True)
        account = run._read(snapshot)
        if account.get("sessions") != {} or account.get("active_records") != []:
            raise admission.AdmissionRefused(
                "handoff requires no active broker records or sessions"
            )
        value = {
            "schema": "s1-parallel-handoff/v1",
            "status": "QUIESCENT",
            "legacy_dispatcher_exited": True,
            "captured_at": time.time(),
        }
        for key, target in (
            ("plan", args.plan),
            ("previous_plan", args.previous_plan),
            ("ledger", args.ledger),
            ("zero_snapshot", snapshot),
        ):
            value[key] = {"path": str(target.resolve()), "sha256": admission.sha256(target)}
        admission.write_once(args.out.resolve(), value)
    return args.out.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=admission.ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--colab", type=Path, required=True)
    parser.add_argument("--account-config", type=Path, required=True)
    parser.add_argument("--zero-snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-dispatcher-exited", action="store_true", required=True)
    print(prepare(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
