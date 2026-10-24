"""Operator CLI for the L4 label lane (judgment JUDGMENT_L4_FAULT_DEFINITION_20261024.md).

Subcommands (all create-only, none dispatches or allocates):
  delegate          write delegation-NN.json (tail block, <=50% share)
  prepare-session   L4 session control/ledger/manifest/DISPATCH.txt for a delegation
  import            verify a stopped L4 session and import its committed parents
  reclaim           return a delegation's uncommitted parents to the Mac
  close             Mac + L4 union close and prefix-dose ledger
  identity-prepare  step-7 quarantined 36-parent payload-identity packet
  identity-compare  hash-only per-life L4 vs Mac comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _control(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("delegate")
    p.add_argument("--delegation-dir", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, action="append", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--sizing-note", default="")

    p = sub.add_parser("prepare-session")
    p.add_argument("--delegation-dir", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--delegation-id", required=True)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--b8", type=Path, required=True)
    p.add_argument("--frp3", type=Path, required=True)
    p.add_argument("--base-minutes", type=float, default=120.0)
    p.add_argument("--broker", type=Path)

    p = sub.add_parser("import")
    p.add_argument("--delegation-dir", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--delegation-id", required=True)
    p.add_argument("--session-control", type=Path, required=True)
    p.add_argument("--session-output", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, action="append", required=True)
    p.add_argument("--union-root", type=Path, required=True)
    p.add_argument("--injection-manifest", type=Path)
    p.add_argument("--allow-transport-fault-import", action="store_true")

    p = sub.add_parser("reclaim")
    p.add_argument("--delegation-dir", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--delegation-id", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--import-record", type=Path, required=True)

    p = sub.add_parser("close")
    p.add_argument("--delegation-dir", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, action="append", required=True)
    p.add_argument("--union-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--declared-pending", type=Path, help="JSON list of pending parent ids")
    p.add_argument(
        "--funding-receipt", type=Path, help="D4096 only: independent PASS funding receipt"
    )

    p = sub.add_parser("identity-prepare")
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mac-root", type=Path, action="append", required=True)
    p.add_argument("--packet-dir", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--b8", type=Path, required=True)
    p.add_argument("--frp3", type=Path, required=True)
    p.add_argument("--base-minutes", type=float, default=120.0)
    p.add_argument("--broker", type=Path)

    p = sub.add_parser("identity-compare")
    p.add_argument("--packet-dir", type=Path, required=True)
    p.add_argument("--l4-output", type=Path, required=True)
    p.add_argument("--b8", type=Path, required=True)
    p.add_argument("--frp3", type=Path, required=True)

    args = parser.parse_args()
    from . import l4_delegation, l4_import, l4_manifests

    if args.command == "delegate":
        out = l4_delegation.write_delegation(
            args.delegation_dir,
            _control(args.control),
            args.mac_root,
            count=args.count,
            start=args.start,
            end=args.end,
            sizing_note=args.sizing_note,
        )
    elif args.command == "prepare-session":
        out = l4_manifests.prepare_session(
            args.delegation_dir,
            _control(args.control),
            args.delegation_id,
            args.session_dir,
            commit=args.commit,
            b8=args.b8,
            frp3=args.frp3,
            base_minutes=args.base_minutes,
            broker=args.broker,
        )
    elif args.command == "import":
        out = l4_import.import_session(
            args.delegation_dir,
            _control(args.control),
            args.delegation_id,
            args.session_control,
            args.session_output,
            args.receipt,
            args.mac_root,
            args.union_root,
            injection_manifest=(
                json.loads(args.injection_manifest.read_text()) if args.injection_manifest else None
            ),
            allow_transport_fault_import=args.allow_transport_fault_import,
        )
    elif args.command == "reclaim":
        out = l4_delegation.write_reclaim(
            args.delegation_dir,
            _control(args.control),
            args.delegation_id,
            args.receipt,
            args.import_record,
        )
    elif args.command == "close":
        result = l4_import.union_close(
            _control(args.control),
            args.mac_root,
            args.delegation_dir,
            args.union_root,
            args.out,
            declared_pending=(
                json.loads(args.declared_pending.read_text()) if args.declared_pending else None
            ),
            funding_receipt=args.funding_receipt,
        )
        print(json.dumps({k: result[k] for k in ("verdict", "prefix_length", "allocated")}))
        if result["verdict"] != "CLEAN":
            raise SystemExit(2)
        return
    elif args.command == "identity-prepare":
        out = l4_manifests.prepare_identity_packet(
            _control(args.control),
            args.mac_root,
            args.packet_dir,
            commit=args.commit,
            b8=args.b8,
            frp3=args.frp3,
            base_minutes=args.base_minutes,
            broker=args.broker,
        )
    else:
        result = l4_manifests.compare_identity(args.packet_dir, args.l4_output, args.b8, args.frp3)
        print(json.dumps({k: result[k] for k in ("verdict", "identical")}))
        if result["verdict"] != "IDENTICAL_ALL":
            raise SystemExit(2)
        return
    print(out)


if __name__ == "__main__":
    main()
