"""Prepare (never run) the single joint Look R-A command for ``funding.main``.

Registration R5/R6.3: Look R-A reads the immutable D256 and D1024 fused
checkpoints jointly, once, on the Mac, after D1024 is frozen and after the
natural-archive census is frozen and hashed. With no published F2 disposition
the F2-strikes branch applies, so no ``--f2-receipt`` is passed.

This preparer verifies, without deserializing any checkpoint or opening any
table row, that every phase is published (local == origin/main) and that the
checkpoint bytes are the published fit units. It derives one authoritative
claim path from the frozen series/reserve, seals
all invocation and input identities, and binds the earliest D4096 start and F2
publication commit/server push event. Only ``funding.main`` atomically claims the look, after
rechecking the packet and timing. Preparation never opens held-out outcomes.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f4_archive_census as census_tool
from research.bl_residual import f4_d256_package as d256
from research.bl_residual import f4_d1024_package as d1024
from research.bl_residual import ra_guard
from research.bl_residual.data import DOSES
from research.bl_residual.protocol import read_ledger

PACKET_SCHEMA = "bl-residual-r-a-command-packet/v3"
PYTHON = "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python"
THREAD_ENV = [
    "OMP_NUM_THREADS=1",
    "OMP_THREAD_LIMIT=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "NUMBA_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "SNAKE_DQN_DEVICE=cpu",
]


def _published_unit(publication: str, inputs: Path, checkpoint: Path, dose: int) -> str:
    """The checkpoint bytes must be the published, verified fit-phase unit."""
    d1024.require_publication(d1024.ROOT / publication, publication, None)
    control = json.loads((inputs / "control.json").read_text())
    units = read_ledger(d1024.ROOT / publication, control["namespace"], control["allocation"])
    index = DOSES.index(dose)
    sha = verify_export(checkpoint)
    if (
        set(units) != {index}
        or units[index]["sha256"] != sha
        or units[index]["path"] != str(checkpoint)
        or units[index]["size"] != checkpoint.stat().st_size
        or control.get("doses") != [dose]
    ):
        raise ValueError(f"HOLD: D{dose} checkpoint is not its published fit unit")
    return sha


def identity(
    dose: int, checkpoint_sha: str, control: Path, allocation: Path, receipt: Path
) -> dict[str, Any]:
    """The six ``heldout.preflight_look`` fields, from bytes and sealed metadata."""
    verify_export(control)
    verify_export(allocation)
    return {
        "dose": dose,
        "checkpoint_sha256": checkpoint_sha,
        "control_sha256": digest(control),
        "allocation_sha256": digest(allocation),
        "archive_receipt_sha256": digest(receipt),
        "registration_sha256": digest(d1024.ROOT / "research/bl_residual/F4_REGISTRATION.md"),
    }


LOCK_PATTERN = "F4_R_A_*.lock.json"


def lock_search_roots() -> list[Path]:
    """Every F4 artifact tree where funding.main could have minted an R-A lock."""
    return [d1024.F4]


def existing_ra_locks(*roots: Path) -> list[str]:
    """Any R-A lock for F4, not only the canonical path of the current inputs.

    Preserve historical v1 locks as spent looks too. New v2 claims additionally
    have a fixed authoritative series/reserve path enforced in funding.main.
    """
    found: set[str] = set()
    for root in roots:
        if root.is_dir():
            found.update(str(path) for path in root.rglob(LOCK_PATTERN))
    return sorted(found)


def command(packet_path: Path, source: Path, pin: str) -> str:
    """Return the single registered R-A dispatch line; never runs it.

    Resolves only absolute paths (independent of the current directory) and
    refuses if the packet is foreign, any F4 R-A lock exists, or output exists.
    """
    packet_path, source = packet_path.resolve(), source.resolve()
    verify_export(packet_path)
    packet = json.loads(packet_path.read_text())
    if (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("status") != "PREPARED_NOT_RUN"
        or packet.get("source_pin") != pin
        or not (source / "tools/ops/dispatch.sh").is_file()
    ):
        raise ValueError("HOLD: packet is not the prepared R-A command at RA_PIN")
    if packet.get("source_identity") != ra_guard.source_identity(pin):
        raise ValueError("HOLD: R-A source identity differs from packet")
    lock_root = Path(packet["lock"]["lock_root"])
    existing = existing_ra_locks(lock_root, *lock_search_roots())
    if existing or Path(packet["lock_path"]).exists():
        raise ValueError(f"HOLD: R-A already opened (once-only look): {existing}")
    if Path(packet["funding_output"]).exists():
        raise ValueError("HOLD: R-A funding output already exists (once-only look)")
    return (
        "# Look R-A (registered, once-only). Operator-owned; requires a fresh ops Mac job ID.\n"
        + shlex.join([str(source / "tools/ops/dispatch.sh"), "--mac", "--repo", str(source)])
        + ' "${R_A_JOB_ID:?fresh ops-registered Mac accounting job ID}" '
        + shlex.quote(pin)
        + " -- "
        + packet["shell"]
    )


def prepare(
    *,
    d256_pack: Path,
    d1024_pack: Path,
    census_dir: Path,
    funding_out: Path,
    packet_out: Path,
    pin: str,
    f2_receipt: Path | None = None,
    d4096_start_receipt: Path | None = None,
) -> dict[str, Any]:
    d256.require_window_closed()
    source = ra_guard.source_identity(pin)
    d256_pack, d1024_pack, census_dir = (
        path.resolve() for path in (d256_pack, d1024_pack, census_dir)
    )
    funding_out, packet_out = funding_out.resolve(), packet_out.resolve()
    f2_receipt = f2_receipt.resolve() if f2_receipt is not None else None
    d4096_start_receipt = d4096_start_receipt.resolve() if d4096_start_receipt is not None else None
    if packet_out.exists() or funding_out.exists():
        raise FileExistsError("R-A packet and funding output are create-only")
    d256_inputs, d1024_inputs = d256_pack / "inputs", d1024_pack / "inputs"
    d256_checkpoint = d256_pack / "d256-output-l4-01/exports/fit/fit-0000.pth"
    d1024_checkpoint = (
        d1024.published_fit_output(d1024_pack) / f"exports/fit/fit-{d1024.INDEX:04d}.pth"
    )
    receipt = census_dir / census_tool.RECEIPT
    table = census_dir / census_tool.TABLE
    left_sha = _published_unit(d256.FIT_PUBLICATION_PATH, d256_inputs, d256_checkpoint, 256)
    right_sha = _published_unit(d1024.FIT_PUBLICATION_PATH, d1024_inputs, d1024_checkpoint, 1024)
    d1024.require_publication(
        d1024.ROOT / census_tool.PUBLICATION_PATH, census_tool.PUBLICATION_PATH, None
    )
    if (d1024.ROOT / census_tool.PUBLICATION_PATH).read_bytes() != receipt.read_bytes():
        raise ValueError("HOLD: local census differs from the published census")
    verify_export(receipt)
    value = json.loads(receipt.read_text())
    if (
        value.get("schema") != census_tool.SCHEMA
        or value.get("complete") is not True
        or value.get("table_sha256") != verify_export(table)
    ):
        raise ValueError("HOLD: census receipt does not bind its frozen table")
    for inputs in (d256_inputs, d1024_inputs):
        if census_tool.archive_identity(census_tool._bound_allocation(inputs))[
            "eligible_life_ids"
        ] != value.get("eligible_life_ids"):
            raise ValueError("HOLD: census eligibility differs from a fit allocation")
    if digest(d256_inputs / "judgment.json") != digest(d1024_inputs / "judgment.json"):
        raise ValueError("HOLD: D256/D1024 approvals differ")
    left = identity(
        256, left_sha, d256_inputs / "control.json", d256_inputs / "life-allocation.json", receipt
    )
    right = identity(
        1024,
        right_sha,
        d1024_inputs / "control.json",
        d1024_inputs / "life-allocation.json",
        receipt,
    )
    allocation = d256_inputs / "life-allocation.json"
    reserve = ra_guard.reserve_identity(
        d256_inputs / "control.json",
        d1024_inputs / "control.json",
        allocation,
        d1024_inputs / "life-allocation.json",
    )
    lock = ra_guard.lock_identity(reserve, left, right)
    lock_path = ra_guard.claim_path(reserve)
    existing = existing_ra_locks(allocation.parent, *lock_search_roots())
    if existing or lock_path.exists():
        raise ValueError(f"HOLD: an R-A lock already exists (once-only look): {existing}")
    if funding_out.parent.exists() and any(funding_out.parent.iterdir()):
        raise ValueError("HOLD: R-A output directory is not empty (once-only look)")
    # This verifies the result-ledger publication before opening the receipt.
    # Keep the exact returned identity in the sealed packet; funding.main repeats
    # this verification and refuses any command-line receipt that differs.
    f2_binding = ra_guard.receipt_binding(f2_receipt) if f2_receipt is not None else None
    timing = ra_guard.timing_identity(
        d4096_start_receipt, ra_guard.publication_evidence(f2_binding)
    )
    prepared_time = ra_guard.trusted_time()
    argv = [
        PYTHON,
        "-m",
        "research.bl_residual.funding",
        "--source-pin",
        pin,
        "--d256-checkpoint",
        str(d256_checkpoint),
        "--d256-table",
        str(table),
        "--d256-allocation",
        str(allocation),
        "--d256-control",
        str(d256_inputs / "control.json"),
        "--d1024-checkpoint",
        str(d1024_checkpoint),
        "--d1024-table",
        str(table),
        "--d1024-allocation",
        str(d1024_inputs / "life-allocation.json"),
        "--d1024-control",
        str(d1024_inputs / "control.json"),
        "--archive-receipt",
        str(receipt),
        "--approval",
        str(d256_inputs / "judgment.json"),
        "--b8",
        str(d256_inputs / "b8.pth"),
        "--look-lock",
        str(lock_path),
        "--command-packet",
        str(packet_out),
        "--out",
        str(funding_out),
    ]
    if f2_receipt is not None:
        argv.extend(("--f2-receipt", str(f2_receipt)))
    if d4096_start_receipt is not None:
        argv.extend(("--d4096-start-receipt", str(d4096_start_receipt)))
    invocation = dict(zip(argv[3::2], argv[4::2]))
    input_identities = {
        option: {"path": path, "sha256": digest(Path(path))}
        for option, path in invocation.items()
        if option not in ("--out", "--look-lock", "--command-packet", "--source-pin")
    }
    packet = {
        "schema": PACKET_SCHEMA,
        "status": "PREPARED_NOT_RUN",
        "source_pin": pin,
        "source_identity": source,
        "invocation": invocation,
        "input_identities": input_identities,
        "timing": timing,
        "prepared_time_observation": prepared_time,
        "lock": lock,
        "lock_path": str(lock_path),
        "funding_output": str(funding_out),
        "f2_branch": (
            "UNPUBLISHED_APPLY_STRIKES_BRANCH"
            if timing["f2_publication"] is None
            else "VERIFIED_PUBLISHED_F2_RECEIPT"
        ),
        "f2_receipt": f2_binding,
        "env": THREAD_ENV,
        "argv": argv,
        "shell": shlex.join(["/usr/bin/env", *THREAD_ENV, *argv]),
        "inputs": {
            "d256_checkpoint_sha256": left_sha,
            "d1024_checkpoint_sha256": right_sha,
            "census_sha256": digest(receipt),
            "table_sha256": value["table_sha256"],
        },
        "outcome_reporting": "none",
    }
    export_json(packet_out, packet)
    return {"packet": str(packet_out), "sha256": digest(packet_out), "lock_path": str(lock_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", type=Path, metavar="PACKET")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--d256-pack", type=Path, default=d1024.D256_PACK)
    parser.add_argument("--d1024-pack", type=Path)
    parser.add_argument("--census-dir", type=Path)
    parser.add_argument("--funding-out", type=Path)
    parser.add_argument("--packet-out", type=Path)
    parser.add_argument("--f2-receipt", type=Path)
    parser.add_argument("--d4096-start-receipt", type=Path)
    parser.add_argument("--pin", required=True)
    args = parser.parse_args()
    if args.print_command:
        if args.source is None:
            parser.error("--print-command requires --source")
        print(command(args.print_command, args.source, args.pin))
        return
    if not all((args.d1024_pack, args.census_dir, args.funding_out, args.packet_out)):
        parser.error("--d1024-pack, --census-dir, --funding-out and --packet-out required")
    print(
        json.dumps(
            prepare(
                d256_pack=args.d256_pack,
                d1024_pack=args.d1024_pack,
                census_dir=args.census_dir,
                funding_out=args.funding_out,
                packet_out=args.packet_out,
                pin=args.pin,
                f2_receipt=args.f2_receipt,
                d4096_start_receipt=args.d4096_start_receipt,
            )
        )
    )


if __name__ == "__main__":
    main()
