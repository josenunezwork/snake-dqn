"""Write explicit observed debit rates and hash synchronized Mac Q0b inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from research.q0q1_20261017.run_queue_job import sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l4-units-per-hour", required=True, type=float)
    parser.add_argument("--a100-units-per-hour", required=True, type=float)
    parser.add_argument("--cpu-units-per-hour", required=True, type=float)
    parser.add_argument(
        "--evidence", required=True, help="where/when the actual debit rates were observed"
    )
    parser.add_argument("--mac-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rates = {
        "L4": args.l4_units_per_hour,
        "A100": args.a100_units_per_hour,
        "cpu": args.cpu_units_per_hour,
    }
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates.values()):
        parser.error("all observed debit rates must be finite and positive")
    if not args.evidence.strip():
        parser.error("--evidence must identify the observed debit-rate source")
    reference = json.loads(args.mac_reference.read_text())
    if (
        reference.get("schema") != "q1-parity/v1"
        or reference.get("runtime", {}).get("system") != "Darwin"
    ):
        parser.error("--mac-reference must be a completed Q1 Mac reference")
    if args.out.exists():
        parser.error(f"refusing to overwrite observed rates: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "q0q1-observed-pricing/v1",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence": args.evidence,
        "rates": rates,
    }
    args.out.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    for path in (args.out, args.mac_reference):
        sidecar = path.with_name(path.name + ".sha256")
        with sidecar.open("x") as stream:
            stream.write(f"{sha256_file(path)}  {path.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
