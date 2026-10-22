"""Per-assignment excluded-world identity check before any selection gameplay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from research.bl_served_search import ceiling
from research.bl_served_search.exports import digest, export_json, verify_export


def semantic(value: Any) -> Any:
    """Remove timing fields and checkpoint directory presentation only."""
    if isinstance(value, list):
        return [semantic(item) for item in value]
    if isinstance(value, dict):
        result = {
            key: semantic(item)
            for key, item in value.items()
            if not key.endswith(("_seconds", "_minutes"))
        }
        if "specs" in result:
            result["specs"] = [
                [kind, Path(name).name if kind == "checkpoint" else name]
                for kind, name in result["specs"]
            ]
        return result
    return value


def verify(reference: Path, actual: Path, bank: Path, out: Path) -> dict:
    """Require all three excluded worlds to match and bind this runtime identity."""
    values = []
    fixture_identity: tuple[str, str, int] | None = None
    for path in (reference, actual):
        verify_export(path)
        value = json.loads(path.read_text())
        identity = value.get("identity", {})
        if (
            value.get("schema") != "bl-served-search-l2-fixture-shard/v1"
            or value.get("status") != "completed"
            or value.get("clean_exit") is not True
            or identity.get("source_manifest") != ceiling.source_manifest()
            or identity.get("bank_sha256") != digest(bank)
            or identity.get("fixture") is not True
            or value.get("rows_sha256") != ceiling.digest_value(value.get("rows"))
        ):
            raise ValueError("invalid source/bank-bound excluded-world parity input")
        mode = (identity.get("search_mode"), identity.get("native_mode"), identity.get("workers"))
        if mode[0] not in ("all", "exact_stop") or mode[1] not in ("stock", "replay", "kernel"):
            raise ValueError("invalid fixture execution mode")
        if not isinstance(mode[2], int) or not 1 <= mode[2] <= 12:
            raise ValueError("invalid fixture worker count")
        if fixture_identity is None:
            fixture_identity = mode
        elif mode != fixture_identity:
            raise ValueError("reference and actual fixture execution modes differ")
        expected = ceiling.validate_bank(json.loads(bank.read_text()), fixture=True)
        if [row["world_id"] for row in value["rows"]] != [row["world_id"] for row in expected]:
            raise ValueError("parity must cover the three excluded worlds in order")
        values.append(semantic(value["rows"]))
    if values[0] != values[1]:
        raise ValueError("INVALID: Mac/runtime excluded-world semantic mismatch")
    result = {
        "schema": "bl-served-search-runtime-parity/v2",
        "status": "VERIFIED",
        "bank_sha256": digest(bank),
        "b8_sha256": ceiling.B8_SHA256,
        "source_manifest": ceiling.source_manifest(),
        "mac_reference_sha256": digest(reference),
        "runtime_verification_sha256": digest(actual),
        "runtime_identity": ceiling.runtime_identity(),
        "selection_worlds": 0,
        "excluded_worlds": 3,
    }
    ceiling.validate_runtime_parity(
        result,
        bank_sha256=digest(bank),
        source=ceiling.source_manifest(),
        require_local_runtime=True,
    )
    export_json(out, result)
    return result


def _run_fixture(
    *, bank: Path, b8_path: Path, checkpoint_dir: Path, out: Path, environment: dict[str, str]
) -> None:
    """Run one create-only exact-stop fixture receipt through the real entrypoint."""
    command = [
        sys.executable,
        "-m",
        "research.bl_served_search.ceiling",
        "--bank",
        str(bank),
        "--out",
        str(out),
        "--fixture",
        "--workers",
        "2",
        "--search-mode",
        "exact_stop",
        "--native-mode",
        "stock",
        "--b8-path",
        str(b8_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    subprocess.run(command, check=True, env=environment)


def prepare_mac(*, bank: Path, b8_path: Path, checkpoint_dir: Path, out_dir: Path) -> dict:
    """Regenerate and verify the two Mac-local excluded-world parity fixtures.

    The three fixture worlds are excluded from the selection bank.  This helper
    deliberately runs the production CLI twice so its receipt proves the
    amended process-pool/native configuration, while never admitting a bank
    world or accepting an existing output directory.
    """
    if os.environ.get("COLAB_CLI_ROOT"):
        raise ValueError("--prepare-mac requires a Mac shell without COLAB_CLI_ROOT")
    if out_dir.exists():
        raise FileExistsError("--out-dir is create-only and already exists")
    out_dir.mkdir(parents=True)
    environment = os.environ.copy()
    reference = out_dir / "reference" / "shard.json"
    actual = out_dir / "verification" / "shard.json"
    receipt = out_dir / "runtime-parity.json"
    _run_fixture(
        bank=bank,
        b8_path=b8_path,
        checkpoint_dir=checkpoint_dir,
        out=reference,
        environment=environment,
    )
    _run_fixture(
        bank=bank,
        b8_path=b8_path,
        checkpoint_dir=checkpoint_dir,
        out=actual,
        environment=environment,
    )
    return verify(reference, actual, bank, receipt)


def _required(value: Path | None, positional: Path | None, name: str) -> Path:
    """Keep legacy positional verification arguments alongside named flags."""
    if value is not None and positional is not None:
        raise ValueError(f"{name} was supplied both positionally and by flag")
    resolved = value if value is not None else positional
    if resolved is None:
        raise ValueError(f"{name} is required")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_pos", nargs="?", type=Path)
    parser.add_argument("actual_pos", nargs="?", type=Path)
    parser.add_argument("bank_pos", nargs="?", type=Path)
    parser.add_argument("out_pos", nargs="?", type=Path)
    for name in ("reference", "actual", "bank", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--prepare-mac", action="store_true")
    parser.add_argument("--b8-path", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    try:
        bank = _required(args.bank, args.bank_pos, "bank")
        if args.prepare_mac:
            if args.b8_path is None or args.checkpoint_dir is None or args.out_dir is None:
                raise ValueError("--prepare-mac requires --b8-path, --checkpoint-dir and --out-dir")
            if any(
                (
                    args.reference,
                    args.actual,
                    args.out,
                    args.reference_pos,
                    args.actual_pos,
                    args.out_pos,
                )
            ):
                raise ValueError(
                    "--prepare-mac does not accept verification input/output arguments"
                )
            result = prepare_mac(
                bank=bank,
                b8_path=args.b8_path,
                checkpoint_dir=args.checkpoint_dir,
                out_dir=args.out_dir,
            )
        else:
            result = verify(
                _required(args.reference, args.reference_pos, "reference"),
                _required(args.actual, args.actual_pos, "actual"),
                bank,
                _required(args.out, args.out_pos, "out"),
            )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
