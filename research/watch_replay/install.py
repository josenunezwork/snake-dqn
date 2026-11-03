"""Install the verified subset and retain operational exclusions on the Mac."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from research.watch_replay.run import identity
from tools.ops.package_spec import publish_immutable, publish_json, unit_id, verify_immutable
from web.backend.watch_replay import CONTROLLERS, Recording, peek_header


def install(contract: dict, roots: list[Path], bank: Path) -> int:
    """Exclude individual failures; refuse foreign contracts and duplicate unit IDs."""
    bank.mkdir(parents=True, exist_ok=False)
    installed, exclusions, seen = [], [], set()
    try:
        # Refuse foreign attempt bindings before installing any payload.
        for root in roots:
            binding = root / "contract.json"
            try:
                verify_immutable(binding)
                if json.loads(binding.read_text()) != contract:
                    raise ValueError("foreign namespace/source/allocation")
            except Exception as exc:
                exclusions.append({"path": str(binding), "error": str(exc)})
                raise
        for root in roots:
            for path in sorted(root.glob("*.watch.gz")):
                try:
                    stem = path.name.removesuffix(".watch.gz")
                    if len(stem) != 6 or not stem.isdigit():
                        raise ValueError("invalid allocation filename")
                    index = int(stem)
                    if not 0 <= index < contract["count"]:
                        raise ValueError("out-of-range allocation filename")
                    if index in seen:
                        raise ValueError("duplicate allocation index")
                    record = Recording(path)
                    if (
                        record.header.get("allocation") != identity(contract)
                        or record.header.get("unit_id")
                        != unit_id(contract["namespace"], "sessions", index)
                        or record.header.get("source_commit") != contract["commit"]
                        or record.header.get("controller") != contract["controller"]
                        or record.header.get("seed") != contract["rows"][index]["seed"]
                    ):
                        raise ValueError("foreign recorded allocation")
                    # Recheck copied bytes against the exact verified payload.
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != record.sha256:
                        raise ValueError("replay changed during install")
                    publish_immutable(bank / path.name, payload)
                    seen.add(index)
                    installed.append(
                        {"index": index, "sha256": record.sha256, "size": len(payload)}
                    )
                except Exception as exc:
                    exclusions.append({"path": str(path), "error": str(exc)})
    finally:
        publish_json(
            bank / "operational.json",
            {
                "allocation": identity(contract),
                "total": contract["count"],
                "completed": len(installed),
                "units": installed,
                "missing_indices": sorted(set(range(contract["count"])) - seen),
                "exclusions": exclusions,
            },
        )
    return len(installed)


def extract_controller_bank(
    source_bank: Path,
    controller: str,
    bank: Path,
    *,
    product: bool = True,
) -> dict[str, Any]:
    """Build a controller-pure bank dir by COPYING verified .watch.gz+.sha256+.complete triples.

    Verifies every source .watch.gz SHA-256 sidecar and .complete marker first, validates
    each matching recording, and copies (never moves) the matching triples into a fresh
    destination directory.
    """
    if controller not in CONTROLLERS:
        raise ValueError(
            f"unsupported controller {controller!r}; expected one of {sorted(CONTROLLERS)}"
        )
    source_bank = Path(source_bank)
    bank = Path(bank)
    if not source_bank.is_dir() or source_bank.is_symlink():
        raise ValueError(f"source bank must be an existing directory: {source_bank}")
    if source_bank.resolve() == bank.resolve():
        raise ValueError("destination bank must differ from source bank")
    if bank.exists() or bank.is_symlink():
        raise FileExistsError(f"destination bank already exists: {bank}")

    paths = sorted(source_bank.glob("*.watch.gz"))
    if not paths:
        raise ValueError(f"no .watch.gz recordings found in {source_bank}")

    skipped_by_controller: Counter[str] = Counter()
    seen_unit_ids: set[str] = set()
    seen_seeds: set[int] = set()
    verified_triples: list[tuple[Path, bytes, dict[str, Any], str]] = []

    for path in paths:
        # 1. Verify .watch.gz + .sha256 + .complete triple SHA-256 first on every file.
        item = verify_immutable(path)
        expected_sha = str(item["sha256"])
        header = peek_header(path, product=product)
        ctrl = str(header.get("controller", ""))
        if ctrl != controller:
            skipped_by_controller[ctrl] += 1
            continue

        # 2. Full recording validation for matching controller sessions before copying.
        record = Recording(path, product=product)
        if record.sha256 != expected_sha:
            raise ValueError(f"SHA mismatch on {path}: {record.sha256} != {expected_sha}")
        uid = str(record.header["unit_id"])
        seed = int(record.header["seed"])
        if uid in seen_unit_ids:
            raise ValueError(f"duplicate unit_id {uid!r} in {source_bank}")
        if seed in seen_seeds:
            raise ValueError(f"duplicate seed {seed} in {source_bank}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"source recording mutated during read: {path}")
        seen_unit_ids.add(uid)
        seen_seeds.add(seed)
        verified_triples.append((path, payload, record.header, expected_sha))

    if not verified_triples:
        raise ValueError(f"no recordings for controller {controller!r} found in {source_bank}")

    bank.mkdir(parents=True, exist_ok=False)
    copied_units: list[dict[str, Any]] = []
    try:
        for src_path, payload, header, expected_sha in verified_triples:
            dst_path = bank / src_path.name
            published = publish_immutable(dst_path, payload)
            verified_dst = verify_immutable(dst_path)
            if verified_dst["sha256"] != expected_sha or published["sha256"] != expected_sha:
                raise ValueError(f"copied recording SHA mismatch for {dst_path}")
            # Ensure source triple was copied, never moved or altered.
            if verify_immutable(src_path)["sha256"] != expected_sha:
                raise ValueError(f"source recording altered after copy: {src_path}")
            copied_units.append(
                {
                    "file": dst_path.name,
                    "unit_id": str(header["unit_id"]),
                    "seed": int(header["seed"]),
                    "horizon": int(header["horizon"]),
                    "sha256": expected_sha,
                    "size": len(payload),
                }
            )
    except Exception:
        shutil.rmtree(bank, ignore_errors=True)
        raise

    return {
        "source_bank": str(source_bank),
        "bank": str(bank),
        "controller": controller,
        "copied_count": len(copied_units),
        "skipped_count": int(sum(skipped_by_controller.values())),
        "skipped_by_controller": dict(sorted(skipped_by_controller.items())),
        "units": copied_units,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--roots", type=Path, nargs="+")
    parser.add_argument(
        "--source-bank",
        type=Path,
        help="Copy verified triples for --controller from an existing bank directory.",
    )
    parser.add_argument(
        "--controller",
        choices=sorted(CONTROLLERS),
        help="Controller to extract when --source-bank is used.",
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.source_bank is not None:
        if args.contract is not None or args.roots is not None:
            parser.error("--source-bank cannot be combined with --contract or --roots")
        if not args.controller:
            parser.error("--source-bank requires --controller")
        summary = extract_controller_bank(args.source_bank, args.controller, args.bank)
        if args.json_out is not None:
            publish_json(args.json_out.resolve(), summary)
        print(summary["copied_count"])
        return

    if args.contract is None or not args.roots:
        parser.error("either (--source-bank and --controller) or (--contract and --roots) required")
    print(install(json.loads(args.contract.read_text()), args.roots, args.bank))


if __name__ == "__main__":
    main()
