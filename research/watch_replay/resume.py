"""Mac-only, outcome-blind union ledger and intra-session checkpoint staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.watch_replay.run import SESSION_CHECKPOINT_SCHEMA, identity
from tools.ops.package_spec import (
    empty_ledger,
    publish_immutable,
    publish_json,
    unit_id,
    verify_immutable,
)


def build(contract: dict, roots: list[Path]) -> dict:
    """Read only frozen paths/sizes/digests, never frames or session stats."""
    value = empty_ledger(contract["namespace"], "sessions", identity(contract))
    seen = set()
    for root in roots:
        for path in sorted(root.glob("*.watch.gz")):
            index = int(path.name.split(".")[0])
            if index in seen or not 0 <= index < contract["count"]:
                raise ValueError("duplicate or foreign allocation index")
            # Every exported attempt must carry the exact frozen allocation.
            binding = json.loads((root / "contract.json").read_text())
            verify_immutable(root / "contract.json")
            if binding != contract:
                raise ValueError("foreign namespace/source/allocation")
            item = verify_immutable(path)
            value["units"].append(
                {
                    "index": index,
                    "unit_id": unit_id(contract["namespace"], "sessions", index),
                    "size": item["size"],
                    "sha256": item["sha256"],
                }
            )
            seen.add(index)
    value["units"].sort(key=lambda row: row["index"])
    return value


def stage_session_checkpoints(
    contract: dict,
    roots: list[Path],
    checkpoint_out: Path,
) -> list[dict[str, Any]]:
    """Stage the newest verified intra-session checkpoint per incomplete session."""
    from research.world_checkpoint.checkpoint import (
        find_latest_checkpoint,
        load_checkpoint,
        verify_checkpoint,
    )

    ledger = build(contract, roots)
    completed_indices = {int(row["index"]) for row in ledger["units"]}
    latest_by_index: dict[int, tuple[int, Path, str]] = {}

    for root in roots:
        binding_path = root / "contract.json"
        if binding_path.is_file():
            verify_immutable(binding_path)
            if json.loads(binding_path.read_text()) != contract:
                raise ValueError("foreign namespace/source/allocation")
        ckpt_base = root / "checkpoints"
        if not ckpt_base.is_dir():
            continue
        for sub in sorted(ckpt_base.iterdir()):
            if not sub.is_dir() or len(sub.name) != 6 or not sub.name.isdigit():
                continue
            index = int(sub.name)
            if index in completed_indices or not 0 <= index < contract["count"]:
                continue
            latest = find_latest_checkpoint(sub, verify_identity=False)
            if latest is None:
                continue
            sha = verify_checkpoint(latest)
            ckpt = load_checkpoint(latest, verify_identity=False)
            row = contract["rows"][index]
            if (
                ckpt.get("session_checkpoint_schema") != SESSION_CHECKPOINT_SCHEMA
                or ckpt.get("allocation") != identity(contract)
                or ckpt.get("namespace") != contract["namespace"]
                or int(ckpt.get("index", -1)) != index
                or int(ckpt.get("seed", -1)) != int(row["seed"])
                or ckpt.get("controller") != contract["controller"]
                or int(ckpt.get("horizon", -1)) != int(contract["horizon"])
            ):
                raise ValueError(f"foreign or mismatched session checkpoint: {latest}")
            frame_num = int(ckpt["frame"])
            prev = latest_by_index.get(index)
            if prev is None or frame_num > prev[0]:
                latest_by_index[index] = (frame_num, latest, sha)

    staged: list[dict[str, Any]] = []
    for index in sorted(latest_by_index):
        frame_num, src_ckpt, sha = latest_by_index[index]
        dst_ckpt = checkpoint_out / f"{index:06d}" / src_ckpt.name
        meta = publish_immutable(dst_ckpt, src_ckpt.read_bytes())
        staged.append(
            {
                "index": index,
                "frame": frame_num,
                "source": str(src_ckpt),
                "dest": str(dst_ckpt.relative_to(checkpoint_out)),
                "sha256": sha,
                "size": int(meta["size"]),
            }
        )
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--roots", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    publish_json(args.out, build(contract, args.roots))
    if args.checkpoint_out is not None:
        stage_session_checkpoints(contract, args.roots, args.checkpoint_out)


if __name__ == "__main__":
    main()
