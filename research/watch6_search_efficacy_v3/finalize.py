"""Seal pending Watch6 V3 shard packets to an actual passing same-pin L4 canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.admission import admit
from tools.ops.package_spec import digest, publish_json


def finalize(package: Path, *, shard: int | None = None, idempotent: bool = False) -> list[Path]:
    """Seal all pending shards in package, or a single shard idempotently for ready_queue."""
    package = package.resolve(strict=True)
    pending_paths = sorted(package.glob("shard-*.pending.json"))
    if not pending_paths:
        raise ValueError(f"no pending shard packets in {package}")
    manifests = [json.loads(path.read_text()) for path in pending_paths]
    shard_ids = [int(m["env"]["WATCH6_SHARD"]) for m in manifests]
    if (
        len(shard_ids) != len(set(shard_ids))
        or any(s not in range(p.SHARDS) for s in shard_ids)
        or any(int(m["env"]["WATCH6_SHARDS"]) != p.SHARDS for m in manifests)
        or len({m["job_id"] for m in manifests}) != len(manifests)
        or len({m["output_dir"] for m in manifests}) != len(manifests)
    ):
        raise ValueError("invalid or duplicate pending shard identities")

    selected = []
    for path, value in zip(pending_paths, manifests):
        s_idx = int(value["env"]["WATCH6_SHARD"])
        if shard is not None and s_idx != shard:
            continue
        selected.append((path, value))
    if not selected:
        raise ValueError(f"requested shard {shard} not found among pending packets in {package}")

    to_publish: list[tuple[Path, dict]] = []
    already_sealed: list[Path] = []
    for path, value in selected:
        target = path.with_name(path.name.replace(".pending.json", ".json"))
        if target.exists():
            if not idempotent:
                raise ValueError(f"final shard packet already exists: {target}")
            admit(json.loads(target.read_text()))
            already_sealed.append(target)
            continue
        if value.pop("pending_qualification", None) is not True:
            raise ValueError(f"packet lacks pending_qualification=true: {path}")
        receipt = Path(value["canary"]["receipt"]["path"])
        value["canary"]["receipt"]["sha256"] = digest(receipt)
        admit(value)
        to_publish.append((target, value))

    for target, value in to_publish:
        publish_json(target, value)
    return already_sealed + [target for target, _ in to_publish]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--idempotent", action="store_true")
    args = parser.parse_args()
    for path in finalize(args.package, shard=args.shard, idempotent=args.idempotent):
        print(path, flush=True)


if __name__ == "__main__":
    main()
