"""Bind pending main packets to an actual same-pin canary; never dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.watch6_search_efficacy.admission import admit
from tools.ops.package_spec import digest, publish_json


def finalize(package: Path) -> list[Path]:
    """Validate the whole set before publishing any dispatchable shard manifest."""
    pending = sorted(package.glob("shard-*.pending.json"))
    if not pending:
        raise ValueError("no pending main packets")
    manifests = [json.loads(path.read_text()) for path in pending]
    shards = int(manifests[0]["env"]["WATCH6_SHARDS"])
    if (
        len(pending) != shards
        or {int(v["env"]["WATCH6_SHARD"]) for v in manifests} != set(range(shards))
        or any(int(v["env"]["WATCH6_SHARDS"]) != shards for v in manifests)
        or len({v["job_id"] for v in manifests}) != shards
        or len({v["output_dir"] for v in manifests}) != shards
    ):
        raise ValueError("missing or duplicate pending shard identities")
    prepared = []
    for path, value in zip(pending, manifests):
        if value.pop("pending_qualification", None) is not True:
            raise ValueError("only explicit pending qualification may be finalized")
        receipt = Path(value["canary"]["receipt"]["path"])
        value["canary"]["receipt"]["sha256"] = digest(receipt)
        admit(value)
        target = path.with_name(path.name.replace(".pending.json", ".json"))
        if target.exists():
            raise ValueError("final main packet already exists; never overwrite")
        prepared.append((target, value))
    for target, value in prepared:
        publish_json(target, value)
    return [target for target, _ in prepared]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    for path in finalize(args.package):
        print(path)


if __name__ == "__main__":
    main()
