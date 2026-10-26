"""Freeze eligible F5 source identities from the immutable metadata annex only.

No payload reference is followed or statted. The historical converted NPZ is
bound from pack metadata; independent acceptance is still required before reads.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ANNEX_SHA = "8ee06291b23a1ef8c657b59a4f4990349ae4c524a680c47f06785a47d227f764"
NPZ_SHA = "6906bffc7241de568965c1a91f4faebe799895ee0b4650e59b20e43db1dac494"
PACK_SHA = "ee84de223246131d7468c115bf0158f3c656ab5834e8bc502d7d3b39f9a3e6ff"
NPZ_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/"
    "d1024-pack-04/inputs/train-1024.npz"
)


def canonical(value: object) -> bytes:
    """Serialize metadata deterministically."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    """Copy only pre-admitted source identity fields, refusing changed annex bytes."""
    folder = Path(__file__).parent
    prior = folder.parent / "f5_r10a_20261025/F5_R10A_METADATA_ANNEX_20261025.json"
    raw = prior.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANNEX_SHA:
        raise ValueError("original annex identity changed")
    annex = json.loads(raw)
    rows = []
    groups = (annex["corpus"]["selected_lives"], annex["corpus"]["ordinary_only_sibling_lives"])
    for ordinary_only, records in enumerate(groups):
        for row in records:
            source = row["payload_identity_from_ledger_only"]
            parent = row.get("ultimate_parent_id", row.get("parent_world"))
            life = row["life_id"]
            if life.split("/")[-2] != parent or row["original_fold"] == 0:
                raise ValueError("life ancestry mismatch")
            rows.append(
                {
                    "life_id": life,
                    "parent_world": parent,
                    "original_fold": row["original_fold"],
                    "allocation_order": row.get(
                        "original_allocation_index", row.get("allocation_index")
                    ),
                    "selection_index": row.get("selection_index"),
                    "ordinary_only": bool(ordinary_only),
                    "stream": row["stream"],
                    "source_namespace": life.rsplit("/", 2)[0],
                    "source_relative_path": source["path"],
                    "source_unit_id": source["unit_id"],
                    "source_sha256": source["sha256"],
                    "source_size": source["size"],
                }
            )
    if len(rows) != 969 or len({row["life_id"] for row in rows}) != 969:
        raise ValueError("eligible life count changed")
    if any(type(row["allocation_order"]) is not int for row in rows):
        raise ValueError("missing original allocation order")
    value = {
        "schema": "bl-f5-status-input-inventory/v1",
        "status": "FROZEN_METADATA_ONLY_REAL_ACCESS_DISABLED",
        "annex_sha256": ANNEX_SHA,
        "converted_container": {
            "path": NPZ_PATH,
            "sha256": NPZ_SHA,
            "size": 45397643,
            "pack_metadata_sha256": PACK_SHA,
            "pack_commit": "de5bde2d7866eb6cd01060f7c2fe4ed8d1253759",
            "schema_source": "research/bl_residual/f4_d256_package.py::_table",
            "correspondence_source": "research/bl_residual/f4_d1024_package.py",
        },
        "original_life_sources": rows,
        "original_life_sources_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
        "counts": {"selected": 960, "ordinary_only_siblings": 9, "total": 969},
        "source_byte_bounds_from_metadata": {
            "total_original_json_bytes": sum(row["source_size"] for row in rows),
            "max_original_json_bytes": max(row["source_size"] for row in rows),
        },
        "bounds": {"max_rows": 94152, "max_ordinary_rows": 7752, "max_U": 46512},
        "bounds_derivation": "960*(ROOT_CAP90+PRESERVATION_ROOTS8)+9*8; ordinary969*8; U6*ordinary",
        "training_exclusions": annex["exclusions"]["training_excluded_parent_union"],
        "exclusion_inventory_sha256": annex["exclusion_inventory_sha256"],
        "real_files_opened_statted_or_hashed": False,
    }
    target = folder / "extractor-input-inventory.json"
    body = canonical(value)
    for path, payload in (
        (target, body),
        (
            Path(str(target) + ".sha256"),
            f"{hashlib.sha256(body).hexdigest()}  {target.name}\n".encode(),
        ),
    ):
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    print(hashlib.sha256(body).hexdigest())


if __name__ == "__main__":
    main()
