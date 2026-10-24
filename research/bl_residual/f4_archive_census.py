"""Freeze and hash the F4 R5 natural ordinary-row archive census before Look R-A.

Registration R5: the archive is every natural-stream ordinary row of every life
(rescue-bearing or not) whose parent is fold 0 and whose allocation_index is <=
the largest reserve allocation_index. The 64 reserve lives additionally keep all
of their rows, because life-macro retention is read on their decisive rows.

This is a mechanical, create-only conversion of *published* F0 union ledgers to
the single table that ``funding.main`` passes to ``heldout.evaluate`` for both
D256 and D1024 (``funding.evaluate`` requires one identical table and receipt).
It reuses ``f4_d256_package._table`` for the row schema. It computes no metric,
prints no outcome, and is not a registered look; the receipt holds identities
and counts only. The table never leaves the Mac. It refuses to open any row
until both the D256 and D1024 fit-phase ledgers are published (held-out
outcomes are loaded only after the final checkpoints are immutable).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_residual import f4_d256_package as d256
from research.bl_residual import f4_d1024_package as d1024
from research.bl_residual.data import allocate_lives, world_fold
from research.bl_residual.protocol import F4_REGISTRATION_SHA, canonical_sha

SCHEMA = "bl-residual-natural-archive/v2"
PUBLICATION_PATH = "research/bl_residual/ledger/F4_R_A_ARCHIVE_CENSUS_20261024.json"
TABLE = "archive-table.npz"
RECEIPT = "census.json"


def require_frozen_fits(d256_pack_inputs: Path, d1024_pack_inputs: Path) -> None:
    """Open no reserve row until both fits are immutable and published (data.py).

    Reads only the published fit-phase ledgers (path/size/hash) against each
    pack's sealed control identity; never deserializes a checkpoint.
    """
    from research.bl_residual.protocol import read_ledger

    for inputs, publication, index in (
        (d256_pack_inputs, d256.FIT_PUBLICATION_PATH, 0),
        (d1024_pack_inputs, d1024.FIT_PUBLICATION_PATH, d1024.INDEX),
    ):
        d1024.require_publication(d1024.ROOT / publication, publication, None)
        verify_export(inputs / "control.json")
        control = json.loads((inputs / "control.json").read_text())
        units = read_ledger(d1024.ROOT / publication, control["namespace"], control["allocation"])
        if set(units) != {index}:
            raise ValueError(f"HOLD: fit not frozen and published before the census: {publication}")


def _bound_allocation(pack_inputs: Path) -> dict[str, Any]:
    """A fit pack's allocation must be the one its sealed control binds."""
    verify_export(pack_inputs / "control.json")
    control = json.loads((pack_inputs / "control.json").read_text())
    path = pack_inputs / "life-allocation.json"
    if control.get("input_hashes", {}).get("life-allocation.json") != digest(path):
        raise ValueError(f"allocation is not bound by its fit control: {path}")
    value = json.loads(path.read_text())
    if allocate_lives(value["catalog"]) != value:
        raise ValueError(f"allocation differs from the frozen allocator: {path}")
    return value


def archive_identity(allocation: dict[str, Any]) -> dict[str, Any]:
    """R5 eligibility from catalog metadata only (no rows, no outcomes)."""
    reserve = list(allocation["doses"]["256"]["heldout_lives"])
    catalog = {row["life_id"]: row for row in allocation["catalog"]}
    if len(reserve) != 64 or len(set(reserve)) != 64:
        raise ValueError("R5 requires the fixed 64-life reserve")
    reserve_max = max(catalog[life]["allocation_index"] for life in reserve)
    eligible = sorted(
        life
        for life, row in catalog.items()
        if world_fold(row["parent_world"]) == 0 and row["allocation_index"] <= reserve_max
    )
    return {
        "reserve": sorted(reserve),
        "reserve_max_allocation_index": reserve_max,
        "eligible_life_ids": eligible,
        "parent_worlds": sorted({catalog[life]["parent_world"] for life in eligible}),
        "reserve_allocation_sha256": canonical_sha(
            [
                {
                    "life_id": life,
                    "parent_world": catalog[life]["parent_world"],
                    "allocation_index": catalog[life]["allocation_index"],
                }
                for life in sorted(reserve)
            ]
        ),
    }


def archive_table(lives: list[dict[str, Any]], identity: dict[str, Any]) -> dict[str, np.ndarray]:
    """Reserve lives keep every row; every other eligible life keeps ordinary rows.

    ``_table`` keeps all rows of "selected" lives and only ordinary rows of
    non-selected, non-rescue lives on the given parents. The reserve is the
    selected set; other eligible lives are presented as non-selected archive
    lives (their rescue flag is masked only for row selection, never stored).
    """
    reserve = set(identity["reserve"])
    eligible = set(identity["eligible_life_ids"])
    by_id = {str(life["life_id"]): life for life in lives}
    if not eligible <= set(by_id):
        raise ValueError("HOLD: an archive-eligible life is absent from the published unions")
    view = [
        by_id[life] if life in reserve else {**by_id[life], "rescue_bearing": False}
        for life in sorted(eligible, key=lambda name: by_id[name]["allocation_index"])
    ]
    table = d256._table(view, reserve, set(identity["parent_worlds"]))
    present = set(table["life_id"].tolist())
    doom_lives = set(table["life_id"][~table["ordinary"]].tolist())
    if (
        not reserve <= present
        or not doom_lives <= reserve
        or not present <= eligible
        or sorted(set(table["parent_world"].tolist())) != identity["parent_worlds"]
        or any(world_fold(str(parent)) != 0 for parent in table["parent_world"])
    ):
        raise ValueError("HOLD: archive table ancestry differs from the R5 census identity")
    return table


def _census(
    output: Path,
    *,
    d256_ledger: Path,
    d1024_ledger: Path,
    d1024_union_sha: str,
    d256_pack_inputs: Path,
    d1024_pack_inputs: Path,
) -> dict[str, Any]:
    d256.require_window_closed()
    d1024.require_publication(d256_ledger, d1024.D256_UNION_PATH, d1024.D256_UNION_SHA)
    d1024.require_publication(d1024_ledger, d1024.D1024_UNION_PATH, d1024_union_sha)
    if output.exists():
        raise FileExistsError(output)
    require_frozen_fits(d256_pack_inputs, d1024_pack_inputs)
    left = archive_identity(_bound_allocation(d256_pack_inputs))
    right = archive_identity(_bound_allocation(d1024_pack_inputs))
    if left != right:
        raise ValueError("HOLD: D256 and D1024 allocations disagree on the R5 archive census")
    lives = d256._life_catalog(d256_ledger) + d256._life_catalog(d1024_ledger)
    table = archive_table(lives, left)
    output.mkdir(parents=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **table)
    export_bytes(output / TABLE, buffer.getvalue())
    natural = sorted(table["row_id"][table["natural_stream"]].tolist())
    eligible = set(left["eligible_life_ids"])
    rescue = {str(life["life_id"]) for life in lives if life["rescue_bearing"]}
    receipt = {
        "schema": SCHEMA,
        "complete": True,
        "table_sha256": digest(output / TABLE),
        "natural_row_ids": natural,
        "parent_worlds": left["parent_worlds"],
        "eligible_life_ids": left["eligible_life_ids"],
        "reserve_life_ids": left["reserve"],
        "reserve_max_allocation_index": left["reserve_max_allocation_index"],
        "reserve_allocation_sha256": left["reserve_allocation_sha256"],
        "includes_non_rescue_lives": True,
        "counts": {
            "rows": int(len(table["row_id"])),
            "natural_rows": len(natural),
            "eligible_lives": len(eligible),
            "eligible_non_rescue_lives": len(eligible - rescue),
            "parents": len(left["parent_worlds"]),
        },
        "sources": {
            "registration_sha256": F4_REGISTRATION_SHA,
            "d256_union_ledger_sha256": digest(d256_ledger),
            "d1024_union_ledger_sha256": digest(d1024_ledger),
            "d256_life_allocation_sha256": digest(d256_pack_inputs / "life-allocation.json"),
            "d1024_life_allocation_sha256": digest(d1024_pack_inputs / "life-allocation.json"),
        },
        "outcome_reporting": "none",
    }
    export_json(output / RECEIPT, receipt)
    return {
        "census": str(output / RECEIPT),
        "census_sha256": digest(output / RECEIPT),
        "table": str(output / TABLE),
        "table_sha256": receipt["table_sha256"],
        "counts": receipt["counts"],
    }


def census(output: Path, **kwargs: Any) -> dict[str, Any]:
    """Hold the Mac resource boundary during the mechanical conversion."""
    d256.require_window_closed()
    from research.mac_lease import held

    with held("f4-r-a-archive-census", "mechanical census; no look", seconds=2400):
        return _census(output, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d256-ledger", type=Path, required=True)
    parser.add_argument("--d1024-ledger", type=Path, required=True)
    parser.add_argument("--d1024-union-sha256", required=True)
    parser.add_argument("--d256-pack-inputs", type=Path, required=True)
    parser.add_argument("--d1024-pack-inputs", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            census(
                args.out,
                d256_ledger=args.d256_ledger,
                d1024_ledger=args.d1024_ledger,
                d1024_union_sha=args.d1024_union_sha256,
                d256_pack_inputs=args.d256_pack_inputs,
                d1024_pack_inputs=args.d1024_pack_inputs,
            )
        )
    )


if __name__ == "__main__":
    main()
