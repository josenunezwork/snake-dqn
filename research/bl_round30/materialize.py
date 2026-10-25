"""G1-approved fixed selector: exact per-life labels to two sealed partitions.

No fit, statistic, summary of outcomes, or general union reader is used. Validation
is mechanically copied and remains on the Mac. The metadata approval precedes
every label opener; materialized hashes are reviewed separately afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from research.bl_round30.admission import (
    AdmissionRefused,
    _atomic_create,
    _publish_json,
    admission_status,
    freeze_inventory,
    require_codex_v_approval,
)


def _npz_bytes(table: dict) -> bytes:
    """Lossless deterministic NPZ; fixed timestamps and sorted member order."""
    import numpy as np

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(table.items()):
            member = io.BytesIO()
            np.lib.format.write_array(member, value, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6
            )
    return output.getvalue()


def _table(values: list[tuple[dict, dict]]) -> dict:
    """Preserve the original strict F4 columns/dtypes and all supplied rows."""
    import numpy as np

    from research.bl_residual.data import validate_rows
    from research.bl_residual.factory_masks import advisory_from_row

    def array(key: str, dtype: Any) -> Any:
        return np.asarray([row[key] for _, row in values], dtype=dtype)

    ordinary = np.asarray([row.get("sampling_kind") == "ordinary" for _, row in values], dtype=bool)
    source_ids = np.asarray([life["source_life_id"] for life, _ in values])
    result = {
        "local": array("local", np.uint8),
        "global": array("global", np.uint8),
        "scalars": array("scalars", np.float32),
        "q": array("q", np.float32),
        "legal": array("legal", bool),
        "action_mask": np.asarray([advisory_from_row(row) for _, row in values], dtype=bool),
        "bare": array("bare", np.int64),
        "status": np.asarray(
            [
                [str(x) if x is not None else "NOT_EVALUATED" for x in row["status"]]
                for _, row in values
            ]
        ),
        "died": np.asarray(
            [[bool(x) if x is not None else False for x in row["died"]] for _, row in values],
            dtype=bool,
        ),
        "mass": np.asarray(
            [[float(x) if x is not None else 0.0 for x in row["mass"]] for _, row in values],
            dtype=np.float32,
        ),
        "ordinary": ordinary,
        # Preserve the original F4 adapter column. This flag does not admit a
        # natural archive: selected rescue-life ordinary rows are conditioned,
        # and the independent admission archive gate remains HOLD.
        "natural_stream": ordinary.copy(),
        "sampling_prob": np.asarray(
            [float(row["sampling_prob"]) for _, row in values], dtype=np.float32
        ),
        "sampling_weight": np.asarray(
            [1.0 / float(row["sampling_prob"]) for _, row in values], dtype=np.float32
        ),
        "parent_world": np.asarray([life["ultimate_parent_id"] for life, _ in values]),
        "life_id": source_ids,
        "source_life_id": source_ids.copy(),
        "global_life_index": np.asarray(
            [life["global_life_index"] for life, _ in values], dtype=np.int64
        ),
        "row_id": np.asarray(
            [f"{life['source_life_id']}/{int(row['decision'])}" for life, row in values]
        ),
    }
    validate_rows(result)
    return result


def materialize(inventory_path: Path, approval_path: Path, output: Path) -> dict:
    """Open exactly the approved selected lives, after all metadata checks pass."""
    started = time.monotonic()
    # The R30-0 allowance is 900 aggregate core-seconds. Leave 300 seconds
    # for metadata/review checks; no lease or other process is inspected.
    deadline = started + 600.0

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise AdmissionRefused("INCOMPLETE: fixed-selector light Mac allowance exhausted")

    inventory = json.loads(inventory_path.read_text())
    admitted = freeze_inventory(inventory)
    approval = json.loads(approval_path.read_text())
    require_codex_v_approval(admitted, approval)
    if admission_status(admitted) == "INSUFFICIENT":
        raise AdmissionRefused("INSUFFICIENT: validation metadata support")
    if (
        approval.get("approval_scope") != "closed-f4-exact-960-metadata"
        or approval.get("reviewer_role") != "Codex-V"
    ):
        raise AdmissionRefused("independent closed-F4 metadata approval scope required")
    # Refuse destinations and symlinks before any data access. The approved
    # inventory is already immutable; never edit it to add payload hashes.
    if output.exists() or output.is_symlink():
        raise FileExistsError("materialized partition destination exists")
    output.mkdir(parents=True, mode=0o700)
    # Convert one bounded life at a time rather than retaining 2.6 GB of raw
    # JSON as Python lists. No excluded life is opened, even transiently.
    import numpy as np

    parts: dict[str, list] = {"train": [], "validation": []}
    counts: Counter = Counter()
    for life in admitted.inventory["lives"]:
        check_deadline()
        path = Path(life["payload_path"])
        if path.is_symlink() or path.stat().st_size != life["payload_size"]:
            raise AdmissionRefused("selected payload location changed after G1")
        if (
            Path(str(path) + ".sha256").read_text() != f"{life['payload_sha256']}  {path.name}\n"
            or Path(str(path) + ".complete").read_text() != "complete\n"
        ):
            raise AdmissionRefused("selected completion evidence changed after G1")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != life["payload_sha256"]:
            raise AdmissionRefused("selected payload identity changed after G1")
        unit = json.loads(raw)
        namespace, parent, number = life["source_life_id"].split("/")
        expected = {
            "schema": "bl-labels-f0-life/v1",
            "status": "COMPLETE",
            "namespace": namespace,
            "parent_world": parent,
            "allocation_id": parent,
            "life": int(number),
            "life_id": life["source_life_id"],
            "stream": life["stream"],
            "rescue_bearing": True,
            "allocation_index": life["original_allocation_index"],
            "allocation": life["original_allocation"],
        }
        if any(unit.get(key) != value for key, value in expected.items()):
            raise AdmissionRefused("selected life wrapper conflicts with approved metadata")
        rows = unit.get("rows")
        if not isinstance(rows, list) or not rows:
            raise AdmissionRefused("selected life has missing rows")
        seen = set()
        for row in rows:
            decision = row.get("decision")
            if type(decision) is not int or decision < 0 or decision in seen:
                raise AdmissionRefused("duplicate/malformed row identity")
            seen.add(decision)
            # Every supplied ordinary and rescue row is included unchanged.
            counts[
                (
                    life["split"],
                    life["stream"],
                    "ordinary" if row.get("sampling_kind") == "ordinary" else "rescue",
                )
            ] += 1
        parts[life["split"]].append(_table([(life, row) for row in rows]))
    partitions = {}
    for split in ("train", "validation"):
        check_deadline()
        table = {
            key: np.concatenate([part[key] for part in parts[split]], axis=0)
            for key in parts[split][0]
        }
        parts[split].clear()
        payload = _npz_bytes(table)
        path = output / f"{split}.npz"
        _atomic_create(path, payload)
        os.chmod(path, 0o600)
        check_deadline()
        partitions[split] = {
            "path": str(path.absolute()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "rows": len(table["q"]),
            "lives": len(set(table["life_id"].tolist())),
            "parents": len(set(table["parent_world"].tolist())),
            "streams": {
                stream: {kind: counts[(split, stream, kind)] for kind in ("ordinary", "rescue")}
                for stream in ("S-Watch6", "S-gate")
            },
        }
    _publish_json(
        output / "life-id-map.json",
        {
            "schema": "r30-global-life-map/v1",
            "inventory_sha256": admitted.inventory_sha256,
            "ids": {row["source_life_id"]: row["global_life_index"] for row in inventory["lives"]},
        },
    )
    manifest = {
        "schema": "r30-partition-manifest/v1",
        "inventory_sha256": admitted.inventory_sha256,
        "allocation_sha256": admitted.allocation_sha256,
        "metadata_approval_sha256": hashlib.sha256(approval_path.read_bytes()).hexdigest(),
        "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partitions": partitions,
        "life_id_map_sha256": hashlib.sha256(
            (output / "life-id-map.json").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "wall_guard_seconds": 600,
        "status": "INDEPENDENT_HASH_COVERAGE_APPROVAL_REQUIRED",
        "outcome_summaries": False,
        "validation_sealed_locally": True,
    }
    _publish_json(output / "partition-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "approval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.inventory, args.approval, args.output)))


if __name__ == "__main__":
    main()
