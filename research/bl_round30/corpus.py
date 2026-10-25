"""Metadata-only, exact-list closed-F4 admission. Never opens a label payload.

Producer: published F4 allocation and immutable F0 union ledgers. Consumers:
independent metadata review, then the fixed per-life partition materializer.
The union is an index only; no general union payload reader is imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from research.bl_round30.admission import (
    INVENTORY_SCHEMA,
    AdmissionRefused,
    _publish_json,
    canonical_bytes,
    parent_split,
    sha256_json,
)
from research.bl_round30.protocol import AMENDMENT_SHA, B8_SHA, JUDGMENT_SHA, SALT

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever")
ALLOCATION_SHA = "31fe074d72c3fe7eaeb1d30c59622a6f5b4add4d3c6c2b8320048265d5986c4d"
PINS = {
    "research/bl_labels/ledger/D256_UNION_FINAL_20261023.json": (
        "30919209b62307f8d4392b4e5925a7c4b0c4a21a2abd55a5e647428453eeb809"
    ),
    "research/bl_labels/ledger/D1024_UNION_FINAL_20261024.json": (
        "f5ec94ccdfae90133362a9072227d83125ae0ce661c44471904ce57333b610b8"
    ),
    "research/bl_residual/ledger/F4_D256_METADATA_FIT_BINDING_20261024.json": (
        "65731d8bf2a4d63e81b18ad5024359604d38fa9e36253ae97bef1f94913d9cba"
    ),
    "research/bl_residual/ledger/F4_D1024_FIT_BINDING_20261024.json": (
        "b44ba54e84b68add3f583d66556d37d5eea19aeccad89dcd764e20076dd9bed2"
    ),
}


def metadata(path: Path, sha: str, bindings: list[dict]) -> dict:
    """Read a specifically selected metadata file only after verifying its pin."""
    if path.is_symlink() or any(x in str(path).lower() for x in ("d4096", "t34-round0-study-01")):
        raise AdmissionRefused("forbidden metadata path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise AdmissionRefused(f"stale metadata pin: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AdmissionRefused("metadata must be an object")
    bindings.append({"path": str(path.absolute()), "sha256": sha, "size": len(raw)})
    return value


def _fold(parent: str) -> int:
    raw = hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % 5


def validate_life_inventory(value: dict) -> None:
    """Fail closed on one foreign, duplicated, missing or cross-partition life."""
    lives = value.get("lives")
    if not isinstance(lives, list) or len(lives) != 960:
        raise AdmissionRefused("closed F4 inventory requires exactly 960 lives")
    ids = [row.get("source_life_id") for row in lives]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 960:
        raise AdmissionRefused("duplicate or missing global source-life ID")
    if sha256_json(ids) != value["corpus"]["provenance"]["admitted_life_set_sha"]:
        raise AdmissionRefused("admitted life set identity differs")
    parent_ids = {row["ultimate_parent_id"] for row in value["parents"]}
    if parent_ids != {row.get("ultimate_parent_id") for row in lives}:
        raise AdmissionRefused("missing or foreign life parents")
    for index, row in enumerate(lives):
        source = row["source_life_id"]
        parent = row.get("ultimate_parent_id")
        parts = source.split("/")
        if (
            len(parts) != 3
            or parts[1] != parent
            or not parts[2].isdigit()
            or parts[0] not in value["source_namespaces"]
            or row.get("original_fold") != _fold(parent)
            or row["original_fold"] not in (1, 2, 3, 4)
            or row.get("split") != parent_split(parent, SALT)
            or row.get("global_life_index") != sorted(ids).index(source)
            or row.get("selection_index") != index
            or row.get("protected") is not False
            or row.get("foreign") is not False
            or row.get("rescue_bearing") is not True
            or row.get("stream") not in ("S-Watch6", "S-gate")
            or not isinstance(row.get("payload_size"), int)
            or row["payload_size"] <= 0
        ):
            raise AdmissionRefused("foreign/protected life, lineage, fold or map")
        path = Path(row.get("payload_path", ""))
        if any(x in str(path).lower() for x in ("d4096", "t34-round0-study-01")):
            raise AdmissionRefused("forbidden life payload path")
        expected_name = f"{parent}--life-{int(parts[2]):06d}.json"
        if path.name != expected_name or not path.is_absolute():
            raise AdmissionRefused("life payload path identity differs")
        for key in ("payload_sha256", "source_union_sha256"):
            digest = row.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise AdmissionRefused("missing life payload/publication hash")


def build_inventory(repo: Path) -> dict:
    """Resolve exactly the published D1024 selection, without inspecting rows."""
    bindings: list[dict] = []
    values = {name: metadata(repo / name, sha, bindings) for name, sha in PINS.items()}
    unions = [values[name] for name in PINS if "UNION_FINAL" in name]
    allocations = []
    controls = []
    for name in (
        "F4_D256_METADATA_FIT_BINDING_20261024.json",
        "F4_D1024_FIT_BINDING_20261024.json",
    ):
        binding = values[f"research/bl_residual/ledger/{name}"]
        manifest_ref = binding["files"]["manifest"]
        manifest = metadata(Path(manifest_ref["path"]), manifest_ref["sha256"], bindings)
        # The manifest input inventory binds the unpacked control. The tar is
        # never extracted or opened for metadata admission.
        pack = Path(binding["pack"])
        control_path = pack / "inputs/control.json"
        archives = [row for row in manifest["inputs"] if row.get("dest") == "metadata.tar"]
        if len(archives) != 1:
            raise AdmissionRefused("bound manifest does not resolve metadata archive")
        archive_path = Path(archives[0]["path"])
        archive_bytes = archive_path.read_bytes()
        if hashlib.sha256(archive_bytes).hexdigest() != archives[0]["sha256"]:
            raise AdmissionRefused("bound metadata archive differs")
        bindings.append(
            {"path": str(archive_path), "sha256": archives[0]["sha256"], "size": len(archive_bytes)}
        )
        # A fixed named metadata member only; never extract the archive or
        # follow arbitrary members. Payload NPZ is not an archive member.
        with tarfile.open(archive_path) as archive:
            members = [m for m in archive.getmembers() if m.name == "control.json"]
            if len(members) != 1 or not members[0].isfile() or members[0].size > 65536:
                raise AdmissionRefused("metadata control member is unsafe")
            control_sha = hashlib.sha256(archive.extractfile(members[0]).read()).hexdigest()
        control = metadata(control_path, control_sha, bindings)
        controls.append(control)
        allocation = metadata(
            pack / "inputs/life-allocation.json",
            control["input_hashes"]["life-allocation.json"],
            bindings,
        )
        allocations.append(allocation)
    prior, allocation = allocations
    if (
        hashlib.sha256(
            (ART / "exp/f4/d1024-pack-04/inputs/life-allocation.json").read_bytes()
        ).hexdigest()
        != ALLOCATION_SHA
    ):
        raise AdmissionRefused("wrong published D1024 allocation")
    selected = allocation["doses"]["1024"]["train_lives"]
    prefix = prior["doses"]["256"]["train_lives"]
    if (
        len(selected) != 960
        or len(prefix) != 192
        or selected[:192] != prefix
        or len(set(selected)) != 960
    ):
        raise AdmissionRefused("D256 prefix/D1024 exact selection mismatch")
    catalog = {}
    for row in allocation["catalog"]:
        source = row["life_id"]
        if source in catalog and catalog[source] != row:
            raise AdmissionRefused("conflicting duplicate catalog identity")
        catalog[source] = row
    unit_index = {}
    for union, sha in zip(unions, list(PINS.values())[:2]):
        for unit in union["units"]:
            if "--life-" not in unit["unit_id"]:
                continue
            key = (union["namespace"], unit["unit_id"])
            if key in unit_index and unit_index[key][0] != unit:
                raise AdmissionRefused("conflicting immutable duplicate unit")
            unit_index[key] = (unit, union, sha)
    rows = []
    protected = {
        allocation["parents"][source] for source in allocation["doses"]["1024"]["heldout_lives"]
    }
    sorted_ids = {source: index for index, source in enumerate(sorted(selected))}
    for index, source in enumerate(selected):
        cat = catalog[source]
        namespace, parent, number = source.split("/")
        if parent in protected or _fold(parent) == 0 or cat["parent_world"] != parent:
            raise AdmissionRefused("selected protected ancestry")
        unit_name = f"{parent}--life-{int(number):06d}"
        unit, union, union_sha = unit_index[(namespace, unit_name)]
        candidates = [Path(root) / unit["path"] for root in union["attempt_exports"]]
        present = [path for path in candidates if path.is_file() and not path.is_symlink()]
        if not present:
            raise AdmissionRefused("selected unit has no declared completed payload location")
        # Metadata checks only; no payload digest/read happens before G1 ACCEPT.
        for path in present:
            if path.stat().st_size != unit["size"]:
                raise AdmissionRefused("selected payload size differs")
            if (
                Path(str(path) + ".sha256").read_text() != f"{unit['sha256']}  {path.name}\n"
                or Path(str(path) + ".complete").read_text() != "complete\n"
            ):
                raise AdmissionRefused("selected completion evidence differs")
        rows.append(
            {
                "source_life_id": source,
                "global_life_index": sorted_ids[source],
                "selection_index": index,
                "ultimate_parent_id": parent,
                "original_fold": _fold(parent),
                "original_allocation_index": cat["allocation_index"],
                "stream": cat["stream"],
                "rescue_bearing": cat["rescue_bearing"],
                "split": parent_split(parent, SALT),
                "payload_path": str(present[0]),
                "payload_size": unit["size"],
                "payload_sha256": unit["sha256"],
                "source_union_sha256": union_sha,
                "completion_evidence": "published-ledger+sha-sidecar+marker",
                "original_allocation": next(
                    a for a in union["allocation"] if a["allocation_id"] == parent
                ),
                "collection_source_commit": (
                    namespace.split("-")[3]
                    if namespace.startswith("f0-hindsight-labels-")
                    else None
                ),
                "protected": False,
                "foreign": False,
            }
        )
    provenance = {
        "kind": "closed_f4_training",
        "original_f4": True,
        "terminal_label": "STOP_FUNDING",
        "reuse_ruling_sha": AMENDMENT_SHA,
        "dose_allocation_sha": ALLOCATION_SHA,
        "admitted_life_set_sha": sha256_json(selected),
        "eligible_life_count": 960,
        "original_folds": [1, 2, 3, 4],
    }
    parents = [
        {
            "ultimate_parent_id": parent,
            "rescue_bearing": True,
            "previously_inspected": True,
            "prior_uses": [
                "F4 D1024 fit; terminal STOP_FUNDING",
                "F4 study motivated retrospective R30 audit",
            ],
            "protected": False,
            "foreign": False,
            "f4": True,
            "stale": False,
            "provenance": provenance,
            "split": parent_split(parent, SALT),
        }
        for parent in sorted({row["ultimate_parent_id"] for row in rows})
    ]
    value = {
        "schema": INVENTORY_SCHEMA,
        "split_salt": SALT,
        "ruling_sha256": JUDGMENT_SHA,
        "amendment_sha256": AMENDMENT_SHA,
        "terminal_ruling_sha256": (
            "be1516a98ac7851ad30cc9694aa1a70b57cd30950ebafe58c4ffbc02ac3deb05"
        ),
        "terminal_r9_label": 5,
        "terminal_result": "STOP_FUNDING",
        "d4096_seal_authority": (
            "CAMPAIGN_LOG.md 2026-10-25 10:50 PT; owner-supplied published seal; "
            "no D4096 tranche or seal file opened by this adapter"
        ),
        "continuation_sha256": B8_SHA,
        "source_namespaces": [x["namespace"] for x in unions],
        "metadata_bindings": bindings,
        "lives": rows,
        "parents": parents,
        "corpus": {
            "collection_authority": "F4_REGISTRATION.md SHA " + controls[-1]["registration_sha256"],
            "source_sha256": hashlib.sha256(canonical_bytes(bindings)).hexdigest(),
            "label_sha256": sha256_json([row["payload_sha256"] for row in rows]),
            "checkpoint_sha256": B8_SHA,
            "explicitly_reusable_development": True,
            "already_unsealed": True,
            "pins_current": True,
            "f4": True,
            **{
                key: False
                for key in (
                    "protected",
                    "strict",
                    "retention",
                    "natural_archive",
                    "probe",
                    "gameplay",
                    "stale",
                )
            },
            "provenance": provenance,
        },
        "natural_ordinary_archive": None,
        "exclusions": {
            "reserve_lives": 64,
            "fold0_parents": "all excluded",
            "unused_tails": "all excluded",
            "archive": "excluded; R30-2 HOLD",
            "D4096": "SEALED_NOT_RUN_BY_RULE; never opened",
            "protected_study": "never opened",
        },
        "row_totals": {
            "status": "DEFERRED_UNTIL_G1_APPROVED_FIXED_SELECTOR",
            "reason": "bound allocation has life metadata only; no payload opened",
        },
        "prior_exposure": (
            "Historical F4 labels; new R30 withholding is not fresh evidence. "
            "Exact F4 harm H was not exported; no recovery authorized."
        ),
        "counts": {
            split: {
                "parents": sum(p["split"] == split for p in parents),
                "lives": sum(r["split"] == split for r in rows),
                "streams": dict(Counter(r["stream"] for r in rows if r["split"] == split)),
            }
            for split in ("train", "validation")
        },
    }
    from research.bl_round30.admission import freeze_inventory

    return freeze_inventory(value).inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_inventory(args.repo)
    args.output.mkdir(parents=True, exist_ok=False)
    _publish_json(args.output / "corpus-inventory.json", value)
    _publish_json(
        args.output / "CORPUS_REUSE_INTENT.json",
        {
            "schema": "bl-r30-closed-f4-reuse-intent/v1",
            "ruling_sha256": JUDGMENT_SHA,
            "amendment_sha256": AMENDMENT_SHA,
            "inventory_sha256": sha256_json(value),
            "status": "G1_METADATA_REVIEW_REQUIRED",
            "numerical_consumption": False,
            "counts": value["counts"],
            "row_totals": value["row_totals"],
            "scope": value["corpus"]["provenance"],
            "exclusions": value["exclusions"],
            "archive_status": "HOLD_R30_2_NO_ADMITTED_NATURAL_ARCHIVE",
        },
    )
    print(json.dumps({"inventory_sha256": sha256_json(value), "counts": value["counts"]}))


if __name__ == "__main__":
    main()
