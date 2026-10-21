"""Seal settled R1 chunk exports as label data (inventory + held-out support), Mac-side.

Run only after every R1 chunk is terminal and **no Path B retention phase is
running or admitted** (RULING_label_ordering (E)); the operator attests that
with ``--attest-no-retention-phase``.  The seal:

* loads every complete world export through the ratified ``load_world`` (sidecar
  SHA-256 + completion marker), refusing duplicates, world ids outside the
  chunk's audited namespace, a carrier/rollout other than B8 s3, beta != 1, or a
  non-v2 kernel;
* records per-world SHA-256, chunk and split (train / heldout), so a later fit
  can bind the exact training set and can never read a held-out world;
* applies the held-out support minimum (>= 60 decisive-bearing worlds and
  >= 300 ``horizon_rescue`` rows) -> ``SUPPORTED`` or ``PRECONDITION_FAILED``
  (reported; no top-up without a new registration);
* writes ``R1_LABELS_SEALED.json`` create-only, with ``.sha256``.

It computes no fit, screen, selection or student metric.  Its counts are label
inventory, not evidence, and may not be cited in the retention verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .plan import CHUNK_WORLDS, Chunk, heldout_support

SCHEMA = "t34-claim-a-r1-sealed-labels/v1"
TEACHER_KERNEL = "shadow-lad-v2"


class SealRefused(ValueError):
    """The settled exports cannot be sealed as the registered R1 label bank."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory_chunk(
    chunk: Chunk, export_root: Path, *, b8_sha256: str, world_paths: Sequence[Path] | None = None
) -> dict[str, Any]:
    from research.t34_exit_b8.exports import load_world

    paths = world_paths
    if paths is None:
        found = sorted(Path(export_root).rglob("world-*.json"))
        paths = [p for p in found if p.with_name(p.name + ".complete").is_file()]
    worlds = []
    for path in paths:
        exported = load_world(path)
        world_id = str(exported["world_id"])
        if not world_id.startswith(chunk.namespace + "/"):
            raise SealRefused(f"{world_id} is outside the chunk namespace {chunk.namespace}")
        if exported.get("teacher_v2", {}).get("kernel") != TEACHER_KERNEL:
            raise SealRefused(f"{world_id}: not a v2 shadow-LAD export")
        if exported["checkpoint_sha256"] != b8_sha256:
            raise SealRefused(f"{world_id}: carrier is not B8 s3")
        if exported.get("rollout_policy_sha256", b8_sha256) != b8_sha256:
            raise SealRefused(f"{world_id}: rollout policy is not B8 s3 (R1 is beta-1 bare B8)")
        if float(exported.get("beta", 1.0)) != 1.0:
            raise SealRefused(f"{world_id}: beta {exported.get('beta')} != 1")
        decisive = sum(
            1
            for label in exported["labels"]
            if label.get("outcome_kind") == "horizon_rescue" and bool(label.get("decisive"))
        )
        worlds.append(
            {
                "world_id": world_id,
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(Path(path)),
                "mix": exported.get("mix"),
                "decisive_rows": decisive,
            }
        )
    if len(worlds) > CHUNK_WORLDS:
        raise SealRefused(f"{chunk.job_id}: more worlds than the registered chunk size")
    return {
        "job_id": chunk.job_id,
        "split": chunk.split,
        "namespace": chunk.namespace,
        "complete_worlds": len(worlds),
        "registered_worlds": CHUNK_WORLDS,
        "worlds": worlds,
    }


def seal(
    chunks: Mapping[Chunk, Path],
    out: Path,
    *,
    b8_sha256: str,
    round0: Mapping[str, Any],
    attest_no_retention_phase: bool,
) -> dict[str, Any]:
    if not attest_no_retention_phase:
        raise SealRefused("sealing is Mac work: attest that no retention phase runs or is admitted")
    from .make_manifest import BINDING_KEYS

    if round0.get("decision") != "GO" or any(not round0.get(k) for k in BINDING_KEYS):
        raise SealRefused("R1 labels seal only under the bound governing Round 0 GO")
    inventories = [
        inventory_chunk(c, p, b8_sha256=b8_sha256)
        for c, p in sorted(
            chunks.items(), key=lambda item: (item[0].split, item[0].shard, item[0].attempt)
        )
    ]
    ids = [w["world_id"] for inv in inventories for w in inv["worlds"]]
    if len(set(ids)) != len(ids):
        raise SealRefused("duplicate world ids across chunks")
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    for inv in inventories:
        by_split[inv["split"]].extend(inv["worlds"])
    support = heldout_support({w["world_id"]: w["decisive_rows"] for w in by_split["heldout"]})
    sealed = {
        "schema": SCHEMA,
        "claim": "R1_LABEL_DATA_ONLY_NOT_EVIDENCE",
        "status": support["status"],
        "round0_governing": dict(round0),
        "b8_sha256": b8_sha256,
        "worlds": {split: len(items) for split, items in by_split.items()},
        "registered_worlds": {"train": 800, "heldout": 200},
        "train_decisive_rows": sum(w["decisive_rows"] for w in by_split["train"]),
        "heldout_support": support,
        "train_world_set_sha256": hashlib.sha256(
            "\n".join(sorted(f"{w['world_id']} {w['sha256']}" for w in by_split["train"])).encode()
        ).hexdigest(),
        "heldout_world_set_sha256": hashlib.sha256(
            "\n".join(
                sorted(f"{w['world_id']} {w['sha256']}" for w in by_split["heldout"])
            ).encode()
        ).hexdigest(),
        "chunks": inventories,
        "attested_no_retention_phase": True,
        "rules": (
            "label data only until the Path B disposition; held-out worlds are never read by a "
            "fit; no top-up without a new registration"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(sealed, indent=2, sort_keys=True) + "\n").encode()
    with out.open("xb") as stream:
        stream.write(encoded)
    out.with_name(out.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {out.name}\n"
    )
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk",
        action="append",
        required=True,
        help="SPLIT:SHARD:ATTEMPT=EXPORT_DIR (one per settled chunk)",
    )
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--round0-binding", type=Path, required=True, help="manifest JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attest-no-retention-phase", action="store_true")
    args = parser.parse_args()
    chunks = {}
    for item in args.chunk:
        spec, _, directory = item.partition("=")
        split, shard, attempt = spec.split(":")
        chunks[Chunk(split, int(shard), int(attempt))] = Path(directory)
    round0 = json.loads(args.round0_binding.read_text())["round0_governing"]
    sealed = seal(
        chunks,
        args.out,
        b8_sha256=sha256_file(args.b8),
        round0=round0,
        attest_no_retention_phase=args.attest_no_retention_phase,
    )
    print(json.dumps({"status": sealed["status"], "worlds": sealed["worlds"]}))


if __name__ == "__main__":
    main()
