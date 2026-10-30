"""Label-free Mac metadata/support census; never read the sealed outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.doom_auc_smoke.data_mac import BANK_SHA
from research.doom_auc_smoke.evidence import digest, json_once

DOCS = (
    "REGISTRATION_OPERATIONAL_CLARIFICATION_02.md",
    "REGISTRATION_PRELOOK_AMENDMENT_01.md",
)
KINDS = ("ordinary", "death_lookback")


def documents() -> dict[str, str]:
    return {name: digest(Path(__file__).parent / name) for name in DOCS}


def census(metadata: list[dict], partitions: list[str]) -> dict:
    """Count opportunities for class support, without inferring any death labels."""
    worlds = {
        str(i): dict(roots=0, late_roots=0, **{kind: 0 for kind in KINDS})
        for i, partition in enumerate(partitions)
        if partition == "assessment"
    }
    roots, seen = [], set()
    for index, row in enumerate(metadata):
        uid, world, kind = row["root_uid"], row["world_id"], row["kind"]
        if (
            row["root_index"] != index
            or uid in seen
            or not 0 <= world < len(partitions)
            or row["partition"] != partitions[world]
            or kind not in KINDS
        ):
            raise ValueError("duplicate/foreign/misaligned root metadata")
        seen.add(uid)
        if row["partition"] != "assessment":
            continue
        root = dict(uid=uid, world=world, length=row["hero_length"], kind=kind)
        roots.append(root)
        counts = worlds[str(world)]
        counts["roots"] += 1
        if root["length"] >= 600:
            counts["late_roots"] += 1
            counts[kind] += 1
    eligible = sum(row["late_roots"] > 0 for row in worlds.values())
    return dict(
        roots=roots,
        per_assessment_world=worlds,
        assessment_roots=len(roots),
        late_roots=sum(row["late_roots"] for row in worlds.values()),
        eligible_parent_worlds=eligible,
        required_positive_parent_worlds=10,
        required_negative_parent_worlds=10,
        necessary_world_count_met=eligible >= 10,
        class_support_verified=False,
        note="Label-free opportunity count only; INCOMPLETE_SUPPORT remains possible.",
    )


def prepare_prelook(packet: Path, bank: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("pre-look metadata is create-only")
    config = json.loads((packet / "config.json").read_text())
    if Path(config["mac_look_file"]).exists():
        raise ValueError("scoring look already reserved; amendment is too late")
    if digest(bank) != BANK_SHA:
        raise ValueError("foreign bank")
    raw = torch.load(bank, map_location="cpu", weights_only=True)
    if raw["schema"] != "ct-stage-0001/v1":
        raise ValueError("foreign bank schema")
    data = raw["dataset"]
    report = census(data["store"]["metadata"], data["world_partitions"])
    report.update(
        schema="doom-prelook/v1",
        identity=config["identity"],
        assessment_sha256=config["assessment_sha256"],
        bank_sha256=BANK_SHA,
        documents=documents(),
        labels_read=False,
    )
    json_once(output, report)
    return report


def read_prelook(path: Path, config: dict) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "doom-prelook/v1"
        or report.get("identity") != config["identity"]
        or report.get("assessment_sha256") != config["assessment_sha256"]
        or report.get("bank_sha256") != BANK_SHA
        or report.get("documents") != documents()
        or report.get("labels_read") is not False
    ):
        raise ValueError("foreign pre-look metadata or amendment binding")
    return report


def attach_kinds(held: dict, report: dict) -> dict:
    roots = report["roots"]
    if (
        held["root_uids"] != [r["uid"] for r in roots]
        or held["world"].tolist() != [r["world"] for r in roots]
        or held["length"].tolist() != [r["length"] for r in roots]
        or any(r["kind"] not in KINDS for r in roots)
    ):
        raise ValueError("pre-look root metadata join differs")
    return dict(held, kind=[r["kind"] for r in roots])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_prelook(args.packet.resolve(), args.bank.resolve(), args.output.resolve())
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"roots", "per_assessment_world"}})
    )


if __name__ == "__main__":
    main()
