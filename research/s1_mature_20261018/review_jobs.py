"""Render ten DRAFT manifests for review and validate them without allocation."""

import argparse
import json
from pathlib import Path

from research.colab_cli import supervisor
from research.s1_mature_20261018 import manifests

PACKAGE = Path(__file__).resolve().parent


def verified_inputs(path: Path = PACKAGE / "INPUTS.json") -> list[manifests.ExternalInput]:
    """Require the frozen local input inventory before preparing any manifest."""
    value = json.loads(path.read_text())
    if value.get("schema") != "s1-input-inventory/v1":
        raise ValueError("wrong S1 input inventory")
    result = []
    for item in value["inputs"]:
        source = Path(item["path"])
        if manifests.sha256_file(source) != item["sha256"]:
            raise ValueError(f"registered input changed: {item['dest']}")
        result.append(manifests.ExternalInput(item["name"], source, item["dest"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("review manifest directory must be fresh")
    rows = manifests.make_manifests(
        seed_bank=PACKAGE / "SEED_BANK.json",
        published_commit=args.commit,
        registration_status="DRAFT_PENDING_OWNER_REVIEW",
        owner_release="DRAFT_PENDING_OWNER_REVIEW",
        output_root=args.output_root,
        inputs=verified_inputs(),
    )
    validated = [supervisor.validate(row) for row in rows]
    args.out.mkdir(parents=True)
    for row in validated:
        with (args.out / f"{row['job_id']}.json").open("x") as stream:
            json.dump(row, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": "VALIDATED_DRAFT_ONLY",
                "jobs": len(validated),
                "commit": args.commit,
                "colab_allocations": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
