"""Build a fresh P1 canary sibling after a verified pre-workload supervisor stop.

The builder deliberately keeps the admitted pack's immutable science and input
inventory.  It only gives the canary and later source a fresh job/output
identity, then re-validates and re-qualifies the resulting manifests locally.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export

QUALIFIED_TREE = "553af3f692515b2bf66173356fe4d099acc96942"
PRIOR_FAILURE_CLASS = "SUPERVISOR"


def _read_export(path: Path) -> dict[str, Any]:
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object export: {path}")
    return value


def _tree(repo: Path, commit: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{commit}:research/colab_cli"], cwd=repo, text=True
    ).strip()


def _retry_identity(manifest: dict[str, Any], *, job_id: str, output: Path) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    result["job_id"] = job_id
    result["output_dir"] = str(output)
    return result


def _copy_export(source: Path, destination: Path) -> None:
    """Copy a verified immutable receipt with fresh sidecars for this sibling."""
    verify_export(source)
    export_bytes(destination, source.read_bytes())


def prepare(*, source: Path, output: Path, qualification: Path, repo: Path) -> dict[str, Any]:
    """Publish a dispatchable canary and held source template with fresh identities."""
    from research.bl_upload.qualify import qualify
    from research.colab_cli.supervisor import validate

    source = source.resolve()
    output = output.resolve()
    repo = repo.resolve()
    if output.exists():
        raise ValueError("retry output must be a fresh directory")
    admitted = _read_export(source / "canary-admitted.json")
    template = _read_export(source / "manifest.json")
    prior = json.loads((source / "canary-output/receipt.json").read_text())
    if (
        prior.get("status") != "INCOMPLETE"
        or prior.get("stop_verified") is not True
        or prior.get("cleanup_error") is not None
        or prior.get("failure_class") != PRIOR_FAILURE_CLASS
    ):
        raise ValueError("prior canary is not the ruled pre-workload supervisor stop")
    record = json.loads(qualification.read_text())
    if record.get("tree_hash") != QUALIFIED_TREE:
        raise ValueError("qualification registry does not name the required transport tree")
    for manifest in (admitted, template):
        if _tree(repo, manifest["commit"]) != QUALIFIED_TREE:
            raise ValueError("manifest pin does not carry the qualified transport tree")
    if (
        admitted.get("scientific_admission") is not True
        or admitted.get("transport", {}).get("dispatch_ready") is not True
    ):
        raise ValueError("base canary is not admitted")
    if (
        template.get("scientific_admission") is not False
        or template.get("transport", {}).get("dispatch_ready") is not False
    ):
        raise ValueError("source must stay held until the actual retry canary passes")

    output.mkdir(parents=True)
    canary = _retry_identity(
        admitted,
        job_id="bl-p1-e8-04-canary-5d877123-retry05",
        output=output / "canary-output",
    )
    held_source = _retry_identity(
        template,
        job_id="bl-p1-e8-04-source-5d877123-retry05",
        output=output / "source-output",
    )
    canary = validate(canary, repo)
    held_canary = copy.deepcopy(canary)
    held_canary["scientific_admission"] = False
    held_canary["transport"]["dispatch_ready"] = False
    # A source cannot pass family-canary validation until its actual retry
    # receipt exists.  It is therefore frozen here and validated by admit().
    _copy_export(source.parent / "y3-debit.json", output.parent / "y3-debit.json")
    export_json(output / "canary-manifest.json", held_canary)
    export_json(output / "canary-admitted.json", canary)
    export_json(output / "source-template.json", held_source)
    upload = qualify(canary, repo=repo)
    export_json(output / "canary-upload-preflight.json", upload)
    export_json(
        output / "preparation.json",
        {
            "schema": "bl-p1-kernel-ambiguity-retry/v1",
            "status": "CANARY_ADMITTED_NOT_DISPATCHED",
            "base_pack": {
                "path": str(source),
                "canary_sha256": digest(source / "canary-admitted.json"),
            },
            "prior_receipt": {
                "path": str(source / "canary-output/receipt.json"),
                "sha256": digest(source / "canary-output/receipt.json"),
                "failure_class": PRIOR_FAILURE_CLASS,
                "stop_verified": True,
            },
            "qualification_registry": {"path": str(qualification), "tree_hash": QUALIFIED_TREE},
            "canary_manifest_sha256": digest(output / "canary-manifest.json"),
            "source_template_sha256": digest(output / "source-template.json"),
            "budget": {"canary_cap": canary["budget"]["unit_cap"]},
            "inventory_binding": {
                "inputs": canary["inputs"],
                "source_inputs_sha256": digest(source / "p1-inputs.tar.gz"),
            },
            "admission": {"canary": True, "source": False},
            "provider_calls": 0,
        },
    )
    return {
        "canary": output / "canary-admitted.json",
        "source": output / "source-template.json",
        "preparation": output / "preparation.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        source=args.source,
        output=args.output,
        qualification=args.qualification,
        repo=args.repo,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
