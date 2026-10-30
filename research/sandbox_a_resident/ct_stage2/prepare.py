"""Seal a branch/fit/evaluation packet against the completed immutable H5000 bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.compute.gpu_preference import prepare_preference, variants
from research.sandbox_a_resident.ct_h5000.prepare import source_paths as collection_sources
from research.sandbox_a_resident.ct_stage2.bank import BANK_BYTES, describe_bank
from research.sandbox_a_resident.ct_stage2.config import expected_science
from research.sandbox_a_resident.ct_stage2.units import STUDY, empty_ledger
from tools.sandbox.registry import digest, world_seed

ROOT = Path(__file__).resolve().parents[3]
BASE_MINUTES = 360
EFFECTIVE_MINUTES = 720
ENDPOINT = 2560 * 6 * 8 * 300
BANK_ROOT = ROOT.parent / (
    "snake-dqn-artifacts/t34-ct-h5000-1029/revision-02/attempt-02/"
    "output-a100-fallback/exports/ct"
)
ARTIFACT_ROOT = ROOT.parent / f"snake-dqn-artifacts/{STUDY}/attempt-01"
JOB_ID = "sandbox-ct-h5000-stage2-20261030-01"


def source_paths(pin: str) -> list[str]:
    """Reuse frozen original runtime closure and add this explicit adapter closure."""
    additions = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", pin, "research/sandbox_a_resident/ct_stage2"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    additions = [
        p
        for p in additions
        if not Path(p).name.startswith("test_")
        and Path(p).name not in {"preflight.py", "prepare.py", "resume.py"}
    ]
    return sorted(set(collection_sources(pin)) | set(additions))


def write_once(path: Path, value: Any) -> None:
    """Never replace sealed packet or artifact bytes."""
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def staging_projection(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Real input sizes, historical conservative 0.75MB/s + per-file staging model."""
    from research.sandbox_a_resident.ct_resume import project_staging_envelope

    ledger = next(r for r in inputs if r["dest"] == "verified-ledger.json")
    report = project_staging_envelope(inputs, ledger_size_bytes=Path(ledger["path"]).stat().st_size)
    report.update(
        {
            "bank_payload_bytes": BANK_BYTES,
            "bank_is_input_not_reexported": True,
            "source_archive_assumption_bytes": 1_100_000,
            "stage_cap_bytes": 256 * 1024**2,
            "recursive_cap_bytes": 512 * 1024**2,
            "transfer_cap_bytes": 768 * 1024**2,
            "downstream_projected_bytes": 24 * 1024**2,
            "rationale": "126341023-byte bank input; ~22.4MiB original downstream forecast; "
            "caps allow serialization/diagnostics headroom without reexporting bank",
        }
    )
    return report


def prepare(packet: Path, pin: str, checkpoint_dir: Path) -> None:
    """Prepare only. Production admission rejects fixture or completion overrides."""
    pin = subprocess.check_output(["git", "rev-parse", pin], cwd=ROOT, text=True).strip()
    bank = describe_bank(BANK_ROOT)
    config = {
        "schema": "ct-stage2-control/v1",
        "study": STUDY,
        "source_commit": pin,
        "fixture": False,
        "bank": bank,
        "science": expected_science(pin),
    }
    packet.mkdir(parents=True, exist_ok=False)
    write_once(packet / "control.json", config)
    write_once(packet / "bank-verification.json", bank)
    write_once(packet / "verified-ledger.json", empty_ledger(config))
    inputs = []
    kinds = {}

    def upload(path: Path, dest: str, kind: str) -> None:
        inputs.append({"path": str(path.resolve()), "dest": dest, "sha256": digest(path)})
        kinds[dest] = kind

    for name, filename, kind in (
        ("b8-s3", "b8_s3_t7_20261016.pth", "frozen-checkpoint"),
        ("frp3-s12", "frp3_m3_s12_u60000_20261005.pth", "frozen-opponent"),
        ("champion", "champion_a5_freespace_20260621.pth", "frozen-opponent"),
    ):
        upload(checkpoint_dir / filename, name + ".pth", kind)
    upload(ROOT / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json", "registry.json", "registry")
    upload(packet / "verified-ledger.json", "verified-ledger.json", "verified-ledger")
    for suffix in ("pt", "json"):
        for extra in ("", ".sha256", ".complete"):
            filename = f"stage-0001.{suffix}{extra}"
            upload(
                BANK_ROOT / filename,
                "bank/" + filename,
                "ct-export-sidecar" if extra else "ct-export",
            )
    manifest = {
        "job_id": JOB_ID,
        "commit": pin,
        "script": "research/sandbox_a_resident/ct_stage2/job.sh",
        "device": "cuda",
        "source_paths": source_paths(pin),
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 2,
            "est_minutes": BASE_MINUTES,
        },
        "timeout_minutes": EFFECTIVE_MINUTES,
        "output_dir": str(ARTIFACT_ROOT / "output"),
        "max_export_bytes": 768 * 1024**2,
        "inputs": inputs,
        "env": {
            "SANDBOX_CT_CONTROL_JSON": json.dumps(config, sort_keys=True),
            "SANDBOX_A_SOURCE_COMMIT": pin,
            "SNAKE_DQN_DEVICE": "cuda",
            "MAX_JOBS": "1",
            "SANDBOX_A_STALL_TIMEOUT_SECONDS": "900",
            "COLAB_CLI_REQUIREMENTS": "research/sandbox_a_resident/requirements.txt",
        },
        "required_exports": ["ct/summary.json", "ct/stage-0010.pt", "ct/stage-0010.json"],
        "budget": {
            "rate_units_per_hour": 8.0,
            "unit_cap": 96.0,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "ct-stage2-snapshot",
            "ledger_path": "verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "sandbox_admission": {
            "schema": "standing-a100-development-sandbox-admission/v1",
            "manifest_path": str(packet / "sandbox.json"),
        },
        "sandbox_source_delivery": "background-sparse-v1",
        "sandbox_execution": "immutable H5000 bank -> original H300 4+4 C/T branches, "
        "three fits, registered held-out gates; simulator-D only",
        "requirements_file": "research/sandbox_a_resident/requirements.txt",
        "technical_envelope": {
            "base_seconds": 21600,
            "effective_seconds": 43200,
            "endpoint": f"at most {ENDPOINT} branch rows + 6000 fit minibatches; "
            "one frozen assessment",
        },
    }
    prepare_preference(manifest)
    write_once(packet / "dispatch-manifest.json", manifest)
    write_once(packet / "a100-fallback-preview.json", variants(manifest)[1])
    write_once(packet / "transfer-projection.json", staging_projection(inputs))
    sandbox = {
        "schema": "standing-a100-development-sandbox/v1",
        "study": STUDY,
        "inputs": [
            {"kind": "worlds", "study": STUDY, "indices": [0], "seeds": [world_seed(STUDY, 0)]}
        ]
        + [
            {"kind": kinds[row["dest"]], "path": row["path"], "sha256": row["sha256"]}
            for row in inputs
        ],
        "salt": f"a100-dev/{STUDY}/",
        "endpoint": {"kind": "steps", "value": ENDPOINT},
        "timeout": {"base_seconds": 21600, "seconds": 43200, "purpose": "runaway-guard"},
        "excluded_studies": [
            "F5",
            "F4-fold-0",
            "29-census-parents",
            "R30",
            "strict",
            "probe",
            "serving",
            "D4096",
            "t34-round0-study-01",
        ],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        "outputs": {
            "label": "simulator-D",
            "paths": [manifest["output_dir"]],
            "broker_manifest": {
                "path": str(packet / "dispatch-manifest.json"),
                "sha256": digest(packet / "dispatch-manifest.json"),
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }
    with (packet / "sandbox.json").open("x") as stream:
        stream.write(
            "\n".join(
                ("{" if i == 0 else " ")
                + json.dumps(k)
                + ": "
                + json.dumps(v)
                + ("," if i < 9 else "}")
                for i, (k, v) in enumerate(sandbox.items())
            )
            + "\n"
        )
    components = [p.name for p in packet.iterdir()]
    write_once(
        packet / "SEAL.json",
        {
            "schema": "ct-stage2-seal/v1",
            "source_commit": pin,
            "components": {n: digest(packet / n) for n in sorted(components)},
            "source_sha256": {
                name: hashlib.sha256(
                    subprocess.check_output(["git", "show", f"{pin}:{name}"], cwd=ROOT)
                ).hexdigest()
                for name in manifest["source_paths"]
            },
        },
    )
    print("PREPARED_CT_STAGE2 allocations=0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    prepare(args.packet.resolve(), args.commit, args.checkpoint_dir)


if __name__ == "__main__":
    main()
