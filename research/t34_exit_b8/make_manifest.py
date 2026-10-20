"""Prepare T34 ExIt-B8 manifests without allocating a Colab runtime.

Public-release transport is intentionally blocked until the owner supplies all
six immutable URLs.  The optional local-upload form is only useful for Mac
supervisor validation; it is never dispatch-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/t34_exit_b8"
INVENTORY_PATH = Path(__file__).with_name("INPUTS_REQUIRED.json")
PUBLIC_CATALOG_PATH = Path(__file__).with_name("PUBLIC_INPUTS.json")
EXPECTED_DESTINATIONS = {
    "B8_s3_final.pth",
    "checkpoints/champion_a5_freespace_20260621.pth",
    "checkpoints/best_apex_fs.pth",
    "checkpoints/best_apex_stage1_fs.pth",
    "checkpoints/best_apex_pre_fs.pth",
    "checkpoints/frp3_m3_s12_u60000_20261005.pth",
}


class ManifestRefused(ValueError):
    """A draft cannot safely become a supervisor job."""


def sha256_file(path: Path) -> str:
    """Return a bounded-memory SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_head(repo: Path = ROOT) -> str:
    """Read the source commit that a manifest must pin."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rows(path: Path, *, local: bool) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestRefused(f"invalid input catalog: {path}") from exc
    rows = value.get("inputs") if isinstance(value, dict) else None
    required = {"name", "dest", "size_bytes", "sha256"}
    if local:
        required.add("local_path")
    if (
        not isinstance(rows, list)
        or len(rows) != 6
        or any(not isinstance(row, dict) or not required.issubset(row) for row in rows)
    ):
        raise ManifestRefused("T34 needs exactly six complete checkpoint identities")
    if {row["dest"] for row in rows} != EXPECTED_DESTINATIONS:
        raise ManifestRefused("T34 input destinations differ from the registered T33 identities")
    return rows


def _same_identities(local: list[dict[str, Any]], public: list[dict[str, Any]]) -> bool:
    fields = ("name", "dest", "size_bytes", "sha256")
    return {tuple(row[field] for field in fields) for row in local} == {
        tuple(row[field] for field in fields) for row in public
    }


def upload_inputs(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Verify local immutable inputs for a local-only supervisor preflight."""
    result: list[dict[str, str]] = []
    for row in rows:
        source = Path(str(row["local_path"])).expanduser().resolve()
        if not source.is_file():
            raise ManifestRefused(f"missing local input: {source}")
        if source.stat().st_size != row["size_bytes"] or sha256_file(source) != row["sha256"]:
            raise ManifestRefused(f"input bytes drifted: {source}")
        result.append({"path": str(source), "sha256": str(row["sha256"]), "dest": str(row["dest"])})
    return result


def _public_catalog_input(path: Path) -> dict[str, str]:
    source = path.resolve(strict=True)
    return {"path": str(source), "sha256": sha256_file(source), "dest": "public-inputs.json"}


def make_manifest(
    *,
    kind: str,
    commit: str,
    output_dir: Path,
    inventory_path: Path = INVENTORY_PATH,
    public_catalog_path: Path = PUBLIC_CATALOG_PATH,
    local_upload_preflight: bool = False,
    round_id: str = "a5",
    shard: int = 0,
    attempt: int = 0,
    worlds: int | None = None,
    policy_factory: str | None = None,
    checkpoint: str = "B8_s3_final.pth",
    student_checkpoint: str | None = None,
    triggers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an A5 or label draft, refusing unsafe identities and namespaces."""
    if kind not in {"benchmark", "label"}:
        raise ManifestRefused("kind must be benchmark or label")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        raise ManifestRefused("output_dir must be an absolute fresh Mac directory")
    if checkpoint != "B8_s3_final.pth" or student_checkpoint:
        raise ManifestRefused(
            "student input publication and SHA binding are pending; only pinned B8 is supported"
        )
    if policy_factory and (":" not in policy_factory or any(c.isspace() for c in policy_factory)):
        raise ManifestRefused("policy_factory must be module:callable")
    trigger_values = {
        "confirm_interval": 8,
        "confirm_min_length": 600,
        "low_length_probability": 0.03,
        "death_window": 150,
        "max_rewinds": 2,
    }
    trigger_values.update(triggers or {})
    # This pure builder must not import the simulator or initialize CPU pools.
    if set(trigger_values) != {
        "confirm_interval",
        "confirm_min_length",
        "low_length_probability",
        "death_window",
        "max_rewinds",
    }:
        raise ManifestRefused("unknown trigger setting")
    integer_keys = ("confirm_interval", "confirm_min_length", "death_window", "max_rewinds")
    if any(type(trigger_values[key]) is not int for key in integer_keys):
        raise ManifestRefused("trigger counters must be integers")
    if (
        trigger_values["confirm_interval"] < 1
        or trigger_values["confirm_min_length"] < 0
        or trigger_values["death_window"] < 1
        or not 0 <= trigger_values["max_rewinds"] <= 2
        or not 0 <= trigger_values["low_length_probability"] <= 1
    ):
        raise ManifestRefused("invalid trigger limits")
    local = _rows(inventory_path, local=True)
    if local_upload_preflight:
        inputs = upload_inputs(local)
        input_mode, claim = "local-preflight", "LOCAL_UPLOAD_PREFLIGHT_ONLY"
        transport = {
            "mode": "local_upload_preflight",
            "status": "LOCAL_ONLY_NOT_A_PRODUCTION_CLAIM",
            "dispatch_ready": False,
        }
    else:
        public = _rows(public_catalog_path, local=False)
        if not _same_identities(local, public):
            raise ManifestRefused("public catalog does not exactly bind the T34 inventory")
        ready = all(
            isinstance(row.get("public_url"), str) and row["public_url"].startswith("https://")
            for row in public
        )
        if ready:
            # The runtime consumes asset URLs (not the per-member convenience
            # fields), so validate that exact archive transport before calling
            # a draft dispatch-ready.
            from research.t34_exit_b8.public_inputs import _load

            _load(public_catalog_path)
        inputs = [_public_catalog_input(public_catalog_path)]
        input_mode, claim = "public", "DRAFT_NO_ALLOCATION_AUTHORIZED"
        transport = {
            "mode": "public_release_urls",
            "status": "PUBLICATION_READY" if ready else "PUBLICATION_BLOCKED",
            "dispatch_ready": ready,
            "publishing_pending": not ready,
        }
    if shard < 0 or attempt < 0:
        raise ManifestRefused("shard and attempt must be nonnegative")
    if kind == "benchmark":
        script, job_id, estimate, cap = (
            f"{PACKAGE}/bench.sh",
            f"t34-exit-b8-a5-s{shard}-a{attempt}",
            15,
            1.0,
        )
        env = {
            "T34_WORLDS": "256",
            "T34_MAX_WORLDS": "256",
            "T34_FRAMES": "5000",
            "T34_SECONDS": "600",
            "T34_NAMESPACE": f"t34-exit-a5/s{shard}/a{attempt}",
        }
    else:
        if shard < 0 or attempt < 0 or not round_id.replace("-", "").replace("_", "").isalnum():
            raise ManifestRefused("round, shard, and attempt must form a safe unique namespace")
        if worlds is None or not 1 <= worlds <= 10000:
            raise ManifestRefused("label worlds must be a bounded positive chunk count")
        script, job_id, estimate, cap = (
            f"{PACKAGE}/label.sh",
            f"t34-exit-b8-{round_id}-s{shard}-a{attempt}",
            180,
            12.0,
        )
        env = {
            "T34_WORLDS": str(worlds),
            "T34_MAX_WORLDS": str(worlds),
            "T34_FRAMES": "5000",
            "T34_SECONDS": "10200",
            "T34_ROUND": round_id,
            "T34_SHARD": str(shard),
            "T34_ATTEMPT": str(attempt),
            "T34_NAMESPACE": f"t34-exit/{round_id}/s{shard}/a{attempt}",
            "T34_BETA": {"r2": "0.5", "r3": "0"}.get(round_id.lower(), "1"),
            "T34_CONFIRM_INTERVAL": "8",
        }
    return {
        "job_id": job_id,
        "commit": commit,
        "script": script,
        "claim": claim,
        "transport": transport,
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_INPUT_MODE": input_mode,
            "T34_DEVICE": "cuda",
            "T34_WORKERS": "2",
            "T34_THREADS": "5",
            "NUMBA_NUM_THREADS": "5",
            "T34_HORIZON": "300",
            "T34_TELEMETRY_SECONDS": "120",
            **env,
            **{f"T34_{key.upper()}": str(value) for key, value in trigger_values.items()},
            **({"T34_POLICY_FACTORY": policy_factory} if policy_factory else {}),
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": 360 if kind == "label" else 30,
        "output_dir": str(output_dir),
        "inputs": inputs,
        "budget": {"unit_cap": cap, "rate_units_per_hour": 2.0, "reserve_units": 0.0},
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
    }


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Run only the supervisor's pure local validator."""
    if manifest.get("transport", {}).get("dispatch_ready") is False:
        raise ManifestRefused("public input publication is pending; manifest is not dispatch-ready")
    from research.colab_cli.supervisor import validate

    return validate(manifest, repo=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("benchmark", "label"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--public-catalog", type=Path, default=PUBLIC_CATALOG_PATH)
    parser.add_argument("--local-upload-preflight", action="store_true")
    parser.add_argument("--round", default="a5")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--policy-factory")
    parser.add_argument("--checkpoint", default="B8_s3_final.pth")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--triggers", type=json.loads, help="JSON object of trigger overrides")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    manifest = make_manifest(
        kind=args.kind,
        commit=args.commit or git_head(),
        output_dir=args.output_dir,
        inventory_path=args.inventory,
        public_catalog_path=args.public_catalog,
        local_upload_preflight=args.local_upload_preflight,
        round_id=args.round,
        shard=args.shard,
        attempt=args.attempt,
        worlds=args.worlds,
        policy_factory=args.policy_factory,
        checkpoint=args.checkpoint,
        student_checkpoint=args.student_checkpoint,
        triggers=args.triggers,
    )
    if args.validate:
        validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest": str(args.out), "validated": args.validate}, sort_keys=True))


if __name__ == "__main__":
    main()
