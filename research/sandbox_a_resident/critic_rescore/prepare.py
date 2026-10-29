"""Seal a finite, shared-holdout critic diagnostic; never allocate or dispatch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from research.compute.gpu_preference import prepare_preference
from research.sandbox_a_resident.critic_rescore.run import empty_ledger, specs
from research.sandbox_a_resident.training_prepare import source_paths
from research.sandbox_a_resident.units import digest
from tools.sandbox.registry import load_registry, validate_input, verify_sources, world_seed

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT.parent / "snake-dqn-artifacts"


def write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def audit(control: dict, registry: Path) -> dict:
    """Check all actual new roots against the frozen exclusion inventory.

    Retain TRAIN-01's registered a100-dev domain, reserve a new remote index
    range above nine billion. Existing two-seed train/prep ranges are below
    one billion; fixed proxy roots are 400M/500M. Protected-bank seeds are
    represented by the Mac-built exclusion inventory, never opened here.
    """
    value = load_registry(registry)
    verify_sources(value)
    indices = [offset + i for _, _, offset in specs(control) for i in range(control["worlds"])]
    if min(indices) < 9_000_000_000 or len(set(indices)) != len(indices):
        raise ValueError("new disjoint critic-rescore range required")
    seeds = [world_seed("sandbox-a-train01", i) for i in indices]
    effective = [s & 0x7FFFFFFFFFFFFFFF for s in seeds]
    excluded = {int(s) & 0x7FFFFFFFFFFFFFFF for s in value["excluded_seeds"]}
    if 0 in effective or len(set(effective)) != len(effective) or set(effective) & excluded:
        raise ValueError("root collision or excluded seed intersection")
    return dict(
        status="PASS",
        namespace="a100-dev/sandbox-a-train01",
        indices=indices,
        seeds=seeds,
        effective_bits=63,
        aliases=0,
        excluded_intersection=0,
        registry_sha256=digest(registry),
        coverage="Frozen sandbox exclusions plus disjoint canonical TRAIN-01 index ranges; "
        "no protected payload access; opaque unregistered external runs unknown",
    )


def prepare(packet: Path, commit: str) -> None:
    checkpoint_dir = ARTIFACTS.parent / "snake-dqn" / "saved_snakes"
    registry = ROOT / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    sources = [
        ("b8-s3.pth", checkpoint_dir / "b8_s3_t7_20261016.pth", "frozen-checkpoint"),
        ("frp3-s12.pth", checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth", "frozen-opponent"),
        ("champion.pth", checkpoint_dir / "champion_a5_freespace_20260621.pth", "frozen-opponent"),
        ("registry.json", registry, "registry"),
    ]
    for seed, directory in [
        (1, ARTIFACTS / "sandbox-a-resident/train01-attempt-11/output"),
        (2, ARTIFACTS / "sandbox-a-resident/train01-s2-attempt-04/output-a100-fallback"),
    ]:
        source = directory / "exports/training/iteration-0000.pt.xz"
        sources.extend(
            [
                (f"seed{seed}.pt.xz", source, "prepared-critic-snapshot"),
                (
                    f"seed{seed}.manifest",
                    Path(str(source) + ".manifest"),
                    "prepared-critic-snapshot",
                ),
            ]
        )
    for _, path, kind in sources:
        validate_input(dict(kind=kind, path=str(path), sha256=digest(path)))
    hashes = {name: digest(path) for name, path, _ in sources}
    control = dict(
        schema="critic-rescore-control/v1",
        worlds=96,
        horizon=5000,
        holdout_offset=9_000_000_000,
        baseline_offset=9_100_000_000,
        max_waves=32,
        late_length=800,
        late_lives_per_mix=8,
        test_only=False,
        input_hashes=hashes,
        policy="released-B8-s3 masked greedy; no veto, no updates",
        base_max_waves=16,
        effective_max_waves=32,
        source_commit=commit,
    )
    check = audit(control, registry)
    packet.mkdir(parents=True, exist_ok=False)
    write(packet / "control.json", control)
    write(packet / "seed-audit.json", check)
    write(packet / "verified-ledger.json", empty_ledger(control))
    sources.append(("verified-ledger.json", packet / "verified-ledger.json", "verified-ledger"))
    uploads = [dict(dest=n, path=str(p.resolve()), sha256=digest(p)) for n, p, _ in sources]
    own = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "research/sandbox_a_resident/critic_rescore",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    runtime = sorted(
        set(source_paths(commit))
        | {p for p in own if p.endswith((".py", ".sh")) and not Path(p).name.startswith("test_")}
    )
    output = ARTIFACTS / "sandbox-a-resident/critic-rescore-20261029-attempt-01/output"
    manifest = dict(
        job_id="critic-rescore-20261029-01",
        commit=commit,
        script="research/sandbox_a_resident/critic_rescore/job.sh",
        device="cuda",
        source_paths=runtime,
        requirements=dict(gpu="A100", allowed_gpus=["A100"], min_vcpu=2, est_minutes=240),
        timeout_minutes=480,
        output_dir=str(output),
        max_export_bytes=1024**3,
        requirements_file="research/sandbox_a_resident/requirements.txt",
        inputs=uploads,
        env=dict(
            SANDBOX_A_CONTROL_JSON=json.dumps(control, sort_keys=True),
            SANDBOX_A_SOURCE_COMMIT=commit,
            SNAKE_DQN_DEVICE="cuda",
            MAX_JOBS="1",
            COLAB_CLI_REQUIREMENTS="research/sandbox_a_resident/requirements.txt",
        ),
        required_exports=["rescore/collection.json.gz"],
        budget=dict(rate_units_per_hour=8.0, unit_cap=64.0, reserve_units=0, other_assignments={}),
        resume=dict(
            unit_kind="complete-shared-holdout-wave",
            ledger_path="verified-ledger.json",
            skip_verified=True,
            max_outcome_blind_retries=2,
        ),
        sandbox_admission=dict(
            schema="standing-a100-development-sandbox-admission/v1",
            manifest_path=str(packet / "sandbox.json"),
        ),
        sandbox_execution="frozen critic shared-holdout simulator-D diagnostic; zero updates",
        technical_envelope=dict(
            base_seconds=14400,
            effective_seconds=28800,
            endpoint="24 late completed lives; 8 per mix; at most 32 E96 waves + E96 baseline",
        ),
    )
    prepare_preference(manifest)
    write(packet / "dispatch-manifest.json", manifest)
    inputs = [
        dict(
            kind="worlds", study="sandbox-a-train01", indices=check["indices"], seeds=check["seeds"]
        )
    ]
    inputs += [dict(kind=k, path=str(p.resolve()), sha256=digest(p)) for _, p, k in sources]
    sandbox = dict(
        schema="standing-a100-development-sandbox/v1",
        study="sandbox-a-train01",
        inputs=inputs,
        salt="a100-dev/sandbox-a-train01/",
        endpoint=dict(kind="steps", value=33 * 96 * 5000),
        timeout=dict(base_seconds=14400, seconds=28800, purpose="runaway-guard"),
        excluded_studies=[
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
        owner_rule="docs/research/governance_a100_sandbox_20261026/ADOPTED.md",
        outputs=dict(
            label="simulator-D",
            paths=[str(output)],
            broker_manifest=dict(
                path=str(packet / "dispatch-manifest.json"),
                sha256=digest(packet / "dispatch-manifest.json"),
            ),
        ),
        crossing=dict(max_frozen_checkpoint_shas=1),
    )
    with (packet / "sandbox.json").open("x") as stream:
        for i, (k, v) in enumerate(sandbox.items()):
            stream.write(
                ("{" if i == 0 else " ")
                + json.dumps(k)
                + ": "
                + json.dumps(v)
                + ("," if i < 9 else "}")
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    a = parser.parse_args()
    prepare(a.packet.resolve(), a.commit)
