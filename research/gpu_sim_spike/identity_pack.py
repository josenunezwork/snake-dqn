"""Mac-side packet for the GPU-vs-CPU closed-loop identity falsifier.

``freeze`` (before any data) verifies the harness commit, audits the world
seeds, writes the pre-registration, ledgers, one metadata tar and the native
compile proof + A100 canary manifests.  ``bind-compile`` and ``bind-canary``
add the hash bindings that the dispatch gates require, after each terminal
receipt exists.  Nothing here allocates, dispatches or reads an outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import identity_compare as compare
from research.gpu_sim_spike import identity_tape as tape

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
EXP = ARTIFACTS / "big-lever/exp"
PACKAGE_DIR = EXP / "gpu-identity-20261024"
QUALIFIED = Path("/Users/josenunez/Projects/ml/snake-dqn-gpuint-qualified-03.noindex")
RUNTIME_PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
QUALIFIED_TRANSPORT_TREE = "d0d92bdb572b770284a1fea6152417ffa5f06bea"
PUBLIC_INPUTS = {
    "dest": "public-inputs.json",
    "path": str(QUALIFIED / "research/t34_exit_b8/PUBLIC_INPUTS.json"),
    "sha256": "736f7b4e7730f760b8b81db00772d14abcf84c74b261a8dc7630732fdd383a10",
}
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
RULINGS = (
    (
        EXP / "JUDGMENT_POST_I0S_DIRECTION_20261024.md",
        "efad89757569070c1c8f332d4cd9c3d7937172f22456e1822f9c00926eb0e9c3",
    ),
    (
        EXP / "ADVICE_POST_I0S_B_20261024.md",
        "0f6631a243dc51098efbe9e7def7037cdc3191172bd89c9db9f904856bbbb5e6",
    ),
    (
        EXP / "JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
        "51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b",
    ),
    (
        ARTIFACTS / "big-lever/round-28/JUDGMENT.md",
        "89996ae9190165aae08036525d3d1b5dbedf1bbfdbd5fd5f1d2b888b13e925dd",
    ),
)
# The qualified I0-03 L4 compile proof; its source inventory is reused verbatim.
I0_COMPILE_MANIFEST = (
    EXP / "gpu-integration-i0-20261024-03/packet/native-compile/native-compile.json",
    "dbca05b6bfff32d6b0a14ce4036c3438e61149c50338126dd92424873dc5d62f",
)
SOURCE_PATHS = [
    "src",
    "web/backend",
    "configs",
    "research/gpu_sim_spike",
    "research/colab_cli",
    "research/bl_upload",
    "research/t34_exit_b8",
    "research/speed_s0_baseline",
    "research/t34_shadow",
    "research/t33_batched_teacher",
    "research/phase0_search_20261019",
    "research/redesign_scope_20261007",
    "research/t3_rollout_labels_20261015",
    "research/t1_commitment_20261015",
    "research/apex_safety_20260926",
    "research/b8_death_census_20261014",
    "research/fixed_search.py",
    "research/metadata_archive.py",
    "requirements.txt",
    "pyproject.toml",
    "research/t34_exit_b8/public_inputs.py",
    "research/t34_exit_b8/PUBLIC_INPUTS.json",
]
# Owner 2x rule applied once to timeouts/caps/reserves; estimates are forecasts.
# A100 at the priced 10 u/h ceiling; the L4 compile proof at I0's 3.0 u/h price.
ENVELOPE = {
    "ruling": "step 3: est 3-6 u, cap 10 u (2x rule), A100, canary first",
    "compile": {
        "gpu": "L4",
        "est_minutes": 12,
        "timeout_minutes": 24,
        "rate": 3.0,
        "unit_cap": 1.2,
        "reserve_units": 0.2,
    },
    "canary": {
        "gpu": "A100",
        "est_minutes": 8,
        "timeout_minutes": 16,
        "rate": 10.0,
        "unit_cap": 2.67,
        "reserve_units": 0.45,
        "producer_seconds": 600,
        "workers": 2,
    },
    "main": {
        "gpu": "A100",
        "est_minutes": 18,
        "timeout_minutes": 36,
        "rate": 10.0,
        "unit_cap": 6.0,
        "reserve_units": 1.0,
        "producer_seconds": 1740,
        "workers": 4,
    },
    "total_effective_cap_units": 9.87,
    "base_units": {"compile": 0.6, "canary": 1.33, "main": 3.0, "total": 4.93},
    "measured_basis": {
        "compile": "I0-03 L4 native compile receipt elapsed 825 s (~14 min)",
        "canary": "I0 A100 canary receipt elapsed 537 s incl. JIT compile; observed 5.3 u/h",
        "main": "12 worlds x 1000 ticks on 4 processes; I0S E6 tick ~0.14 s (padding floor)",
    },
    "compile_note": (
        "The ruling says 'no new compile'. Engine/native bytes are unchanged, but the "
        "dispatch A100-canary gate binds a native-compile PASS receipt to the manifest "
        "commit, and the harness commit differs from 968f0276. Dispatch is HELD until a "
        "ruling admits the L4 compile proof (or another lawful route)."
    ),
}


class PackRefused(RuntimeError):
    """The packet cannot be frozen without violating the ruling or a gate."""


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=60).strip()


def sha256_file(path: Path) -> str:
    return tape.sha256_file(path)


def check_harness_commit(commit: str) -> dict[str, Any]:
    """The harness commit adds new files only on top of the runtime pin."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PackRefused("harness commit must be a full lowercase SHA")
    parents = _git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents != [RUNTIME_PIN]:
        raise PackRefused("harness commit must be a direct child of the runtime pin")
    changes = _git("diff", "--name-status", RUNTIME_PIN, commit).splitlines()
    added = []
    for line in changes:
        status, _, name = line.partition("\t")
        if status != "A":
            raise PackRefused(f"harness commit modifies an existing file: {line}")
        if not re.fullmatch(
            r"research/gpu_sim_spike/(test_identity|identity_[a-z_]+)\.(py|sh)", name
        ):
            raise PackRefused(f"harness commit adds a file outside the identity harness: {name}")
        added.append(name)
    if _git("rev-parse", f"{commit}:research/colab_cli") != QUALIFIED_TRANSPORT_TREE:
        raise PackRefused("harness transport tree differs from the qualified tree")
    remote = _git("branch", "-r", "--contains", commit)
    if "origin/" not in remote:
        raise PackRefused("harness commit must be published before freezing")
    return {
        "commit": commit,
        "parent": RUNTIME_PIN,
        "added_files": {
            name: hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            ).hexdigest()
            for name in added
        },
        "engine_and_native_bytes": "identical to the runtime pin (diff is additions only)",
        "transport_tree": QUALIFIED_TRANSPORT_TREE,
        "published_refs": sorted({ref.strip() for ref in remote.splitlines()}),
    }


# --------------------------------------------------------------------------- seeds


REGISTERED_BANKS = {
    "round0": "research/t34_round0/BANK_ROUND0.json",
    "round0d": "research/t34_round0d/BANK_ROUND0D.json",
    "retention_v2": "research/t34_pathb/RETENTION_BANK_V2.json",
}
SCAN_EXCLUDES = ("t34-round0-study-01", "mac-lease.json*", "gpu-identity-20261024")


def seed_audit(scan: bool = True) -> dict[str, Any]:
    """Registered-bank intersection plus a whole-history word scan; no raw seeds."""
    from research.t34_claim_a.seed_audit import _bank_seeds, seed_digest

    seeds = {mode: [int(row["world_seed"]) for row in tape.world_rows(mode)] for mode in tape.MODES}
    everything = [seed for values in seeds.values() for seed in values]
    if len(set(everything)) != len(everything):
        raise PackRefused("identity namespaces collide internally")
    banks = {}
    for name, relative in REGISTERED_BANKS.items():
        path = ROOT / relative
        values = _bank_seeds(json.loads(path.read_text()))
        if not values:
            raise PackRefused(f"no seeds parsed from {relative}")
        hits = set(everything) & values
        if hits:
            raise PackRefused(f"identity seeds collide with {name}")
        banks[name] = {
            "path": relative,
            "sha256": sha256_file(path),
            "bank_seeds": len(values),
            "collisions": 0,
        }
    history = None
    if scan:
        history = _history_scan(everything)
    return {
        "schema": "gpu-identity-seed-audit/v1",
        "status": "PASS",
        "domain": tape.DOMAIN,
        "namespaces": {
            tape.MODES[mode]["namespace"]: {"count": len(values), "sha256": seed_digest(values)}
            for mode, values in seeds.items()
        },
        "registered_banks": banks,
        "history_scan": history,
        "note": "raw seeds omitted; recompute with identity_tape.world_rows(mode)",
    }


def _history_scan(candidates: list[int]) -> dict[str, Any]:
    roots = [ROOT / "docs", ROOT / "research", ROOT / "src", ROOT / "scripts", ARTIFACTS]
    command = [
        "rg",
        "--threads",
        "1",
        "--hidden",
        "--no-ignore",
        "--fixed-strings",
        "--word-regexp",
        "--only-matching",
        "--no-heading",
        "--no-messages",
    ]
    for pattern in ("*.json", "*.jsonl", "*.md", "*.py", "*.yaml", "*.yml", "*.sh"):
        command += ["--glob", pattern]
    for excluded in SCAN_EXCLUDES:
        command += ["--glob", f"!{excluded}"]
    for seed in candidates:
        command += ["-e", str(seed)]
    hits = {}
    for root in roots:
        if not root.is_dir():
            continue
        result = subprocess.run([*command, str(root)], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise PackRefused(f"freshness scan failed for {root}")
        found = {int(line.rsplit(":", 1)[-1]) for line in result.stdout.splitlines() if line}
        hits[str(root)] = len(found & set(candidates))
    if any(hits.values()):
        raise PackRefused(f"identity seeds appear in prior history: {hits}")
    exclusions = sorted(ARTIFACTS.glob("frp-*/namespaces*/training_exclusions.json"))
    for path in exclusions:
        if set(candidates) & set(json.loads(path.read_text())):
            raise PackRefused(f"identity seeds collide with FRP exclusions: {path}")
    return {
        "method": "rg word scan, single thread",
        "roots": [str(root) for root in roots if root.is_dir()],
        "excluded_globs": list(SCAN_EXCLUDES),
        "hits": hits,
        "frp_exclusion_files": len(exclusions),
    }


# --------------------------------------------------------------------------- registration


def preregistration(commit: str, harness: dict[str, Any], rulings: list[dict[str, str]]) -> dict:
    return {
        "schema": "gpu-identity-preregistration/v1",
        "frozen_before_data": True,
        "authority": rulings,
        "runtime_pin": RUNTIME_PIN,
        "harness": harness,
        "engine": {
            "arm": "I (indexed MOVE, native vector61 geometry, static flood, v8 indexed OFF)",
            "env": {
                "GPU_SIM_VECTOR_FLOOD": "static",
                "GPU_SIM_VECTOR_GEOMETRY": "native",
                "GPU_SIM_V8_INDEXED": "0",
            },
            "precision": "fp32; torch.backends.{cuda.matmul,cudnn}.allow_tf32 = False",
            "batch": "each world replicated to batch width E12; replicas must agree bitwise",
            "native_proof": "move_cuda and vector_geometry extensions loaded (no fallback)",
        },
        "worlds": {
            mode: {
                "namespace": spec["namespace"],
                "count": spec["worlds"],
                "mixes": list(spec["mixes"]),
                "horizon": spec["horizon"],
                "replicas": spec["replicas"],
                "generator": "trajectory.make_world_rows (dev_screen seeds, domain "
                + tape.DOMAIN
                + ")",
                "scored": mode == "main",
            }
            for mode, spec in tape.MODES.items()
        },
        "cpu_authority": (
            "trajectory.build_world + prepare_world; hero ExitTeacher(cpu).evaluate greedy "
            "(bare B8, fp32, batch 1); opponents ExitTeacher.actions (T3 dispatcher, frozen v8 "
            "hooks); sim.step; runtime.observe_step. Device actions are never replayed."
        ),
        "seam": "prepared selector seam (after frame, food maintenance, respawn)",
        "tier_L_identical_means": {
            "order": ["replicas", "state", "hero_mask", "hero_action", "opponent_actions"],
            "state_fields": list(tape.SNAKE_FIELDS),
            "state_extra": [
                "live logical body (head first) SHA-256 for alive snakes",
                "frame",
                "ordered food list with corpse flags (exact order, no sorting)",
            ],
            "dead_snake_masked": list(tape.DEAD_MASKED_FIELDS) + ["body"],
            "opponents_compared": "alive at the seam; dead opponents' actions are inert",
            "integer_widening": "int64 widening only; no rounding, sorting or dedup",
        },
        "tier_D_recorded_only": [
            "hero Q float32 bytes",
            "hero observation (local, global, scalars) digests",
            "opponent Q float32 bytes, raw advisory masks, vector61 state digests",
            "_last_reward (CPU float64 vs device float32: REWARD_PRECISION)",
        ],
        "classes": compare.CLASSES,
        "explained_classes": sorted(compare.EXPLAINED),
        "defect_rule": "an unexplained (non-RNG/tie/float/inherited-order) divergence is "
        "also recorded as a device-correctness defect candidate",
        "outcomes": {
            "OUTCOME_A_IDENTICAL": "all 12 main worlds identical through H1000 or the same "
            "hero death; permits Q5-main/Q6 to go to a new ruling",
            "OUTCOME_B_IDENTITY_FAILS": "any Tier-L divergence in any world, including an "
            "exact float tie broken differently; round-28 section 3 closure applies",
            "INCOMPLETE": "missing world without any divergence",
            "INVALID": "integrity failure, start-state mismatch or native route not loaded",
            "precedence": "INVALID > B > INCOMPLETE > A",
        },
        "diagnostics_not_deciding": {
            "rng_bridged": "CPU regenerated with device food/respawn draws supplied; reports "
            "the first non-RNG divergence",
            "natural_pass_stop": "one transition after the first divergence",
        },
        "single_look": "identity_compare writes IDENTITY_REPORT.json once (create-only)",
        "harness_commit": commit,
    }


def _rulings() -> list[dict[str, str]]:
    rows = []
    for path, expected in RULINGS:
        if sha256_file(path) != expected:
            raise PackRefused(f"binding document hash differs: {path}")
        rows.append({"path": str(path), "sha256": expected})
    return rows


def _ledger(mode: str, commit: str, path: Path) -> dict[str, str]:
    tape.publish_json(path, tape.empty_ledger(tape.namespace(mode, commit), tape.allocation(mode)))
    return {"path": str(path), "sha256": sha256_file(path)}


def _manifest(
    *,
    commit: str,
    mode: str,
    output: Path,
    job_id: str,
    metadata: dict[str, str],
    ledger_path: str,
) -> dict[str, Any]:
    spec = ENVELOPE[mode]
    env = {
        "BL_PUBLIC_INPUTS": "1",
        "BL_PUBLIC_INPUT_ALIASES": "{}",
        "COLAB_CLI_REQUIREMENTS": "research/gpu_sim_spike/integration_requirements.txt",
        "GPU_SIM_SPIKE_COMMIT": commit,
        "GPU_SIM_V8_INDEXED": "0",
        "GPU_SIM_VECTOR_FLOOD": "static",
        "GPU_SIM_VECTOR_GEOMETRY": "native",
        "IDENTITY_DEVICE": "cuda",
        "IDENTITY_LEDGER_PATH": ledger_path,
        "IDENTITY_MODE": mode,
        "IDENTITY_SECONDS": str(spec["producer_seconds"]),
        "IDENTITY_WORKERS": str(spec["workers"]),
        "METADATA_ARCHIVE_SHA256": metadata["sha256"],
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    value: dict[str, Any] = {
        "budget": {
            "other_assignments": {},
            "rate_units_per_hour": spec["rate"],
            "reserve_units": spec["reserve_units"],
            "unit_cap": spec["unit_cap"],
        },
        "claim": "Engineering identity tapes only; no hero, label, release or serving claim.",
        "commit": commit,
        "env": env,
        "identity_envelope": {**spec, "series": ENVELOPE["total_effective_cap_units"]},
        "inputs": [dict(PUBLIC_INPUTS), {"dest": "metadata.tar", **metadata}],
        "job_id": job_id,
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024 * 1024,
        "output_dir": str(output),
        "required_exports": ["identity-device.json"],
        "requirements": {
            "allowed_gpus": ["A100"],
            "est_minutes": spec["est_minutes"],
            "gpu": "A100",
            "min_vcpu": 12,
        },
        "resume": {
            "ledger_path": ledger_path,
            "max_outcome_blind_retries": 2,
            "skip_verified": True,
            "unit_kind": "world-tape",
        },
        "scientific_admission": True,
        "script": "research/gpu_sim_spike/identity_device.sh",
        "source_paths": list(SOURCE_PATHS),
        "timeout_minutes": spec["timeout_minutes"],
        "transport_pin": TRANSPORT_PIN,
    }
    if mode == "canary":
        env["SNAKE_COLAB_CANARY"] = "1"
        env["SNAKE_COLAB_CANARY_PACKAGE"] = "research/gpu_sim_spike"
        value["canary"] = {
            "family": "research/gpu_sim_spike/identity_device.sh",
            "mode": "probe",
            "schema": "colab-family-canary/v1",
        }
    return value


def compile_manifest(commit: str, output: Path, requirements: dict[str, str]) -> dict[str, Any]:
    """The I0-03 L4 native compile proof shape, rebound to the harness commit."""
    spec = ENVELOPE["compile"]
    path, expected = I0_COMPILE_MANIFEST
    if sha256_file(path) != expected:
        raise PackRefused("qualified I0-03 compile manifest differs")
    sources = json.loads(path.read_text())["source_paths"]
    for name in sources:
        if _git("rev-parse", f"{commit}:{name}") != _git("rev-parse", f"{RUNTIME_PIN}:{name}"):
            raise PackRefused(f"compile source differs from the runtime pin: {name}")
    return {
        "budget": {
            "other_assignments": {},
            "rate_units_per_hour": spec["rate"],
            "reserve_units": spec["reserve_units"],
            "unit_cap": spec["unit_cap"],
        },
        "canary": {
            "family": "research/gpu_sim_spike/native_compile.sh",
            "mode": "probe",
            "schema": "colab-family-canary/v1",
        },
        "commit": commit,
        "compile_envelope": {**spec, "note": ENVELOPE["compile_note"]},
        "device": "cpu",
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/gpu_sim_spike/native_compile_requirements.txt",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/gpu_sim_spike",
        },
        "inputs": [{"dest": "compile-requirements.txt", **requirements}],
        "job_id": f"native-compile-l4-{commit[:8]}-identity",
        "lease_seconds": 600,
        "max_export_bytes": 1024 * 1024,
        "output_dir": str(output),
        "required_exports": ["native-compile.json"],
        "requirements": {
            "allowed_gpus": ["L4"],
            "est_minutes": spec["est_minutes"],
            "gpu": "L4",
            "min_vcpu": 2,
        },
        "resume_exemption": {
            "kind": "short-canary",
            "reason": "Bounded host compile/link only; no outcomes, gameplay or trajectories",
        },
        "script": "research/gpu_sim_spike/native_compile.sh",
        "source_paths": list(sources),
        "timeout_minutes": spec["timeout_minutes"],
    }


def _write_json_once(path: Path, value: dict[str, Any]) -> str:
    return tape.publish_json(path, value)


def freeze(commit: str, package: Path, *, scan: bool = True) -> dict[str, Any]:
    from research.metadata_archive import bundle_inputs

    harness = check_harness_commit(commit)
    rulings = _rulings()
    packet = package / "packet"
    if packet.exists():
        raise PackRefused(f"packet already frozen: {packet}")
    audit = seed_audit(scan=scan)
    _write_json_once(packet / "SEED_AUDIT.json", audit)
    prereg = preregistration(commit, harness, rulings)
    _write_json_once(packet / "PREREGISTRATION.json", prereg)
    staging = packet / "metadata-staging"
    rows = []
    for mode, prefix in (("main", "inputs"), ("canary", "canary-inputs")):
        ledger = _ledger(mode, commit, staging / prefix / "verified-ledger.json")
        for suffix in ("", ".sha256", ".complete"):
            path = Path(ledger["path"] + suffix)
            rows.append(
                {
                    "dest": f"{prefix}/verified-ledger.json{suffix}",
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    for name in ("PREREGISTRATION.json", "SEED_AUDIT.json"):
        rows.append(
            {
                "dest": f"inputs/{name}",
                "path": str(packet / name),
                "sha256": sha256_file(packet / name),
            }
        )
    direction = Path(rulings[0]["path"])
    rows.append(
        {
            "dest": f"inputs/rulings/{direction.name}",
            "path": str(direction),
            "sha256": rulings[0]["sha256"],
        }
    )
    bundled = bundle_inputs(rows, packet / "metadata.tar")
    metadata = {key: bundled[-1][key] for key in ("path", "sha256")}
    requirements_path = QUALIFIED / "research/gpu_sim_spike/native_compile_requirements.txt"
    requirements = {"path": str(requirements_path), "sha256": sha256_file(requirements_path)}
    if _git("show", f"{commit}:research/gpu_sim_spike/native_compile_requirements.txt") != (
        requirements_path.read_text().strip()
    ):
        raise PackRefused("compile requirements differ between the qualified tree and harness")
    native = compile_manifest(commit, package / "native-compile/output", requirements)
    canary = _manifest(
        commit=commit,
        mode="canary",
        output=package / "canary-output",
        job_id=f"gpu-identity-canary-{commit[:12]}",
        metadata=metadata,
        ledger_path="canary-inputs/verified-ledger.json",
    )
    hashes = {
        "native": _write_json_once(package / "native-compile/native-compile.json", native),
        "canary": _write_json_once(packet / "identity-canary.json", canary),
    }
    hold = (
        "HOLD: JUDGMENT_POST_I0S_DIRECTION step 3 says 'no new compile'; the A100 canary gate "
        "needs a native-compile PASS receipt bound to the harness commit. Dispatch nothing "
        "until a ruling admits the L4 compile proof (see HANDOFF_IDENTITY_20261024.md).\n"
    )
    for directory in (packet, package / "native-compile"):
        tape.write_text_once(directory / "DISPATCH.txt", hold)
    return {
        "harness": harness,
        "preregistration_sha256": sha256_file(packet / "PREREGISTRATION.json"),
        "seed_audit_sha256": sha256_file(packet / "SEED_AUDIT.json"),
        "metadata": metadata,
        "manifests": hashes,
    }


def bind_compile(package: Path, receipt: Path) -> Path:
    """Canary manifest + the native compile PASS receipt binding (dispatch gate)."""
    from research.gpu_sim_spike.native_compile import binding

    packet = package / "packet"
    canary = json.loads((packet / "identity-canary.json").read_text())
    canary["native_cuda_compile"] = {"receipt": binding(receipt, canary["commit"], ROOT)}
    target = packet / "identity-canary-compiled.json"
    _write_json_once(target, canary)
    return target


def bind_canary(package: Path, receipt: Path) -> Path:
    """Main manifest bound to a PASS family canary receipt at the same commit."""
    packet = package / "packet"
    canary = json.loads((packet / "identity-canary.json").read_text())
    result = json.loads(receipt.read_text())
    if result.get("status") != "PASS" or result.get("stop_verified") is not True:
        raise PackRefused("canary receipt is not a verified-stop PASS")
    if result.get("commit") != canary["commit"]:
        raise PackRefused("canary receipt commit differs")
    summary = receipt.parent / "exports/identity-device.json"
    tape.verify(summary)
    if json.loads(summary.read_text()).get("status") != "COMPLETE":
        raise PackRefused("canary producer did not complete both canary worlds")
    metadata = next(row for row in canary["inputs"] if row["dest"] == "metadata.tar")
    main = _manifest(
        commit=canary["commit"],
        mode="main",
        output=package / "main-output",
        job_id=f"gpu-identity-main-{canary['commit'][:12]}",
        metadata={"path": metadata["path"], "sha256": metadata["sha256"]},
        ledger_path="inputs/verified-ledger.json",
    )
    main["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": sha256_file(receipt)}}
    target = packet / "identity-main.json"
    _write_json_once(target, main)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--commit", required=True)
    frozen.add_argument("--package", type=Path, default=PACKAGE_DIR)
    frozen.add_argument("--no-scan", action="store_true")
    compiled = sub.add_parser("bind-compile")
    compiled.add_argument("--receipt", type=Path, required=True)
    compiled.add_argument("--package", type=Path, default=PACKAGE_DIR)
    canaried = sub.add_parser("bind-canary")
    canaried.add_argument("--receipt", type=Path, required=True)
    canaried.add_argument("--package", type=Path, default=PACKAGE_DIR)
    audit = sub.add_parser("seed-audit")
    audit.add_argument("--no-scan", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.commit, args.package, scan=not args.no_scan), indent=2))
    elif args.command == "bind-compile":
        print(bind_compile(args.package, args.receipt))
    elif args.command == "bind-canary":
        print(bind_canary(args.package, args.receipt))
    else:
        print(json.dumps(seed_audit(scan=not args.no_scan), indent=2))


if __name__ == "__main__":
    main()
