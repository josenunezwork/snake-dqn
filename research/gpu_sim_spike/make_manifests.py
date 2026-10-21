"""Create and optionally validate the E1 L4 spike and optional probe manifests.

This module is deliberately allocation-free.  It binds the primary checkout's
frozen B8 and vector-opponent checkpoints as supervisor uploads, produces an L4
spike and an optional L4 probe manifest, and writes dispatch instructions that
require an operator to refresh account assignment and provider-rate evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn")
DEFAULT_ARTIFACT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1")
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/colab.sh"

CHECKPOINTS = (
    (
        "b8_s3_t7_20261016.pth",
        "B8_s3_final.pth",
        "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    ),
    (
        "champion_a5_freespace_20260621.pth",
        "checkpoints/champion_a5_freespace_20260621.pth",
        "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
    ),
    (
        "best_apex_fs.pth",
        "checkpoints/best_apex_fs.pth",
        "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195",
    ),
    (
        "best_apex_stage1_fs.pth",
        "checkpoints/best_apex_stage1_fs.pth",
        "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d",
    ),
    (
        "best_apex_pre_fs.pth",
        "checkpoints/best_apex_pre_fs.pth",
        "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
    ),
    (
        "frp3_m3_s12_u60000_20261005.pth",
        "checkpoints/frp3_m3_s12_u60000_20261005.pth",
        "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
    ),
)


class ManifestRefused(ValueError):
    """A draft would not meet the frozen spike transport contract."""


def sha256_file(path: Path) -> str:
    """Return a regular file's SHA-256 without loading a checkpoint in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(repo: Path = ROOT) -> str:
    """Return the exact current source commit for a pinned draft."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def checkpoint_inputs(saved_snakes: Path = PRIMARY_ROOT / "saved_snakes") -> list[dict[str, str]]:
    """Hash-verify B8 plus every vector checkpoint required by the reference constructor."""
    rows: list[dict[str, str]] = []
    for filename, destination, expected_sha in CHECKPOINTS:
        path = (saved_snakes / filename).resolve()
        if not path.is_file():
            raise ManifestRefused(f"required primary-checkout checkpoint is absent: {path}")
        actual = sha256_file(path)
        if actual != expected_sha:
            raise ManifestRefused(f"checkpoint identity drift: {path}")
        rows.append({"path": str(path), "sha256": actual, "dest": destination})
    return rows


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ManifestRefused("commit must be a full lowercase 40-character SHA")


def preflight_binding(path: Path) -> dict[str, Any]:
    """Bind the real tiny Mac preflight before turning estimates into drafts."""
    receipt_path = path.expanduser().resolve(strict=True)
    try:
        receipt = json.loads(receipt_path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestRefused(f"invalid preflight receipt: {receipt_path}") from exc
    if receipt.get("status") != "PREFLIGHT" or receipt.get("device") != "cpu":
        raise ManifestRefused("preflight must be a successful CPU PREFLIGHT receipt")
    elapsed = receipt.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or elapsed <= 0:
        raise ManifestRefused("preflight must record positive elapsed_seconds")
    if receipt.get("parity", {}).get("decisions", 0) < 3:
        raise ManifestRefused("preflight must exercise at least three searched decisions")
    if not receipt_path.with_name(receipt_path.name + ".complete").is_file():
        raise ManifestRefused("preflight completion marker is missing")
    sidecar = receipt_path.with_name(receipt_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != sha256_file(receipt_path):
        raise ManifestRefused("preflight SHA256 sidecar differs")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "elapsed_seconds": elapsed,
        "status": receipt["status"],
        "decisions": receipt["parity"]["decisions"],
        "device_loop_seconds": receipt["device_loop_seconds"],
        "source_sha256": receipt["source_sha256"],
        "kill_rule_sha256": receipt["e1_kill_rule_sha256"],
    }


def make_manifest(
    *, kind: str, commit: str, output_dir: Path, preflight: dict[str, Any]
) -> dict[str, Any]:
    """Create a no-substitution, locally uploadable supervisor draft for one GPU type."""
    _require_commit(commit)
    output = output_dir.expanduser()
    if not output.is_absolute():
        raise ManifestRefused("output_dir must be an absolute fresh Mac directory")
    if kind == "l4":
        gpu, rate, estimate, timeout, seconds = "L4", 2.0, 190, 400, 10800
        unit_cap, reserve = 13.5, 1.5
        horizon, decisions = "300", "3"
    elif kind == "probe":
        gpu, rate, estimate, timeout, seconds = "L4", 2.0, 60, 135, 3300
        unit_cap, reserve = 4.5, 0.5
        horizon, decisions = "8", "1"
    else:
        raise ManifestRefused("kind must be l4 or probe")
    rule = subprocess.check_output(["git", "show", f"{commit}:{PACKAGE}/KILL_RULE.md"], cwd=ROOT)
    if hashlib.sha256(rule).hexdigest() != preflight["kill_rule_sha256"]:
        raise ManifestRefused("kill rule differs from preflight")
    for name, expected in preflight.get("source_sha256", {}).items():
        if name.startswith("test_") or name == "make_manifests.py":
            continue
        payload = subprocess.check_output(["git", "show", f"{commit}:{PACKAGE}/{name}"], cwd=ROOT)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ManifestRefused(f"runtime source differs from measured preflight: {name}")
    return {
        "job_id": f"bl-e1-{kind}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "DEVELOPMENT_SPIKE_EXCLUDED_FROM_REGISTERED_VERDICTS_NOT_DISPATCHED",
        "transport": {"mode": "local_supervisor_upload", "dispatch_ready": False},
        "env": {
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_DEVICE": "cuda",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_SPIKE_HORIZON": horizon,
            "GPU_SIM_SPIKE_SECONDS": str(seconds),
            "GPU_SIM_SPIKE_THREADS": "1",
            "GPU_SIM_SPIKE_CPU_WORKERS": "12",
            "GPU_SIM_SPIKE_FIXTURES": "all",
            "GPU_SIM_SPIKE_DECISIONS_PER_CASE": decisions,
            "GPU_SIM_SPIKE_RATE_UNITS_PER_HOUR": str(rate),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        },
        "requirements": {
            "gpu": gpu,
            "allowed_gpus": [gpu],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output),
        "inputs": checkpoint_inputs(),
        "budget": {
            "unit_cap": unit_cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 32 * 1024**2,
        "source_paths": [
            "src",
            "web/backend",
            "configs",
            PACKAGE,
            "research/colab_cli",
            "research/speed_s0_baseline",
            "research/t34_exit_b8",
            "research/t34_shadow",
            "research/t33_batched_teacher",
            "research/phase0_search_20261019",
            "research/redesign_scope_20261007",
            "research/t3_rollout_labels_20261015",
            "research/t1_commitment_20261015",
            "research/apex_safety_20260926",
            "research/b8_death_census_20261014",
            "research/fixed_search.py",
            "requirements.txt",
            "pyproject.toml",
        ],
        "gpu_sim_spike": {
            "spec": "docs/research/specs/2026-10-21_gpu_sim_spike.md",
            "baseline": f"{PACKAGE}/BASELINE.json",
            "kill_rule": f"{PACKAGE}/KILL_RULE.md",
            "benchmark": f"six S0 fixtures x {decisions} decisions D{horizon}; CPU12 plus CUDA",
            "role": "full spike" if kind == "l4" else "optional probe; cannot qualify D300 rule",
            "forecast_scope": (
                "bounded allocation timebox, not a completion forecast; tiny Mac extrapolation "
                "does not measure full-board CUDA speed; see E1_PREFLIGHT.md"
            ),
            "provider_rate_notice": (
                "Manifest rate is the conservative preflight ceiling; "
                "final cost uses operator-measured provider rate."
            ),
            "mac_preflight": preflight,
        },
    }


def validate_manifest(manifest: dict[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Call the supervisor's local pure validator and never allocate a runtime."""
    from research.colab_cli.supervisor import validate

    return validate(manifest, repo=repo)


def dispatch_text(manifests: dict[str, Path]) -> str:
    """Give Claude the only admissible next steps; this is intentionally not executable dispatch."""
    return f"""# E1 dispatch handoff — PREPARED ONLY, NO COLAB CALLS BY SOL

Development experiment excluded from every registered verdict. Rule frozen at
896310e3 before E1 data. No A100 run. Claude is the dispatcher, using only the
qualified research.colab_cli supervisor. Preserve Round 0, D0, O1 priorities.
Do not dispatch both manifests as automatic retries. The probe is optional,
D8 diagnostic work; it cannot qualify the original D300 cost/agreement rule.

- L4 spike: `{manifests['l4']}` — 190 min planning timebox; 400 min wall cap;
  13.5 u execution cap + 1.5 u reserve = 15 u envelope at 2 u/h.
- Optional L4 probe: `{manifests['probe']}` — 60 min planning timebox;
  135 min wall cap; 4.5 u execution cap + 0.5 u reserve = 5 u envelope.

First wall/unit cap wins. No substitution, automatic retry, or direct colab calls.
Both estimates include setup/export headroom but are not measured CUDA completion
forecasts. See committed E1_PREFLIGHT.md for measured tiny Mac extrapolation.

Before each dispatch Claude must locally validate the exact manifest, refresh
qualified transport/source binding, and record all active assignments and their
reviewed positive rates in budget.other_assignments. Empty assignments is a draft
placeholder, NOT live inventory. Refuse unknown rates/endpoints. Refresh the
provider rate and preserve the 15/5 u envelopes. Never edit pinned source/rule.

From /Users/josenunez/Projects/ml/snake-dqn-bl-e1 (or a checkout containing the pin):
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
$PY -m research.colab_cli.supervisor \\
  --state-dir {manifests['l4'].parent}/supervisor-l4 \\
  validate {manifests['l4']}
$PY -m research.colab_cli.supervisor \\
  --state-dir {manifests['probe'].parent}/supervisor-probe \\
  validate {manifests['probe']}

Claude dispatches the selected manifest through research/colab_cli/launch.sh
in an owned tmux session with AC/lid guards. Retain terminal receipt, verified
exports and independently confirmed assignment disappearance. Missing exports or
INCOMPLETE cannot support CONTINUE. Apply <20k full world-steps/s on any L4 long
mix => STOP engine thesis this week. Report decisive-root agreement separately;
zero decisive samples is NO_COVERAGE.

Every exported JSON must have immutable .sha256 and .complete sidecars, including
benchmark.json, cpu-reference-before-device.json, decision-*.json and host-export.json.
The package test exercises actual supervisor export discovery and checksum checks.
"""


def write_once(path: Path, payload: str) -> None:
    """Avoid overwriting a prior artifact handoff or its immutable draft."""
    if path.exists():
        raise ManifestRefused(f"refusing to overwrite existing artifact: {path}")
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit", default=None, help="full commit containing the complete package"
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--preflight",
        type=Path,
        required=True,
        help="successful tiny CPU receipt whose identity is bound into both drafts",
    )
    parser.add_argument(
        "--validate", action="store_true", help="run only the local supervisor validator"
    )
    args = parser.parse_args()
    commit = args.commit or git_head()
    artifacts = args.artifact_dir.expanduser()
    if not artifacts.is_absolute():
        raise ManifestRefused("artifact-dir must be absolute")
    preflight = preflight_binding(args.preflight)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    for kind in ("l4", "probe"):
        output_dir = artifacts / f"{kind}-output"
        manifest = make_manifest(
            kind=kind, commit=commit, output_dir=output_dir, preflight=preflight
        )
        if args.validate:
            validate_manifest(manifest)
        target = artifacts / f"{kind}-manifest.json"
        write_once(target, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[kind] = target
    write_once(artifacts / "DISPATCH.txt", dispatch_text(manifests))
    print(json.dumps({"artifacts": str(artifacts), "validated": args.validate}, sort_keys=True))


if __name__ == "__main__":
    main()
