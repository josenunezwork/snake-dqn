"""Freeze K-1 roots, thresholds, source closure and Mac-only dispatch forecast."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import secrets
import subprocess
from pathlib import Path
from typing import Any

from research.bl_served_search import ruler
from research.bl_served_search.exports import digest, export_bytes, export_json, verify_export
from research.bl_served_search.ruler_operator import ExactT3Operator

ROOT = Path(__file__).resolve().parents[2]
ORACLE_COMMIT = "a7bd01e99114c80f862db33f77d4ac252a44496c"


def sources() -> dict[str, str]:
    """Bind every K-1 instrument plus the stock/S2/kernel decision code."""
    names = [
        *ROOT.joinpath("research/bl_k1").glob("*.py"),
        ROOT / "research/bl_k1/run_mac.sh",
        ROOT / "research/t34_speed_s2/native_v8.py",
        ROOT / "src/simd_env/vector61_policy.py",
        ROOT / "src/simd_env/v8_nb.py",
    ]
    names += list(ROOT.joinpath("src/simd_env").glob("*.py"))
    names += list(ROOT.joinpath("src/evaluation").glob("safety_veto*.py"))
    for directory in ("src/core", "src/game", "src/model", "src/evaluation"):
        names += list(ROOT.joinpath(directory).glob("*.py"))
    names += list(ROOT.joinpath("configs").glob("*.yaml"))
    names += [
        ROOT / "src/training/action_mask.py",
        ROOT / "research/bl_mac_latency/host.py",
        ROOT / "research/bl_mac_latency/ruler_run.py",
        ROOT / "research/bl_ppo/exports.py",
        ROOT / "research/bl_served_search/exports.py",
        ROOT / "research/perf_sim_20261007/with_slots.py",
    ]
    names += [
        ROOT / "research/b8_death_census_20261014/census.py",
        ROOT / "research/redesign_scope_20261007/grid_h5000_identity.py",
    ]
    return {
        **ruler.source_identity(),
        **{str(path.relative_to(ROOT)): digest(path) for path in names},
    }


def verify_manifest(path: Path) -> dict[str, Any]:
    verify_export(path)
    manifest = json.loads(path.read_text())
    if manifest["sources"] != sources():
        raise ValueError("K-1 source closure drift; prepare a new immutable manifest")
    if manifest["bar"] != {
        "primary_mode": "kernel",
        "stage": "opponent_v8",
        "fixture_id": "frozen-1500",
        "minimum_speedup": 5.0,
        "estimator": "ratio of medians, warm29 (first excluded)",
    }:
        raise ValueError("K-1 frozen timing bar changed")
    if len(manifest["fixtures"]) != 9 or len(manifest["fresh_roots"]) != 24:
        raise ValueError("K-1 requires nine Y1 fixtures and 24 fresh roots")
    return manifest


def prepare(fixtures_path: Path, out: Path) -> Path:
    """Reconstruct fresh roots locally; transport only compact reconstruction rows."""
    verify_export(fixtures_path)
    fixture_bank = json.loads(fixtures_path.read_text())
    rows = ruler.validate_fixtures(fixture_bank)
    operator = ExactT3Operator()
    fresh = []
    excluded = {int(row["plan"]["seed"]) for row in rows}
    stock_path = "src/simd_env/vector61_policy.py"
    stock_bytes = subprocess.check_output(["git", "show", f"{ORACLE_COMMIT}:{stock_path}"])
    if hashlib.sha256(stock_bytes).hexdigest() != digest(ROOT / stock_path):
        raise ValueError("stock hook differs from frozen pre-K1 oracle")
    nonce = secrets.token_hex(32)
    for mix in ruler.MIXES:
        template = next(row for row in rows if row["mix"] == mix)
        for index in range(8):
            seed = int.from_bytes(
                hashlib.sha256(f"bl-k1/{nonce}/{mix}/{index}".encode()).digest()[:4], "big"
            )
            if seed in excluded:
                raise ValueError("fresh seed collision; no automatic reselection")
            excluded.add(seed)
            request = copy.deepcopy(template)
            request["fixture_id"] = f"fresh-{mix}-{index:02d}"
            request["target_length"] = ruler.FIXTURE_TARGETS[index % 3]
            request["plan"]["seed"] = seed
            result = operator.prepare(request)
            raw = result.pop("root_bytes")
            fresh.append({**request, **result, "root_sha256": hashlib.sha256(raw).hexdigest()})

    # Frozen fixture bytes remain in their original create-only export.  The
    # runner reconstructs and verifies root_sha256, including food/RNG/carry.
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if key != "root_bytes_b64"}

    manifest = {
        "schema": "bl-k1-manifest/v1",
        "builder": "Sol-K",
        "reviewer": "Codex-V",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": sources(),
        "stock_oracle": {
            "commit": ORACLE_COMMIT,
            "path": stock_path,
            "sha256": hashlib.sha256(stock_bytes).hexdigest(),
        },
        "fixtures": [compact(row) for row in rows],
        "fresh_roots": [compact(row) for row in fresh],
        "fresh_nonce": nonce,
        "freshness": (
            "new cryptographic nonce; unique seeds disjoint from nine Y1 seeds; "
            "constructed long bodies, not efficacy"
        ),
        "original_fixture_export": {
            "path": str(fixtures_path),
            "sha256": digest(fixtures_path),
            "source_identity": fixture_bank["source_identity"],
        },
        "bar": {
            "primary_mode": "kernel",
            "stage": "opponent_v8",
            "fixture_id": "frozen-1500",
            "minimum_speedup": 5.0,
            "estimator": "ratio of medians, warm29 (first excluded)",
        },
        "identity": {
            "fixture_batches": 30,
            "fresh_roots": 24,
            "depth": 300,
            "modes": ["stock", "replay", "kernel"],
            "action_mismatch": "INVALID_MODE_STOP",
        },
        "timing": {
            "date": "2026-10-22",
            "start_pt": "20:00",
            "end_pt": "22:00",
            "workers": 1,
            "threads": 1,
            "batches_per_fixture_per_mode": 30,
            "mode_order": ["stock", "replay", "kernel"],
            "max_seconds": 7200,
            "operator": "Claude",
            "paced": False,
        },
        "deadlines_pt": {"code": "2026-10-22T16:00:00-07:00", "D2": "2026-10-22T22:00:00-07:00"},
        "forecast": {
            "mac_timing_seconds": [3600, 7200],
            "identity_seconds": [5400, 14400],
            "basis": "Y1 stock stage receipts; native/tracing overhead unmeasured; forecasts only",
            "colab_units": 0,
            "upload_bytes": 0,
            "upload_seconds": 0,
            "transport": "MAC_ONLY_NO_COLAB; compact hash-verified runtime reconstruction rows",
            "Z9": (
                "NOT_APPLICABLE_MAC_ONLY; any future Colab manifest needs "
                "real-supervisor upload receipt <=150s of 300s cap"
            ),
        },
        "kill_rules": [
            "any action/Q/world/RNG/decision-carry mismatch invalidates mode",
            "kernel opponent_v8 speedup <5 on frozen-1500 closes exact served search for "
            "week 2; cancel WP-4",
            "missing identity or incomplete timing => NOT_READY, never PASS",
            "no retries, no process/daemon killing; retain stopped attempt",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    path = out / "manifest.json"
    export_json(path, manifest)
    argv = f"{ROOT}/research/bl_k1/run_mac.sh {path}"
    receipt_root = out.parent
    export_bytes(
        out / "DISPATCH.txt",
        (
            "CLAUDE ONLY: no Colab dispatch in K-1.\n"
            "Do not run timing before Thu 2026-10-22 20:00 PT.\n"
            "Identity first (any time on AC/lid open, diagnostic unpaced):\n"
            f"{argv} identity {receipt_root}\n"
            "Then Thu 20:00-22:00 PT:\n"
            f"{argv} timing {receipt_root} {receipt_root}/identity.json\n"
            "No retry: retain stopped receipt, prepare new attempt/manifest.\n"
            "No process/tmux kills.\n"
            "After full audit: publish create-only exp/bl-k1/identity.json;\n"
            "after timing exp/bl-k1/timing.json.\n"
            "No WP-4 admission until identity and >=5x frozen-1500 opponent_v8 both PASS.\n"
        ).encode(),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        value = verify_manifest(args.check)
        print(
            json.dumps(
                {"status": "PREPARED_NOT_TIMED", "bar": value["bar"], "forecast": value["forecast"]}
            )
        )
    elif args.fixtures and args.out:
        print(prepare(args.fixtures, args.out))
    else:
        parser.error("use --check or --fixtures and --out")


if __name__ == "__main__":
    main()
