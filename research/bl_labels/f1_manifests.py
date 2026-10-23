"""Prepare the source-bound F1 ruler; all live operations belong to the operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

from . import manifests
from .exports import _verify, _write_once, publish_json
from .f1_ruler import bank_digest, load_bank
from .f1_units import empty_ledger, load_ledger

SCRIPT = "research/bl_labels/f1_colab.sh"
QUALIFIED_TREE = "d0d92bdb572b770284a1fea6152417ffa5f06bea"
QUALIFIED_RECORD = manifests.QUALIFIED_RECORD
REFUND_SHA256 = "f2f35631e65e4234341a5404e3f512a740b43241c9060f4cf8a467673091b6bc"
REFUND_SOURCE = "2af48cd0b98a2063f46871cf4d320a5541b789af"
# Exact fresh F1-R caps; the ruling has already applied the owner's factor.
REFUND_CAPS = {
    "cpu-probe": (40, 0.07),
    "cpu-science": (180, 0.30),
    "l4-probe": (20, 0.65),
    "l4-science": (94, 3.02),
}
# Base forecasts/caps; owner factor applies once, including reserves/timeouts.
STAGES = {
    "cpu-probe": {"minutes": 10, "units": 0.02, "reserve": 0.004, "finalize": 60},
    "cpu-science": {"minutes": 45, "units": 0.08, "reserve": 0.016, "finalize": 150},
    "l4-probe": {"minutes": 5, "units": 0.18, "reserve": 0.036, "finalize": 60},
    "l4-science": {"minutes": 23.5, "units": 0.76, "reserve": 0.152, "finalize": 150},
}


def source_paths() -> list[str]:
    """Retain the runtime closure while excluding docs/tests from sparse delivery."""
    paths = set(manifests.source_paths())
    for directory in ("research/bl_labels", "research/colab_cli"):
        paths.discard(directory)
        for path in (manifests.ROOT / directory).iterdir():
            if (
                path.is_file()
                and path.suffix in {".py", ".sh", ".txt"}
                and not path.name.startswith("test_")
                and (
                    directory != "research/bl_labels"
                    or path.name not in {"manifests.py", "f1_manifests.py", "f1_analyze.py"}
                )
            ):
                paths.add(path.relative_to(manifests.ROOT).as_posix())
    return sorted(paths)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(path: Path, dest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "dest": dest}


def triplet(path: Path, dest: str) -> list[dict[str, str]]:
    _verify(path)
    # F0 sidecars bind the payload basename as well as its content digest.
    # Uploading unchanged sidecar bytes under a renamed payload breaks _verify.
    if Path(dest).name != path.name:
        raise ValueError(f"F1 triplet upload must preserve basename: {path.name} -> {dest}")
    return [
        _input(Path(str(path) + suffix), dest + suffix) for suffix in ("", ".sha256", ".complete")
    ]


def envelope(stage: str, *, refund: bool = False) -> dict[str, Any]:
    base = STAGES[stage]
    if refund:
        minutes, units = REFUND_CAPS[stage]
        return {
            "base": {"minutes": minutes / 2, "units": units / 2, "reserve": units / 10},
            "effective": {
                "minutes": minutes,
                "units": units,
                "reserve": units / 5,
                "finalize": 2 * base["finalize"],
            },
            "owner_multiplier": 2,
            "authority_sha256": REFUND_SHA256,
            "further_doubling": False,
            "estimate_minutes_unchanged": base["minutes"],
            "reserve_basis": "20 percent of rounded cap, inside cap",
        }
    return {
        "base": dict(base),
        "effective": {k: 2 * v for k, v in base.items()},
        "owner_multiplier": 2,
    }


def qualification() -> dict[str, Any]:
    record = json.loads(QUALIFIED_RECORD.read_text())
    if record["tree_hash"] != QUALIFIED_TREE:
        raise ValueError("F1 qualified transport tree differs")
    path = Path(record["acceptance_result"])
    ready, reason = False, "acceptance missing or hash changed"
    if path.is_file() and _sha(path) == record["acceptance_result_sha256"]:
        result = json.loads(path.read_text())
        transfer = result.get("final_artifact_transfer", {})
        ready = (
            result.get("status") == "PASS"
            and result.get("qualified") is True
            and transfer.get("status") == "PASS"
            and transfer.get("hardware") == "L4"
            and transfer.get("verified_on_mac") is True
        )
        reason = "qualified L4 mitigation acceptance" if ready else "L4 acceptance not passed"
    return {
        "tree_hash": QUALIFIED_TREE,
        "record_sha256": _sha(QUALIFIED_RECORD),
        "acceptance_result": str(path),
        "acceptance_result_sha256": record["acceptance_result_sha256"],
        "l4_ready": ready,
        "reason": reason,
    }


def make_template(
    commit: str,
    output: Path,
    bank: Path,
    ledger: Path,
    b8: Path,
    frp3: Path,
    *,
    platform: str,
    qualified: dict[str, Any],
    is_probe: bool = False,
    refund: bool = False,
) -> dict[str, Any]:
    from research.bl_upload import assert_upload_budget, convert_public_checkpoints

    if platform == "l4" and not qualified["l4_ready"]:
        raise ValueError("F1 L4 NOT_RUN: stall mitigation is not qualified")
    bank_sha = bank_digest(bank)
    saved = load_ledger(ledger, bank_sha, platform, commit)
    stage = f"{platform}-" + ("probe" if is_probe else "science")
    limits = envelope(stage, refund=refund)["effective"]
    inputs = triplet(bank, "inputs/f1-root-bank.json.xz") + triplet(
        ledger, "inputs/f1-verified-ledger.json"
    )
    ledger_value = json.loads(ledger.read_text())
    for row in ledger_value["units"]:
        inputs += triplet(ledger.parent / row["path"], "inputs/" + row["path"])
    checkpoints = manifests._pool_inputs(b8, frp3)
    if len(checkpoints) != 6:
        raise ValueError("F1 requires all six pinned public checkpoints")
    inputs += checkpoints
    timeout = limits["minutes"]
    rate = 1.54 if platform == "l4" else 0.08
    if rate * timeout / 60 + limits["reserve"] > limits["units"] + 1e-12:
        raise ValueError("F1 whole-session forecast plus reserve exceeds cap")
    value = {
        "job_id": f"bl-f1-{platform}-" + ("probe-" if is_probe else "ruler-") + commit[:8],
        "commit": commit,
        "script": SCRIPT,
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_labels/requirements.txt",
            "F1_DEVICE": "cpu",
            "F1_COMMIT": commit,
            "F1_PLATFORM": platform,
            "PYTHONUNBUFFERED": "1",
            "F1_SESSION_SECONDS": str(int(timeout * 60)),
            "F1_FINALIZE_SECONDS": str(int(limits["finalize"])),
            **{
                key: "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "requirements": {
            "gpu": "L4" if platform == "l4" else "cpu",
            "allowed_gpus": ["L4"] if platform == "l4" else ["cpu"],
            "min_vcpu": 12 if platform == "l4" else 2,
            "est_minutes": STAGES[stage]["minutes"],
        },
        "timeout_minutes": timeout,
        "output_dir": str(output.resolve()),
        "inputs": inputs,
        "budget": {
            "unit_cap": limits["units"],
            "rate_units_per_hour": rate,
            "reserve_units": limits["reserve"],
            "other_assignments": {},
        },
        "lease_seconds": 180,
        "max_export_bytes": 64 * 1024**2,
        "source_paths": source_paths(),
        "required_exports": ["runtime-result.json", "session-budget.json"]
        + [
            f"units/config-{label}.json"
            for label in ("one", "half", "all")
            if is_probe or label not in saved
        ],
        "qualified_transport": qualified,
        "canary": {"template": True, "family": SCRIPT},
        "resume": {
            "unit_kind": "64-root-worker-schedule",
            "ledger_path": "inputs/f1-verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "f1": {
            "bank_sha256": bank_sha,
            "stage": stage,
            "envelope": envelope(stage, refund=refund),
            "schedules": ["one", "half", "all"],
            "workers": "1,max(1,nproc//2),nproc",
            "series_base_units": 2.02 if refund else 1.04,
            "series_effective_units": 4.04 if refund else 2.08,
            "effective_ruling_ceiling_units": 5.0 if refund else 2.5,
            "retry_policy": (
                "original plus at most two manual mechanical completions; "
                "all attempts share stage and series caps; "
                "verified stop and missing units required; "
                "no automatic job retry"
            ),
            "already_verified_schedules": sorted(saved),
        },
    }
    if is_probe:
        value["env"].update(
            F1_TRANSPORT_PROBE="1",
            SNAKE_COLAB_CANARY="1",
            SNAKE_COLAB_CANARY_PACKAGE="research/bl_labels",
        )
        value["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
        value.pop("resume")
        value["resume_exemption"] = {
            "kind": "short-canary",
            "reason": (
                "One frozen root through the actual F1/F0 entrypoint; "
                "excluded from timing comparison"
            ),
            "timeout_minutes": timeout,
        }
    if refund:
        value["f1"].update(
            refund_authority_sha256=REFUND_SHA256,
            lifetime_account_cap_units=4.20,
            max_completion_retries=1,
            retry_policy=(
                "original plus at most one manual outcome-blind mechanical completion; "
                "all attempts share this fresh stage cap; verified stop and missing units; "
                "any further INVALID requires HOLD and a new ruling; no automatic job retry"
            ),
        )
    value = convert_public_checkpoints(value, repo=manifests.ROOT)
    assert_upload_budget(value, repo=manifests.ROOT)
    return value


def bind(template: Path, receipt: Path, out: Path) -> None:
    """Require the actual same-pin, same-hardware PASS before materializing main."""
    value = json.loads(template.read_text())
    from research.colab_cli.canary import check

    value["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": _sha(receipt)}}
    # The qualified verifier checks pin/script/hardware/input inventory identity,
    # completed workload evidence, verified exports, and backend release.
    check(value, manifests.ROOT)
    publish_json(out, value)


def charge_failed_probe(value: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    """Carry the verified packaging failure into its original frozen stage caps.

    Preserve measured estimates: insufficient remaining headroom is an admission
    refusal, never a reason to lower a forecast or reset the series envelope.
    """
    receipt = json.loads(receipt_path.read_text())
    manifest_path = receipt_path.parent / "manifest.json"
    previous = json.loads(manifest_path.read_text())
    runtime_path = receipt_path.parent / "exports/runtime-result.json"
    runtime_sha = _verify(runtime_path)
    runtime = json.loads(runtime_path.read_text())
    platform = value["env"]["F1_PLATFORM"]
    stage = f"{platform}-probe"
    if (
        receipt.get("manifest_sha256") != _sha(manifest_path)
        or receipt.get("job_id") != previous.get("job_id")
        or receipt.get("status") != "INCOMPLETE"
        or receipt.get("stop_verified") is not True
        or receipt.get("failure_class") != "CHILD_NONZERO_EXIT"
        or previous.get("f1", {}).get("stage") != stage
        or previous["f1"].get("prior_attempts")
        or value["f1"].get("stage") != stage
        or value["f1"].get("prior_attempts")
        or previous["f1"].get("bank_sha256") != value["f1"]["bank_sha256"]
        or runtime.get("bank_sha256") != value["f1"]["bank_sha256"]
        or runtime.get("status") != "INVALID"
        or runtime.get("reason") != "F0 export digest mismatch"
        or runtime.get("platform") != platform
        or not any(
            row.get("path") == "runtime-result.json" and row.get("sha256") == runtime_sha
            for row in receipt.get("exports", [])
        )
    ):
        raise ValueError("F1 prior probe is not a verified stopped packaging failure")
    elapsed = receipt.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("F1 prior probe elapsed time is unverified")
    limits = envelope(stage)["effective"]
    rate = value["budget"]["rate_units_per_hour"]
    # Conservatively charge the whole supervisor lifetime, including setup and stop.
    charged = elapsed * rate / 3600
    remaining_seconds = math.floor(limits["minutes"] * 60 - elapsed)
    remaining_units = limits["units"] - charged
    if remaining_seconds <= 0 or remaining_units <= 0:
        raise ValueError("F1 prior probe exhausted its frozen envelope")
    value["timeout_minutes"] = remaining_seconds / 60
    value["budget"]["unit_cap"] = remaining_units
    value["env"]["F1_SESSION_SECONDS"] = str(remaining_seconds)
    evidence = {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": _sha(receipt_path),
        "manifest_sha256": _sha(manifest_path),
        "elapsed_seconds_charged": elapsed,
        "units_charged": charged,
        "remaining_seconds": remaining_seconds,
        "remaining_units": remaining_units,
        "reserve_units_inside_remaining_cap": value["budget"]["reserve_units"],
        "completion_attempt": 1,
        "max_completion_retries": 2,
        "headroom_admitted": remaining_seconds >= 120 * value["requirements"]["est_minutes"],
    }
    value["f1"]["prior_attempts"] = [evidence]
    return evidence


def build(
    commit: str,
    bank: Path,
    b8: Path,
    frp3: Path,
    out: Path,
    output_root: Path,
    prior_probe_receipts: list[Path] | None = None,
    refund_ruling: Path | None = None,
) -> None:
    manifests.require_source_pin(commit)
    refund = refund_ruling is not None
    if refund and (_sha(refund_ruling) != REFUND_SHA256 or commit != REFUND_SOURCE):
        raise ValueError("F1 refund requires the exact judgment and repaired source pin")
    if out.exists() or output_root.exists():
        raise ValueError("F1 pack and output root must be fresh")
    load_bank(bank)
    qualified = qualification()
    prior = {}
    for receipt in prior_probe_receipts or []:
        platform = json.loads((receipt.parent / "manifest.json").read_text())["env"]["F1_PLATFORM"]
        if platform not in {"cpu", "l4"} or platform in prior:
            raise ValueError("duplicate or foreign prior F1 probe platform")
        prior[platform] = receipt
    if refund and set(prior) != {"cpu", "l4"}:
        raise ValueError("F1 refund must preserve both failed pack-08 probes")
    census = None
    if refund:
        from .f1_analyze import (
            REFUND_CENSUS_SCHEMA,
            REFUND_EXCLUSION,
            REFUND_PACK08_EVIDENCE,
            REFUND_WITNESS_RECEIPT_SHA256,
        )

        witness = output_root.parent / "canary-output-05-probe/receipt.json"
        if _sha(witness) != REFUND_WITNESS_RECEIPT_SHA256:
            raise ValueError("F1 refund qualified witness differs")
        excluded = []
        for platform, receipt in prior.items():
            paths = {
                "receipt_sha256": receipt,
                "manifest_sha256": receipt.parent / "manifest.json",
                "runtime_sha256": receipt.parent / "exports/runtime-result.json",
            }
            pinned = REFUND_PACK08_EVIDENCE[f"{platform}-probe"]
            if any(_sha(path) != pinned[key] for key, path in paths.items()):
                raise ValueError("F1 refund only covers the exact ruled pack-08 failures")
            excluded.append(
                {
                    "path": str(receipt.parent.resolve()),
                    **{key: _sha(path) for key, path in paths.items()},
                    "classification": REFUND_EXCLUSION,
                }
            )
        census = {
            "schema": REFUND_CENSUS_SCHEMA,
            "ruling": {"path": str(refund_ruling.resolve()), "sha256": REFUND_SHA256},
            "qualification_witness": {
                "path": str(witness.resolve()),
                "receipt_sha256": REFUND_WITNESS_RECEIPT_SHA256,
            },
            "excluded_attempts": excluded,
        }
    prior_spend = []
    for platform, receipt in prior.items():
        old = json.loads((receipt.parent / "manifest.json").read_text())
        if old["f1"]["bank_sha256"] != bank_digest(bank):
            raise ValueError("F1 refund prior bank differs")
        evidence = charge_failed_probe(old, receipt)
        prior_spend.append(
            {
                "platform": platform,
                **{
                    key: evidence[key]
                    for key in (
                        "receipt",
                        "receipt_sha256",
                        "manifest_sha256",
                        "elapsed_seconds_charged",
                        "units_charged",
                    )
                },
                "classification": "PACKAGING_INVALID_EXCLUDED (JUDGMENT_F1_REFUND)",
            }
        )
    if refund and 4.04 + sum(row["units_charged"] for row in prior_spend) > 4.20:
        raise ValueError("F1 refund lifetime account exceeds 4.20 units")
    packed = out / "f1-root-bank.json.xz"
    payload = lzma.compress(bank.read_bytes(), preset=6)
    _write_once(packed, payload)
    _write_once(Path(str(packed) + ".sha256"), f"{_sha(packed)}  {packed.name}\n".encode())
    _write_once(Path(str(packed) + ".complete"), b"complete\n")
    if bank_digest(packed) != bank_digest(bank):
        raise ValueError("compressed F1 bank differs after lossless round trip")
    publish_json(
        out / "bank-transport.json",
        {
            "raw_path": str(bank),
            "raw_sha256": bank_digest(bank),
            "compressed_sha256": _sha(packed),
            "raw_bytes": bank.stat().st_size,
            "compressed_bytes": packed.stat().st_size,
        },
    )
    from tools.ops.import_preflight import check

    preflight = check(
        manifests.ROOT,
        commit,
        source_paths(),
        "research/bl_labels/requirements.txt",
        [*manifests.RUNTIME_ROOTS, "research.bl_labels.f1_ruler"],
    )
    publish_json(out / "clean-import-preflight.json", preflight)
    publish_json(
        out / "budget-plan.json",
        {
            "schema": "bl-labels-f1-budget/v1",
            "stages": {k: envelope(k, refund=refund) for k in STAGES},
            "base_series_units": 2.02 if refund else 1.04,
            "effective_series_units": 4.04 if refund else 2.08,
            "base_series_reserve": 0.404 if refund else 0.208,
            "effective_series_reserve": 0.808 if refund else 0.416,
            "reserves_are_inside_caps": True,
            "effective_ruling_ceiling": 5.0 if refund else 2.5,
            "max_completion_retries": 1 if refund else 2,
            "retry_liability": "shared stage caps, never additive or reset",
            "basis": (
                "F0 observed setup116.8s and original qualified-session guards; "
                "source natural replay and full matrix charged to whole session; "
                "timing-only rate excludes separately reported replay setup"
            ),
            "qualified": qualified,
            "refund_authority": (
                {"path": str(refund_ruling.resolve()), "sha256": REFUND_SHA256} if refund else None
            ),
            "prior_spend": prior_spend,
            "prior_spent_units": sum(row["units_charged"] for row in prior_spend),
            "prior_spend_charged_against_fresh_stages": not refund,
            "lifetime_account_cap_units": 4.20 if refund else 2.5,
            "lifetime_worst_case_units": (
                4.04 + sum(row["units_charged"] for row in prior_spend) if refund else 2.08
            ),
        },
    )
    if refund:
        publish_json(out / "refund-census.json", census)
        publish_json(
            out / "f1-a100-NOT_RUN.json",
            {
                "status": "NOT_RUN",
                "reason": "NOT_RUN by ruling: CPU-only labeller; no A100 arm",
                "authority_sha256": REFUND_SHA256,
            },
        )
    for platform in ("cpu", "l4"):
        if platform == "l4" and not qualified["l4_ready"]:
            publish_json(
                out / "f1-l4-NOT_RUN.json",
                {"status": "NOT_RUN", "reason": "stall mitigation qualification absent"},
            )
            continue
        ledger = out / platform / "f1-verified-ledger.json"
        publish_json(ledger, empty_ledger(bank_digest(bank), platform, source_commit=commit))
        for is_probe in (True, False):
            name = f"f1-{platform}-" + ("probe.json" if is_probe else "template.json")
            output = output_root / (platform + ("-probe" if is_probe else "-ruler"))
            value = make_template(
                commit,
                output,
                packed,
                ledger,
                b8,
                frp3,
                platform=platform,
                qualified=qualified,
                is_probe=is_probe,
                refund=refund,
            )
            if is_probe and platform in prior and not refund:
                charge_failed_probe(value, prior[platform])
            if refund:
                value["f1"]["refund_census"] = {
                    "path": str((out / "refund-census.json").resolve()),
                    "sha256": _sha(out / "refund-census.json"),
                }
            publish_json(out / name, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("build")
    prepare.add_argument("--commit", required=True)
    prepare.add_argument("--prior-probe-receipt", type=Path, action="append", default=[])
    prepare.add_argument("--refund-ruling", type=Path)
    for name in ("bank", "b8", "frp3", "out", "output-root"):
        prepare.add_argument("--" + name, type=Path, required=True)
    binder = sub.add_parser("bind")
    for name in ("template", "receipt", "out"):
        binder.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bind":
        bind(args.template, args.receipt, args.out)
    else:
        build(
            args.commit,
            args.bank,
            args.b8,
            args.frp3,
            args.out,
            args.output_root,
            args.prior_probe_receipt,
            args.refund_ruling,
        )


if __name__ == "__main__":
    main()
