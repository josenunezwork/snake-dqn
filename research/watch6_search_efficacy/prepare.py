"""Prepare exact hashed inputs/manifests; no allocation, account access or dispatch."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy import protocol as p
from research.watch6_search_efficacy.mirror import validate as validate_mirror
from research.watch6_search_efficacy.performance import forecast as measured_forecast
from research.watch6_search_efficacy.run import completed, ledger_identity
from research.watch6_search_efficacy.uploads import METADATA, checkpoint_inputs, pack_inputs
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs


def prepare(
    out: Path,
    benchmark: Path,
    commit: str,
    approval: Path | None = None,
    ledger: Path | None = None,
    attempt: int = 1,
    previous_receipts: list[Path] | None = None,
    mirror_receipt: Path | None = None,
    canary_receipt: Path | None = None,
    main_shards: bool = False,
    empirical_costs: Path | None = None,
    pending_canary: bool = False,
) -> dict:
    """Prepare canary only, or main shards after measured same-pin admission."""
    if main_shards and canary_receipt is None and not pending_canary:
        raise ValueError("main preparation requires measured passing same-pin canary receipt")
    if pending_canary and (not main_shards or empirical_costs is None or canary_receipt):
        raise ValueError(
            "pending preparation requires empirical main and no claimed canary receipt"
        )
    if mirror_receipt is None:
        raise ValueError("full Mac mirror receipt required before preparation")
    validate_mirror(json.loads(mirror_receipt.read_text()))
    subprocess.run(["git", "cat-file", "-e", commit], cwd=p.ROOT, check=True)
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=p.ROOT, text=True
    )
    if "origin/astra-watch6-search-efficacy" not in refs:
        raise ValueError("pin must be pushed on requested branch")
    if subprocess.check_output(["git", "diff", commit, "--", *p.source_paths()], cwd=p.ROOT):
        raise ValueError("local executable source differs from pin")
    measured = json.loads(benchmark.read_text())
    if measured["science"] != p.identity():
        raise ValueError("stale Mac preflight")
    if type(attempt) is not int or attempt not in (1, 2, 3):
        raise ValueError("one original plus at most two completion attempts")
    previous_receipts = previous_receipts or []
    frozen_layout = None
    if attempt == 1 and (ledger is not None or previous_receipts):
        raise ValueError("first attempt must have empty ledger and no prior receipts")
    if attempt > 1:
        if ledger is None or len(previous_receipts) != attempt - 1:
            raise ValueError("completion needs verified ledger and all prior stopped attempts")
        verified = completed(ledger)
        if len(verified) == p.COUNT:
            raise ValueError("all units verified; never recompute")
        if len({digest(path) for path in previous_receipts}) != len(previous_receipts):
            raise ValueError("duplicate previous-attempt receipt")
        for previous_attempt, receipt_path in enumerate(previous_receipts, 1):
            receipt = json.loads(receipt_path.read_text())
            former_path = receipt_path.parent / "manifest.json"
            former = json.loads(former_path.read_text())
            if (
                digest(former_path) != receipt.get("manifest_sha256")
                or former.get("commit") != commit
                or former.get("watch6_efficacy", {}).get("science") != p.identity()
                or former.get("watch6_efficacy", {}).get("attempt") != previous_attempt
            ):
                raise ValueError("foreign source or completion-attempt lineage")
            error = str(receipt.get("error", "")).upper()
            mechanical = any(
                word in error
                for word in (
                    "TIMEOUT",
                    "LEASE",
                    "TRANSFER",
                    "UPLOAD",
                    "STAGING",
                    "INTERRUPT",
                    "RUNTIME_LOST",
                )
            )
            if (
                receipt.get("stop_verified") is not True
                or receipt.get("status") == "PASS"
                or not mechanical
            ):
                raise ValueError("completion requires stopped, verified mechanical failure")
            prior_layout = json.loads(former["env"]["WATCH6_FROZEN_LAYOUT"])
            if frozen_layout is not None and frozen_layout != prior_layout:
                raise ValueError("completion lineage changed frozen operational layout")
            frozen_layout = prior_layout
    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out / "verified-ledger.json"
    publish_json(
        ledger_path,
        (
            json.loads(ledger.read_text())
            if ledger
            else empty_ledger(p.NAMESPACE, "watch6", ledger_identity())
        ),
    )
    pending = out / "judge-approval.json"
    publish_json(
        pending,
        {
            "status": "RATIFIED",
            "judge": "Astra",
            "implementer": "Astra",
            "independent_implementation_review": False,
            "science": p.identity(),
            "registration_sha256": digest(p.registration()),
            "ruling_sha256": digest(Path(__file__).with_name("RULING.md")),
        },
    )
    saved = p.ROOT.parent / "snake-dqn/saved_snakes"
    contract = out / "runtime-contract.json"
    publish_json(contract, {"science": p.identity(), "allocation": p.allocation()})
    inputs = [
        Input(contract, "runtime-contract.json"),
        Input(ledger_path, "verified-ledger.json"),
        Input(approval or pending, "judge-approval.json"),
        Input(mirror_receipt, "mac-mirror.json"),
        *checkpoint_inputs(saved, out / "checkpoint-parts"),
    ]
    report = None
    if empirical_costs:
        from research.watch6_search_efficacy.costs import verify_report_evidence

        report = json.loads(empirical_costs.read_text())
        verify_report_evidence(report)
        inputs.append(Input(empirical_costs, "cost-evidence.json"))
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    release_bundle = out / "release-evidence.tar.gz"
    with (
        release_bundle.open("xb") as raw_bundle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_bundle, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, pin in pins["checkpoints"].items():
            receipt = pin["strict_receipt"]
            for ref in receipt.values():
                if isinstance(ref, dict):
                    path = Path(receipt["root"]) / ref["path"]
                    if digest(path) != ref["sha256"]:
                        raise ValueError("legacy release evidence mismatch")
                    data = path.read_bytes()
                    member = tarfile.TarInfo(name + "/" + ref["path"])
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
    inputs.append(Input(release_bundle, "release-evidence.tar.gz"))
    staged = pack_inputs(stage_inputs(out / "staged", inputs), out / METADATA)
    if main_shards and report is None:
        receipt = json.loads(canary_receipt.read_text())
        report_path = canary_receipt.parent / "exports/performance.json"
        report = json.loads(report_path.read_text())
        if receipt.get("status") != "PASS" or receipt.get("stop_verified") is not True:
            raise ValueError("measured canary must PASS and stop_verified")
        forecast = measured_forecast(report, frozen_layout)
    elif not main_shards:
        forecast = {"status": "CANARY_ONLY_MAIN_NOT_PREPARED", "base_minutes_per_shard": 15}
    else:
        forecast = measured_forecast(report, frozen_layout)
    base_minutes = forecast["base_minutes_per_shard"]
    rate = 2.5  # planning rate; qualified dispatcher refreshes live accounting
    publish_json(out / "forecast.json", forecast)
    paths = p.source_paths()
    common = {
        "commit": commit,
        "script": p.SCRIPT,
        "source_paths": paths,
        "device": "cpu",
        "lease_seconds": 300,
        "max_export_bytes": 4 * 1024**2,
        "required_exports": ["efficacy/operational.json", "performance.json"],
        "resume": {
            "unit_kind": "paired-session",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "inputs": [{k: r[k] for k in ("path", "dest", "sha256")} for r in staged],
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
            "SNAKE_DQN_DEVICE": "cpu",
            "WATCH6_SCIENCE": p.identity(),
            "WATCH6_METADATA_SHA256": next(
                row["sha256"] for row in staged if row["dest"] == METADATA
            ),
            "PYTHONUNBUFFERED": "1",
            **{
                k: "1"
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "TORCHINDUCTOR_COMPILE_THREADS",
                )
            },
        },
        "watch6_efficacy": {
            "judge": "Astra",
            "implementer": "Astra",
            "independent_implementation_review": False,
            "cost_limitations": (
                "Late-game main-loop cost is unmeasured; "
                "H300 fixtures are forecasts, not all-state bounds."
            ),
            "science": p.identity(),
            "allocation": p.allocation(),
            "attempt": attempt,
            "forecast_sha256": digest(out / "forecast.json"),
        },
    }
    jobs = (
        ([("canary", 0, True)] if pending_canary else [])
        + [(f"shard-{i}", i, False) for i in range(forecast["shards"])]
        if main_shards
        else [("canary", 0, True)]
    )
    for label, shard, probe in jobs:
        value = json.loads(json.dumps(common))
        base = 15 if probe else base_minutes
        unit = base / 60 * rate
        value.update(
            job_id=f"watch6-search-{label}-{out.name}",
            output_dir=str(out / f"output-{label}"),
            timeout_minutes=2 * base,
            requirements={"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 12, "est_minutes": base},
            budget={
                "unit_cap": 2 * unit,
                "rate_units_per_hour": rate,
                "reserve_units": 2 * unit * 0.2,
                "other_assignments": {},
            },
        )
        value["watch6_efficacy"].update(
            base_minutes=base, base_unit_cap=unit, base_reserve_units=unit * 0.2
        )
        value["env"].update(
            WATCH6_SHARD=str(shard),
            WATCH6_FIXTURE="1" if probe else "0",
            WATCH6_WORKERS=str(4 if probe else forecast["workers"]),
            WATCH6_SHARDS=str(6 if probe else forecast["shards"]),
            WATCH6_BASE_MINUTES=str(base),
        )
        if probe:
            value["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/watch6_search_efficacy"
            )
            value["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(value),
            }
        else:
            qualification = out / "output-canary/receipt.json" if pending_canary else canary_receipt
            value["canary"] = {
                "schema": canary.SCHEMA,
                "receipt": {
                    "path": str(qualification.resolve()),
                    "sha256": "0" * 64 if pending_canary else digest(qualification),
                },
            }
            value["env"]["WATCH6_MEASUREMENTS"] = json.dumps(report, sort_keys=True)
            value["env"]["WATCH6_FROZEN_LAYOUT"] = json.dumps(
                {key: forecast[key] for key in ("workers", "shards", "base_minutes_per_shard")},
                sort_keys=True,
            )
            # Re-derive transport/upload/export/hardware evidence; a fabricated
            # report or a PASS string alone can never prepare main manifests.
            if pending_canary:
                value["pending_qualification"] = True
            else:
                canary.check(value, p.ROOT)
        from research.watch6_search_efficacy.admission import admit

        upload_report = admit(value, pending_qualification=pending_canary and not probe)
        publish_json(out / f"{label}-upload-preflight.json", upload_report)
        # The uploaded contract and complete input inventory are identical for
        # canary and all shards; per-shard controls are pinned manifest env.
        suffix = ".pending.json" if pending_canary and not probe else ".json"
        (out / f"{label}{suffix}").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--mirror-receipt", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--empirical-costs", type=Path)
    parser.add_argument("--pending-canary", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--previous-receipts", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.out.resolve(),
                args.benchmark,
                args.commit,
                args.approval,
                args.ledger,
                args.attempt,
                args.previous_receipts,
                args.mirror_receipt,
                args.canary_receipt,
                args.main,
                args.empirical_costs,
                args.pending_canary,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
