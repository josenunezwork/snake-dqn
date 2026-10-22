"""Publish ten fresh V3 transports and local qualification, never dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from research.bl_upload import qualify
from research.colab_cli.inputs import preflight
from research.colab_cli.supervisor import validate
from research.t34_o2_harvest.bootstrap_preflight_v3 import run_bootstrap
from research.t34_o2_harvest.frozen_sources_v3 import PIN, require_registration
from research.t34_o2_harvest.make_manifest_v3 import ROOT, make_manifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(path: Path) -> None:
    """Create publication sidecars without replacing any existing evidence."""
    with Path(str(path) + ".sha256").open("x") as stream:
        stream.write(f"{sha(path)}  {path.name}\n")
    with Path(str(path) + ".complete").open("x") as stream:
        stream.write("complete\n")


def save(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    seal(path)


def prepare(artifacts: Path) -> None:
    """Preserve V2 evidence and create-only all new manifests and receipts."""
    artifacts = artifacts.resolve(strict=True)
    registration = ROOT / "research/t34_o2_harvest/REGISTRATION_O2_V2.json"
    require_registration(registration)
    manifests = artifacts / "v3-manifests"
    # Refuse the whole attempt before creating any V3 publication if used.
    planned = [manifests, artifacts / "v3-manifest-validation.json"]
    planned += [artifacts / f"v3-chunk-{i:02d}" for i in range(10)]
    planned += [artifacts / f"v3-supervisor-chunk-{i:02d}" for i in range(10)]
    if any(path.exists() for path in planned):
        raise FileExistsError("V3 publication/output/state paths must all be fresh")
    manifests.mkdir()
    results = []
    public = None
    dispatch = [
        "O2 V3 — PREPARED / VALIDATED / NOT DISPATCHED",
        "Mechanical transport recovery only; registration and all rules remain V2.",
        f"Runtime commit: {PIN}",
        f"Registration: {registration}",
        f"Registration SHA-256: {sha(registration)}",
        "The V2 registration, bank, caps, scientific launcher and one look are unchanged.",
        "CUDA visible for bootstrap; frozen launch_v2.sh hides it before CPU workers.",
        "Control inputs use hash/size-bound public URLs; only the tiny timing receipt uploads.",
        "No Colab/account query, allocation, execution or analysis was performed.",
        "Local bootstrap uses offline installer/hardware/runner probes; GPU remains unqualified.",
        "Before future admission refresh assignment rates, cap/budget headroom and qualification.",
        "Prior failed-attempt spend remains chargeable; unchanged job caps are not fresh funding.",
        "Never dispatch more than three account assignments; no automatic retry.",
        "",
        "LOCAL VALIDATION (no allocation):",
    ]
    launch = ROOT / "research/colab_cli/launch.sh"
    colab = Path.home() / ".local/bin/colab"
    run_commands = []
    for index in range(10):
        manifest = make_manifest(
            registration=registration,
            catalog=ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json",
            preflight_receipt=artifacts / "v2-preflight-serial/receipt.json",
            output_dir=artifacts / f"v3-chunk-{index:02d}",
            commit=PIN,
            chunk=index,
        )
        validate(manifest, repo=ROOT)
        bootstrap = run_bootstrap(manifest)
        if public is None:
            # HTTPS GETs only: validate the exact public objects on the Mac.
            # The qualified runtime repeats these checks before bootstrap.
            public = preflight(i for i in manifest["inputs"] if "url" in i)
            save(artifacts / "v3-public-input-preflight.json", public)
        path = manifests / f"chunk-{index:02d}.json"
        ready_path = artifacts / f"v3-upload-qualification-{index:02d}.json"
        ready = qualify(manifest, repo=ROOT, receipt_path=ready_path)
        seal(ready_path)
        save(path, manifest)
        results.append(
            {
                "manifest": str(path),
                "manifest_sha256": sha(path),
                "bootstrap": bootstrap,
                "upload_receipt": str(ready_path),
                "upload_receipt_sha256": sha(ready_path),
                "normal_upload_bytes": ready["normal_upload_bytes"],
                "normal_forecast_seconds": ready["normal_forecast_seconds"],
                "fallback_total_bound_seconds": ready["fallback_total_bound_seconds"],
            }
        )
        state = artifacts / f"v3-supervisor-chunk-{index:02d}"
        command = f"{launch} --state-dir {state} --colab {colab}"
        dispatch.append(f"{launch} --check --state-dir {state} --colab {colab} validate {path}")
        run_commands.append(f"# tmux new-session -d -s o2-v3-{index:02d} '{command} run {path}'")
        print(f"VALIDATED_NOT_DISPATCHED {index:02d}", flush=True)
    save(
        artifacts / "v3-manifest-validation.json",
        {
            "status": "PASS_NOT_DISPATCHED",
            "runtime_commit": PIN,
            "registration_sha256": sha(registration),
            "manifests": results,
        },
    )
    dispatch += ["", "FUTURE DISPATCH — NOT RUN; NEW DISPATCH INSTRUCTION REQUIRED:", *run_commands]
    runs = " ".join(f"--run-dir {artifacts}/v3-chunk-{i:02d}" for i in range(10))
    dispatch += [
        "",
        "AFTER ALL RUNS: verify terminal receipts, immutable exports and session disappearance.",
        "Use the additive V3 transport analyzer, retaining the SAME V2 one-look marker:",
        f"# cd {ROOT}",
        f"# {Path(__import__('sys').executable)} -m research.t34_o2_harvest.analyze_v3 "
        f"--registration {registration} {runs} --out-dir {artifacts}/v3-analysis",
        "Serving/release remains owner-only. All old logs/receipts remain preserved.",
    ]
    dispatch_path = manifests / "DISPATCH.txt"
    with dispatch_path.open("x") as stream:
        stream.write("\n".join(dispatch) + "\n")
    seal(dispatch_path)
    preserved = artifacts / "v2-dispatch-preserved"
    preserved.mkdir()
    for suffix in ("", ".sha256", ".complete"):
        old = artifacts / ("DISPATCH.txt" + suffix)
        shutil.copy2(old, preserved / old.name)
    # Explicitly requested pointer update, after every qualification completed.
    for suffix in ("", ".sha256", ".complete"):
        shutil.copy2(Path(str(dispatch_path) + suffix), artifacts / ("DISPATCH.txt" + suffix))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    prepare(parser.parse_args().artifacts)


if __name__ == "__main__":
    main()
