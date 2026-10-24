"""I0R §2 precondition probe: does the qualified canary matcher accept a resume ledger?

JUDGMENT_I0_RESIZE_20261024 (RESTRUCTURE_I0_AS_I0R) admits I0 as serial resumable
attempts. Each attempt after the first binds the Mac-verified ledger of earlier
attempts. Section 2 makes it a precondition that the qualified canary exact-input
matcher accepts a non-empty resume ledger as the declared input; if it does not,
"I0R is HOLD and needs requalification. No matcher exception is granted."

Host-only and zero-allocation. Using the unchanged qualified runtime/broker
(``snake-dqn-gpuint-qualified-03.noindex`` at 968f0276), this builds two
throwaway manifests in a temporary directory from the frozen main template and
the real I0 canary receipt:

* ``attempt1-empty-ledger``: the I0R attempt shape (INTEGRATION_BENCH_SECONDS=1440,
  est 30, timeout 60, cap 10 u at 10 u/h) with the original empty ledger;
* ``attempt2-nonempty-ledger``: the same, with a ledger that holds one synthetic,
  outcome-free verified unit, rebundled into ``metadata.tar`` exactly as the
  frozen ``integration_pack.manifest`` would bundle it.

It runs the qualified ``canary.check`` and ``supervisor.validate`` on each and,
unless ``--no-dispatch``, the governed ``tools/ops/dispatch.sh --dry-run``. No
packet, reservation, provider query or dispatch is made. The manifests are
diagnostics only and are never published as admissible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/Users/josenunez/Projects/ml")
RUN = BASE / "snake-dqn-gpuint-qualified-03.noindex"
PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"
EXP = BASE / "snake-dqn-artifacts/big-lever/exp"
PACKET = EXP / "gpu-integration-i0-20261024-03/packet"
RECEIPT = PACKET / "canary-output/receipt.json"
RECEIPT_SHA = "d230135d78fdf51f8647659d4926b19677bc47136903172d3cff3fcc91157ad0"
TEMPLATE = PACKET / "integration-main-template.json"
METADATA = PACKET / "metadata-main.tar"
METADATA_SHA = "350f4de106285510fcb611fd974ac2b3801bc28b104cdddd851987937bdcb327"
RULING = EXP / "JUDGMENT_I0_RESIZE_20261024.md"
RULING_SHA = "dacd2f4b1abf5505d4317caed6d81102794d653eb35168cc1b5433eb6324f2c2"
LEDGER_DEST = "inputs/verified-ledger.json"
INVENTORY = ".metadata-manifest.json"
# I0R attempt parameters (ruling §4); diagnostics here, never raised.
ATTEMPT = {
    "INTEGRATION_BENCH_SECONDS": "1440",
    "est_minutes": 30,
    "timeout_minutes": 60,
    "unit_cap": 10.0,
    "rate_units_per_hour": 10.0,
}
SYNTHETIC_UNIT = "cell-frozen-e1024"
HOLD = "HOLD_RESUME_LEDGER_REFUSED_BY_QUALIFIED_MATCHER"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(path: Path, digest: str, label: str) -> dict[str, str]:
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise SystemExit(f"{label} identity differs: {path}")
    return {"path": str(path), "sha256": digest}


def modules() -> dict[str, Any]:
    """Import the qualified broker/runtime modules; refuse any other checkout."""
    if str(RUN) not in sys.path[:1]:
        raise SystemExit("run through tools/i0r-resume-precondition.sh (runtime first)")
    from research import metadata_archive
    from research.colab_cli import canary, supervisor
    from research.gpu_sim_spike import integration_units

    for module in (canary, supervisor, integration_units, metadata_archive):
        if not Path(module.__file__).resolve().is_relative_to(RUN):
            raise SystemExit(f"{module.__name__} did not resolve from the qualified checkout")
    return {
        "canary": canary,
        "supervisor": supervisor,
        "units": integration_units,
        "archive": metadata_archive,
    }


def attempt_shape(template: dict[str, Any], job: str) -> dict[str, Any]:
    """Template + real canary binding + the I0R attempt parameters only."""
    value = copy.deepcopy(template)
    value.pop("admission_reason", None)
    value["scientific_admission"] = True
    value["job_id"] = job
    value["canary"] = {"receipt": {"path": str(RECEIPT), "sha256": RECEIPT_SHA}}
    value["env"]["INTEGRATION_BENCH_SECONDS"] = ATTEMPT["INTEGRATION_BENCH_SECONDS"]
    value["requirements"]["est_minutes"] = ATTEMPT["est_minutes"]
    value["timeout_minutes"] = ATTEMPT["timeout_minutes"]
    value["budget"]["unit_cap"] = ATTEMPT["unit_cap"]
    value["budget"]["rate_units_per_hour"] = ATTEMPT["rate_units_per_hour"]
    value["budget"]["reserve_units"] = ATTEMPT["unit_cap"] / 6  # template ratio (20 -> 10/3)
    return value


def nonempty_ledger_inputs(
    mods: dict[str, Any], template: dict[str, Any], work: Path
) -> dict[str, Any]:
    """Rebundle the frozen metadata members with a one-unit verified main ledger."""
    units = mods["units"]
    source = work / "metadata-src"
    source.mkdir()
    with tarfile.open(METADATA) as tar:
        tar.extractall(source, filter="data")
    (source / INVENTORY).unlink()
    ledger = source / LEDGER_DEST
    for suffix in ("", ".sha256", ".complete"):
        Path(str(ledger) + suffix).unlink()
    ns, cells = units.namespace(PIN), units.allocation()
    payload = units.publish_unit(
        ledger.parent, ns, cells, SYNTHETIC_UNIT, {"synthetic_precondition_probe": True}
    )
    body = units.empty_ledger(ns, cells)
    body["units"].append(
        {
            "unit_id": SYNTHETIC_UNIT,
            "path": f"{units.UNIT_DIRECTORY}/{SYNTHETIC_UNIT}.json",
            "sha256": sha(payload),
            "size": payload.stat().st_size,
        }
    )
    units.publish_json(ledger, body)
    loaded = units.load_ledger(ledger, ns, cells)  # what the producer would read
    rows = [
        {"path": str(path), "sha256": sha(path), "dest": path.relative_to(source).as_posix()}
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    rows += [row for row in template["inputs"] if row["dest"] != "metadata.tar"]
    inputs = mods["archive"].bundle_inputs(rows, work / "metadata-main.tar")
    archive = next(row for row in inputs if row["dest"] == "metadata.tar")
    return {
        "inputs": inputs,
        "metadata_sha256": archive["sha256"],
        "metadata_size": Path(archive["path"]).stat().st_size,
        "ledger_units": sorted(loaded),
    }


def check(mods: dict[str, Any], manifest: dict[str, Any], work: Path) -> dict[str, str]:
    value = dict(manifest, output_dir=str(work / "never-created-output"))
    result = {}
    for name, call in (
        ("canary_check", lambda: mods["canary"].check(copy.deepcopy(value), RUN)),
        ("supervisor_validate", lambda: mods["supervisor"].validate(value, repo=RUN)),
    ):
        try:
            call()
            result[name] = "ACCEPT"
        except Exception as error:  # noqa: BLE001 - record the qualified refusal text
            result[name] = f"REFUSE {type(error).__name__}: {error}"
    return result


def governed_dry_run(manifest: dict[str, Any], work: Path) -> dict[str, Any]:
    folder = work / f"dry-{manifest['job_id']}"
    folder.mkdir()
    value = dict(manifest, output_dir=str(folder / "output"))
    path = folder / "manifest.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    result = subprocess.run(
        [
            str(ROOT / "tools/ops/dispatch.sh"),
            manifest["job_id"],
            str(path),
            "--broker",
            str(RUN),
            "--cwd",
            str(RUN),
            "--state-dir",
            str(folder / "state"),
            "--dispatch-txt",
            str(folder / "DISPATCH.txt"),
            "--dry-run",
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    return {"returncode": result.returncode, "tail": lines[-2:]}


def probe(dispatch: bool = True) -> dict[str, Any]:
    bindings = {
        "ruling": require(RULING, RULING_SHA, "I0R ruling"),
        "canary_receipt": require(RECEIPT, RECEIPT_SHA, "I0 canary receipt"),
        "canary_metadata_archive": require(METADATA, METADATA_SHA, "canary metadata archive"),
        "main_template": {"path": str(TEMPLATE), "sha256": sha(TEMPLATE)},
    }
    mods = modules()
    template = json.loads(TEMPLATE.read_text())
    receipt = json.loads(RECEIPT.read_text())
    proved = receipt["canary_evidence"]["inputs"]
    cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="i0r-precondition-") as directory:
        work = Path(directory)
        first = attempt_shape(template, "i0r-precondition-diag-a1")
        second = attempt_shape(template, "i0r-precondition-diag-a2")
        bundled = nonempty_ledger_inputs(mods, template, work)
        second["inputs"] = bundled.pop("inputs")
        second["env"]["METADATA_ARCHIVE_SHA256"] = bundled["metadata_sha256"]
        for label, manifest, extra in (
            ("attempt1-empty-ledger", first, {"ledger_units": []}),
            ("attempt2-nonempty-ledger", second, bundled),
        ):
            row = dict(extra)
            row["metadata_sha256"] = row.get("metadata_sha256", METADATA_SHA)
            row["input_inventory"] = mods["canary"].input_inventory(manifest)
            row.update(check(mods, manifest, work))
            if dispatch:
                row["governed_dry_run"] = governed_dry_run(manifest, work)
            cases[label] = row

    def accepted(label: str) -> bool:
        row = cases[label]
        return row["canary_check"] == row["supervisor_validate"] == "ACCEPT"

    first_ok, second_ok = accepted("attempt1-empty-ledger"), accepted("attempt2-nonempty-ledger")
    status = (
        "PRECONDITION_MET"
        if first_ok and second_ok
        else HOLD if first_ok else "HOLD_UNEXPECTED_ATTEMPT1_REFUSAL"
    )
    return {
        "schema": "i0r-resume-precondition/v1",
        "status": status,
        "question": "ruling §2: does the qualified canary exact-input matcher accept a "
        "non-empty resume ledger as the declared resume input?",
        "runtime_pin": PIN,
        "bindings": bindings,
        "canary_proved_inputs": proved,
        "attempt_parameters": ATTEMPT,
        "cases": cases,
        "allocations": 0,
        "provider_queries": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dispatch", action="store_true", help="skip governed dry-runs")
    parser.add_argument("--publish", type=Path, help="create-only record path (+ sidecars)")
    args = parser.parse_args()
    result = probe(dispatch=not args.no_dispatch)
    if args.publish:
        from tools import small_bench_operator as operator

        if args.publish.exists():
            raise SystemExit(f"record already published (create-only): {args.publish}")
        result["implementation"] = [
            {"path": str(path), "sha256": sha(path)}
            for path in (
                ROOT / "tools/i0r_resume_precondition.py",
                ROOT / "tools/i0r-resume-precondition.sh",
            )
        ]
        operator.publish_json(args.publish, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
