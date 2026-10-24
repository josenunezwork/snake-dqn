"""Mac-only V8 initial-main cap amendment; never changes the shipped runtime.

Run with PYTHONPATH pointing at BROKER. The private canary checker below is a
literal copy of the frozen checker, changing only its 20-minute guard to 50.
The AST regression test enforces that boundary. No frozen module is patched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shlex
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from research.colab_cli.supervisor import validate as supervisor_validate
from research.gpu_sim_spike import v8_split_pack as pack
from research.gpu_sim_spike import v8_split_units
from research.gpu_sim_spike.v8_split_pack import SCRIPT, ManifestRefused, _verify_file, sha256_file
from research.metadata_archive import unpack
from tools.ops.dispatch import require_resume_policy
from tools.ops.qualification import OPS, check

BASE = Path("/Users/josenunez/Projects/ml")
BROKER = BASE / "snake-dqn-v8-metadata-qualified-02.noindex"
SOURCE = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-v8-metadata-20261024-02"
ART = BASE / "snake-dqn-artifacts/big-lever/exp/gpu-v8-main-amended-20261024-01"
PIN = "669405e2bef629f7f69bc951cf49486d8b0a87bf"
RULING = SOURCE.parent / "JUDGMENT_V8_SPLIT_MAIN_20261024.md"
RULING_SHA = "f8c1386dc6b82c3cd0f0b74a491b18433d78da48cd3fb28472f16c3a68db22b4"
TEMPLATE_SHA = "ae2070abbccaced79e0aa45f2f4e4215162a4fd0540ee82831d0d98c3d9b123e"
ARCHIVE_SHA = "770fd1daa01a0100d1e84551d1d34c483ba615234b5fe976824256e6b636f315"
CANARY_SHA = "57339dc463486a947ca23301e1d4ac13bf21b6089fcd69844c1581dfe1fae288"
JOB = "gpu-v8-split-main-669405e2bef6-amended-01"
AMENDMENT = Path(__file__).resolve().with_name("V8_SPLIT_MAIN_CAP_AMENDMENT_20261024.md")
TESTS = Path(__file__).resolve().parents[1] / "tests/test_v8_split_main_amend.py"


def authority(path: Path, digest: str) -> dict[str, str]:
    """Refuse missing/foreign authority before accepting the amended guard."""
    if path.resolve() != RULING or digest != RULING_SHA or sha256_file(path) != RULING_SHA:
        raise ManifestRefused("missing or wrong cap-amendment authority hash")
    return {"path": str(path), "sha256": digest}


def check_forecast(elapsed: float, benchmark: float) -> float:
    """Recompute the registered formula, without reducing measured work."""
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (elapsed, benchmark)
        )
        or elapsed < benchmark
    ):
        raise ManifestRefused("invalid canary timing")
    minutes = max(180.0, elapsed - benchmark + benchmark * 63) / 60
    if minutes > 50:
        raise ManifestRefused("cap amendment cannot admit a forecast >50 minutes")
    return minutes


def _canary_binding(receipt: Path, commit: str, preflight: dict[str, Any]) -> dict[str, Any]:
    if not receipt.is_file():
        raise ManifestRefused("same-pin V8 split canary receipt is absent")
    value = json.loads(receipt.read_text())
    manifest_path = receipt.parent / "manifest.json"
    if (
        value.get("status") != "PASS"
        or value.get("commit") != commit
        or value.get("stop_verified") is not True
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != value.get("manifest_sha256")
    ):
        raise ManifestRefused("same-pin native V8 split canary PASS/release receipt required")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("script") != SCRIPT
        or manifest.get("commit") != commit
        or manifest.get("mac_preflight") != preflight
    ):
        raise ManifestRefused("V8 split canary receipt is not from this exact packet")
    producer = receipt.parent / "exports" / "v8-split-bench.json"
    _verify_file(producer)
    output = json.loads(producer.read_text())
    if (
        output.get("status") != "MEASURED_PENDING_MAC_REVIEW"
        or output.get("commit") != commit
        or output.get("namespace") != v8_split_units.namespace(commit, canary=True)
        or output.get("completed_units") != list(v8_split_units.allocation(canary=True))
    ):
        raise ManifestRefused("V8 split canary producer evidence is incomplete")
    native = receipt.parent / "exports" / "native-parity.json"
    _verify_file(native)
    native_body = json.loads(native.read_text())
    if (
        native_body.get("status") != "PASS"
        or native_body.get("native_cuda_executed") is not True
        or native_body.get("commit") != commit
        or native_body.get("source_sha256") != output.get("source_sha256")
    ):
        raise ManifestRefused("V8 split canary native parity gate did not PASS")
    inventory = value.get("exports")
    if not isinstance(inventory, list):
        raise ManifestRefused("V8 split canary terminal export inventory is absent")
    listed = {row.get("path"): row for row in inventory if isinstance(row, dict)}
    if len(listed) != len(inventory):
        raise ManifestRefused("V8 split canary terminal export inventory is malformed")
    actual_payloads = {
        path.relative_to(receipt.parent / "exports").as_posix()
        for path in (receipt.parent / "exports").rglob("*")
        if path.is_file() and not path.name.endswith((".sha256", ".complete"))
    }
    if set(listed) != actual_payloads:
        raise ManifestRefused("V8 split canary terminal export inventory is incomplete")
    for name, row in listed.items():
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ManifestRefused("V8 split canary terminal export path is unsafe")
        path = receipt.parent / "exports" / name
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("sha256") != sha256_file(path)
            or row.get("size") != path.stat().st_size
        ):
            raise ManifestRefused("V8 split canary terminal export inventory differs")
    for name in ("v8-split-bench.json", "native-parity.json", "discrepancy-register.json"):
        if not isinstance(listed.get(name), dict):
            raise ManifestRefused("V8 split canary terminal export inventory differs")
    from research.gpu_sim_spike import v8_split_analyze

    analysis = v8_split_analyze.aggregate([receipt.parent / "exports"], commit, canary=True)
    if analysis.get("verdict") != "PASS_CANARY":
        raise ManifestRefused("independent V8 split canary reducer did not PASS_CANARY")
    elapsed, benchmark = value.get("elapsed_seconds"), output.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or not isinstance(benchmark, (int, float)):
        raise ManifestRefused("V8 split canary lacks operational timing for main estimate")
    # One canary E64 sample exists per roster. Main uses nine timing passes per
    # arm and frozen E64/E256/E1024 shape factors 1/4/16: 9*(1+4+16)/3 = 63.
    main_seconds = max(180.0, elapsed - benchmark + benchmark * 63)
    if main_seconds > 50 * 60:
        raise ManifestRefused(
            "canary operational timing cannot support the bounded 50-minute main estimate"
        )
    return {
        "path": str(receipt.resolve()),
        "sha256": sha256_file(receipt),
        "analysis": analysis,
        "main_estimate_minutes": max(3, math.ceil(main_seconds / 60)),
    }


def template() -> dict[str, Any]:
    path = SOURCE / "v8-split-main-template.json"
    if _verify_file(path) != TEMPLATE_SHA:
        raise ManifestRefused("accepted main template differs")
    return json.loads(path.read_text())


def no_main_evidence() -> dict[str, Any]:
    """Inspect broker history by identity, never open scientific main payloads."""
    inspected = []
    for directory in (OPS / "manifests", OPS / "state", OPS / "dispatch"):
        for path in sorted(directory.rglob("*.json")):
            if "v8" not in str(path).lower():
                continue
            value = json.loads(path.read_text())
            if (
                value.get("commit") == PIN
                and value.get("env", {}).get("V8_SPLIT_BENCH_CANARY") == "0"
            ):
                raise ManifestRefused(f"main already reserved or run: {path}")
            inspected.append({"path": str(path), "sha256": sha256_file(path)})
    for path in (
        SOURCE / "main-output",
        SOURCE / "state-main",
        ART / "main-output",
        ART / "state-main",
        OPS / "dispatch" / JOB,
    ):
        if path.exists():
            raise ManifestRefused(f"initial main identity already used: {path}")
    return {
        "no_main_unit_has_run": True,
        "scope": "local package and broker history",
        "inspected": inspected,
    }


def accept_package() -> dict[str, Any]:
    """Run the preserved package path at the qualified, clean runtime pin."""
    if pack.ROOT != BROKER:
        raise ManifestRefused("run with PYTHONPATH set to the exact qualified broker")
    pack.require_source_pin(PIN)
    qualified = check(BROKER, OPS / "colab-cli-qualified.json", ref=PIN)
    preflight = pack.preflight_binding(SOURCE / "mac-preflight/mac-preflight.json", PIN)
    original = template()
    if original["mac_preflight"] != preflight:
        raise ManifestRefused("template preflight differs")
    receipt = SOURCE / "canary-output/receipt.json"
    if sha256_file(receipt) != CANARY_SHA:
        raise ManifestRefused("passed canary receipt differs")
    binding = _canary_binding(receipt, PIN, preflight)
    operational = json.loads(receipt.read_text())
    producer = json.loads((receipt.parent / "exports/v8-split-bench.json").read_text())
    forecast = check_forecast(operational["elapsed_seconds"], producer["elapsed_seconds"])
    if producer["source_sha256"] != preflight["source_sha256"]:
        raise ManifestRefused("native canary source differs from Mac preflight")
    metadata = next(row for row in original["inputs"] if row["dest"] == "metadata.tar")
    if metadata["sha256"] != ARCHIVE_SHA or _verify_file(Path(metadata["path"])) != ARCHIVE_SHA:
        raise ManifestRefused("exact accepted metadata archive differs")
    with tempfile.TemporaryDirectory(prefix="v8-amendment-archive-") as temporary:
        root = Path(temporary).resolve()
        archive = unpack(Path(metadata["path"]), root, ARCHIVE_SHA)
        for relative, canary in (
            ("inputs/verified-ledger.json", False),
            ("canary-inputs/verified-ledger.json", True),
        ):
            ledger = root / relative
            loaded = v8_split_units.load_ledger(
                ledger,
                v8_split_units.namespace(PIN, canary=canary),
                v8_split_units.allocation(canary=canary),
            )
            if loaded or json.loads(ledger.read_text())["units"] != []:
                raise ManifestRefused("initial ledgers must be empty")
        for row in pack._ruling_inputs() + pack._preflight_inputs(preflight):
            if sha256_file(root / row["dest"]) != row["sha256"]:
                raise ManifestRefused("archived protocol/ruling/discrepancy differs")
    return {
        "canary": binding,
        "forecast_minutes": forecast,
        "canary_elapsed_seconds": operational["elapsed_seconds"],
        "benchmark_elapsed_seconds": producer["elapsed_seconds"],
        "qualified_transport_tree": qualified,
        "archive": archive,
        "pre_main": no_main_evidence(),
    }


def build(evidence: dict[str, Any], ruling: dict[str, str]) -> dict[str, Any]:
    """Apply exactly the ruled local manifest delta to the immutable template."""
    result = copy.deepcopy(template())
    result["job_id"] = JOB
    result["output_dir"] = str(ART / "main-output")
    result["requirements"]["est_minutes"] = 50
    result["timeout_minutes"] = 120
    result["budget"].update(unit_cap=20, reserve_units=10 / 3)
    result["env"]["V8_SPLIT_BENCH_SECONDS"] = "5400"
    result["v8_split"]["effective_envelope"] = {"wall_minutes": 144, "units": 24}
    result["v8_split"]["main"] = {"wall_minutes": 120, "units": 20, "estimate_minutes_max": 50}
    result["canary"] = {"receipt": evidence["canary"]}
    result["cap_amendment"] = {
        "ruling": ruling,
        "sibling_amendment": {"path": str(AMENDMENT), "sha256": sha256_file(AMENDMENT)},
        "preparation_receipt": str(ART / "PREPARATION.json"),
        "forecast_minutes": evidence["forecast_minutes"],
        "base_envelopes": {"canary": [12, 2], "main": [60, 10], "packet": [72, 12]},
        "effective_envelopes": {"canary": [24, 4], "main": [120, 20], "packet": [144, 24]},
        "operator_accounting_required_before_allocation": True,
        "initial_main_only": True,
    }
    return result


def validate_final(value: dict[str, Any]) -> dict[str, Any]:
    """Refuse every delta outside the exact prospective amendment."""
    binding = value.get("cap_amendment", {}).get("ruling", {})
    ruling = authority(Path(binding.get("path", "")), binding.get("sha256", ""))
    evidence = accept_package()
    expected = build(evidence, ruling)
    if value != expected:
        raise ManifestRefused(
            "manifest differs from exact amended identity/budget/scientific contract"
        )
    require_resume_policy(value)
    supervisor_validate(value, repo=BROKER)
    return evidence


def implementation() -> dict[str, dict[str, str]]:
    return {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in (
            ("source", Path(__file__).resolve()),
            ("tests", TESTS),
            ("amendment", AMENDMENT),
        )
    }


def prepare() -> dict[str, Any]:
    """Create a fresh local package after all offline checks; never allocate."""
    ruling = authority(RULING, RULING_SHA)
    evidence = accept_package()
    value = build(evidence, ruling)
    validate_final(value)
    if ART.exists():
        raise ManifestRefused("fresh sibling package already exists")
    ART.mkdir()
    v8_split_units.publish_json(ART / "v8-split-main.json", value)
    command = dispatch_script()
    pack._write(ART / "dispatch-main.sh", command)
    (ART / "dispatch-main.sh").chmod(0o755)
    pack._write(ART / "DISPATCH.txt", f"nice -n 15 bash {ART}/dispatch-main.sh\n")
    receipt = {
        "status": "PREPARED_OFFLINE_VALIDATED_NOT_LIVE_ADMITTED",
        "allocations": 0,
        "runtime_commit": PIN,
        "ruling": ruling,
        "evidence": evidence,
        "implementation": implementation(),
        "manifest_sha256": sha256_file(ART / "v8-split-main.json"),
        "dispatch_sha256": sha256_file(ART / "dispatch-main.sh"),
        "operator_accounting": "Required per ruling admission step 6 before allocation",
    }
    v8_split_units.publish_json(ART / "PREPARATION.json", receipt)
    return receipt


def verify_preparation() -> None:
    path = ART / "PREPARATION.json"
    _verify_file(path)
    value = json.loads(path.read_text())
    if (
        value["implementation"] != implementation()
        or value["manifest_sha256"] != _verify_file(ART / "v8-split-main.json")
        or value["dispatch_sha256"] != _verify_file(ART / "dispatch-main.sh")
    ):
        raise ManifestRefused("prepared source/tests/manifest/dispatch binding differs")
    validate_final(json.loads((ART / "v8-split-main.json").read_text()))


def dispatch_script() -> str:
    """Emit operator handoff; local admission evidence is never uploaded."""
    source = shlex.quote(str(Path(__file__).resolve()))
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {BROKER}
export PYTHONPATH={BROKER}
export PYTHONDONTWRITEBYTECODE=1
PY={pack.PRIMARY_PY}
"$PY" {source} verify
# Operator/Codex-V records original T0, history, liabilities, all remaining
# parent allowances, account balance/rates/assignments and calendar deadline.
# Full 120 min / 20 u must fit, including reserve; no clock/counter reset.
"$PY" {source} admission
exec {BROKER}/tools/ops/dispatch.sh {JOB} \\
  {ART}/v8-split-main.json --broker {BROKER} --cwd {BROKER} \\
  --state-dir {ART}/state-main --dispatch-txt {ART}/DISPATCH.txt
"""


def admission() -> None:
    """Require the operator's independently accepted live accounting record."""
    review_path = ART / "CODEX_V_LIGHT.json"
    _verify_file(review_path)
    reviewed_package = json.loads(review_path.read_text())
    if (
        reviewed_package.get("status") != "ACCEPT_OFFLINE"
        or reviewed_package.get("preparation_sha256") != _verify_file(ART / "PREPARATION.json")
        or reviewed_package.get("manifest_sha256") != _verify_file(ART / "v8-split-main.json")
        or reviewed_package.get("implementation") != implementation()
    ):
        raise ManifestRefused("INCOMPLETE_ADMISSION: exact package lacks Codex-V acceptance")
    path = ART / "OPERATOR_ADMISSION.json"
    _verify_file(path)
    body = json.loads(path.read_text())
    if (
        body.get("status") != "ADMIT_INITIAL_MAIN"
        or body.get("manifest_sha256") != sha256_file(ART / "v8-split-main.json")
        or body.get("ruling_sha256") != RULING_SHA
        or body.get("no_main_unit_has_run") is not True
        or body.get("previous_endpoint_release_verified") is not True
        or body.get("a100_serial_sprint") is not True
    ):
        raise ManifestRefused("INCOMPLETE_ADMISSION: operator record not admitted")
    for key in (
        "original_t0",
        "historical_expenditure",
        "liabilities_and_reservations",
        "q2_q4_remaining",
        "v8_series_remaining",
        "live_unencumbered_units",
        "current_assignments_and_rates",
        "calendar_deadline_utc",
        "accounting_evidence",
    ):
        if key not in body or body[key] is None or body[key] == "":
            raise ManifestRefused(f"INCOMPLETE_ADMISSION: missing {key}")
    if body.get("reserved_main") != {"wall_minutes": 120, "units": 20}:
        raise ManifestRefused("INCOMPLETE_ADMISSION: full main cap not reserved")
    if body.get("fits_all_original_parent_caps_and_deadlines") is not True:
        raise ManifestRefused("INCOMPLETE_ADMISSION: parent allowance/deadline")
    accounting_time(body, datetime.now(timezone.utc))
    review = body.get("codex_v_review", {})
    review_path = Path(review.get("path", ""))
    if _verify_file(review_path) != review.get("sha256"):
        raise ManifestRefused("INCOMPLETE_ADMISSION: accounting review binding")
    accepted = json.loads(review_path.read_text())
    # Bind the accounting body without its review pointer to avoid a hash cycle.
    reviewed = dict(body)
    reviewed.pop("codex_v_review", None)
    digest = hashlib.sha256(
        json.dumps(reviewed, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if accepted.get("status") != "ACCEPT" or accepted.get("accounting_sha256") != digest:
        raise ManifestRefused("INCOMPLETE_ADMISSION: Codex-V accounting not accepted")


def accounting_time(body: dict[str, Any], now: datetime) -> None:
    """Refuse expired accounting or a full ceiling beyond the original deadline."""
    parsed = {}
    for key in ("accounting_as_of_utc", "valid_until_utc", "calendar_deadline_utc"):
        try:
            value = datetime.fromisoformat(body[key])
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError("UTC offset required")
            parsed[key] = value
        except (KeyError, ValueError, TypeError) as error:
            raise ManifestRefused(f"INCOMPLETE_ADMISSION: invalid {key}") from error
    if not parsed["accounting_as_of_utc"] <= now <= parsed["valid_until_utc"]:
        raise ManifestRefused("INCOMPLETE_ADMISSION: expired or future accounting")
    if now + timedelta(minutes=120) > parsed["calendar_deadline_utc"]:
        raise ManifestRefused("INCOMPLETE_ADMISSION: full main exceeds calendar deadline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify", "admission"))
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare()
    elif args.operation == "verify":
        verify_preparation()
    else:
        admission()
    print(json.dumps({"status": "PASS", "operation": args.operation, "allocations": 0}))


if __name__ == "__main__":
    main()
