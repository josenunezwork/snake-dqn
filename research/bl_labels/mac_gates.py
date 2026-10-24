"""Mac placement admission and the create-only, single registered Canary-B look.

Preparing or verifying a ledger never reads branch outcomes. Only ``look-canary``
invokes the registered aggregator, after publishing an immutable look claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

from .exports import (
    _materialize_triplet,
    _verify,
    aggregate_attempts,
    analyze_aggregate,
    completed_session_parents,
    empty_ledger,
    load_ledger,
    publish_json,
)
from .f1_ruler import bank_digest, compare_mac_configs, load_bank
from .mac_package import (
    CALIBRATION_RUNNER_COMMIT,
    CANARY_WALL_SECONDS,
    CANARY_WORKERS,
    F1_BANK_SHA256,
)

ROOT = Path(__file__).resolve().parents[2]
GLOBAL_LOOK_ROOT = ROOT.parent / "snake-dqn-artifacts/bl-labels-f0/canary-b-single-look"
CODE_PATHS = tuple(
    "research/bl_labels/" + name
    for name in (
        "factory.py",
        "exports.py",
        "runtime.py",
        "f1_ruler.py",
        "f1_units.py",
        "mac_driver.py",
        "mac_package.py",
        "mac_gates.py",
        "replay_audit.py",
    )
)


def _read(path: Path) -> dict:
    _verify(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Mac gate expects a verified JSON object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pin(pack: Path) -> str:
    control = _read(pack / "control.json")
    pin = control.get("source_commit")
    if not isinstance(pin, str) or len(pin) != 40:
        raise ValueError("Mac package lacks frozen source commit")
    for name in CODE_PATHS:
        frozen = subprocess.check_output(["git", "show", f"{pin}:{name}"], cwd=ROOT)
        if frozen != (ROOT / name).read_bytes():
            raise ValueError(f"Mac executable source drift: {name}")
    return pin


def _f1(pack: Path, output: Path, pin: str) -> tuple[dict, dict]:
    bank_path = Path(_read(pack / "canary-b-manifest.json")["f1_bank_path"])
    if bank_digest(bank_path) != F1_BANK_SHA256:
        raise ValueError("foreign frozen F1 bank")
    bank = load_bank(bank_path)
    result = _read(output / "runtime-result.json")
    if (
        result.get("status") != "COMPLETE"
        or result.get("platform") != "mac"
        or result.get("source_commit") != pin
        or result.get("bank_sha256") != F1_BANK_SHA256
        or result.get("transport_probe") is not False
    ):
        raise ValueError("Mac F1 worker-invariance gate has not passed")
    configs = result.get("configs", [])
    for config in configs:
        if config.get("source_commit") != pin or config.get("bank_sha256") != F1_BANK_SHA256:
            raise ValueError("foreign Mac F1 cell source/bank")
    comparison = compare_mac_configs(configs, [r["root_id"] for r in bank["roots"]])
    if result.get("mac_equivalence") != comparison:
        raise ValueError("Mac F1 comparison receipt differs from verified cells")
    return result, comparison


def _approval(path: Path) -> None:
    from dataclasses import asdict

    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import canonical_sha, require_judgment

    require_judgment(
        path, ROOT / "research/bl_residual/F4_REGISTRATION.md", canonical_sha(asdict(FitRecipe()))
    )


def _approved_control(pack: Path, stage: str, bindings: dict) -> dict:
    name = "control.json" if stage == "canary-b" else "d256-control.json"
    original = _read(pack / name)
    value = copy.deepcopy(original)
    if stage == "d256":
        look = _read(Path(bindings["f0"]["path"]))
        value["operational_canary_rate"] = (
            look["analysis"]["rescue_bearing_lives_per_mac_worker_hour"] * CANARY_WORKERS
        )
    value["mac_gate"] = {
        "stage": stage,
        "status": "APPROVED",
        "package_root": str(pack.resolve()),
        "original_control_sha256": _sha(pack / name),
        "bindings": bindings,
    }
    return value


def admit_canary(pack: Path, f1_output: Path, out: Path) -> dict:
    pin = _source_pin(pack)
    _f1(pack, f1_output, pin)
    path = f1_output / "runtime-result.json"
    value = _approved_control(
        pack, "canary-b", {"f1": {"path": str(path.resolve()), "sha256": _verify(path)}}
    )
    publish_json(out, value)
    return value


def require_admission(control: dict) -> None:
    """Recheck original control bytes and every admission dependency before launch."""
    gate = control.get("mac_gate")
    if gate is None:
        return  # Tiny development fixtures do not impersonate a registered package.
    if not isinstance(gate, dict) or gate.get("status") != "APPROVED":
        raise ValueError("Mac factory HOLD: required placement/F4 gates have not cleared")
    pack, stage = Path(gate["package_root"]), gate["stage"]
    if stage in {"d1024", "d4096"}:
        _require_future_admission(control)
        return
    if stage not in {"canary-b", "d256"}:
        raise ValueError("foreign Mac admission stage")
    pin = _source_pin(pack)
    bindings = gate["bindings"]
    f1 = bindings["f1"]
    f1_path = Path(f1["path"])
    if _verify(f1_path) != f1["sha256"]:
        raise ValueError("Mac F1 receipt changed after admission")
    _f1(pack, f1_path.parent, pin)
    if stage == "d256":
        look = bindings["f0"]
        look_path = Path(look["path"])
        if _verify(look_path) != look["sha256"]:
            raise ValueError("Mac F0 receipt changed after admission")
        _require_f0(pack, look_path, pin)
        approval = bindings["f4"]
        approval_path = Path(approval["path"])
        if _sha(approval_path) != approval["sha256"]:
            raise ValueError("F4 approval changed after admission")
        _approval(approval_path)
        audit = bindings["seed_audit"]
        if _verify(Path(audit["path"])) != audit["sha256"]:
            raise ValueError("D256 seed audit changed after admission")
        _seed_audit(pack)
    if control != _approved_control(pack, stage, bindings):
        raise ValueError("Mac admitted control differs from its frozen template")


def _future_source_pin(control: dict) -> str:
    """Bind science to its sealed pin and an operational wrapper to its descendant."""
    pin = control.get("source_commit")
    if not isinstance(pin, str) or len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("future Mac control lacks a full source pin")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("future Mac executable worktree is dirty")
    if head == pin:
        return pin
    operational = os.environ.get("F0_OPERATIONAL_COMMIT")
    science_root = os.environ.get("F0_SCIENCE_ROOT")
    if operational != head or not science_root:
        raise ValueError("future Mac executable is not the clean sealed speed pin")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", pin, head], cwd=ROOT, check=False
        ).returncode
        != 0
    ):
        raise ValueError("future operational wrapper is not descended from sealed science pin")
    from . import frozen_factory

    root = Path(science_root)
    if root.resolve() == ROOT.resolve():
        raise ValueError("future science root must be separate from the operational worktree")
    frozen_factory._validate(root, pin)
    return pin


def _future_calibration(receipt: Path, legacy_pin: str, runner_pin: str) -> str:
    """Accept legacy PASS, ruled admission, or measured W6 fallback evidence."""
    from . import mac_driver

    value = _read(receipt)
    if value.get("schema") == "bl-labels-f0-w12-admission/v1":
        # The ruling's admission record deliberately binds the original FAIL
        # calibration rather than changing its status.  Its verifier replays
        # the cells, source identity, duration and clean host fields.
        from . import w12_admission

        w12_admission.verify(receipt, expected_code_sha=legacy_pin)
        return "PASS"
    expected_runner_sha = hashlib.sha256(
        subprocess.check_output(
            ["git", "show", f"{runner_pin}:research/bl_labels/w12_calibration.py"], cwd=ROOT
        )
    ).hexdigest()
    if (
        value.get("schema") != "bl-labels-f0-w12-calibration/v1"
        or value.get("code_sha") != legacy_pin
        or value.get("bank_sha256") != F1_BANK_SHA256
        or value.get("platform") != "darwin-arm64"
        or value.get("runner_sha256") != expected_runner_sha
        or value.get("owner_approval_sha256")
        != "32255309812d4df71aa4f6e0635cf0307e3356708262f3cbc99948ebf49569e3"
        or value.get("thermal_not_ok") != 0
        or any(
            not isinstance(value.get(key), str)
            or len(value[key]) != 64
            or any(c not in "0123456789abcdef" for c in value[key])
            for key in ("b8_sha256", "frp3_sha256")
        )
    ):
        raise ValueError("future calibration provenance differs; HOLD")
    samples = receipt.parent / "host-samples.jsonl"
    if (
        not samples.is_file()
        or samples.stat().st_size != value.get("host_samples_bytes")
        or _sha(samples) != value.get("host_samples_sha256")
    ):
        raise ValueError("future calibration host samples differ; HOLD")
    if value.get("status") == "PASS":
        from . import w12_calibration

        cells = value.get("cells")
        if not isinstance(cells, list) or w12_calibration.wall_ratio(cells) != value.get(
            "w12_w6_wall_ratio"
        ):
            raise ValueError("future calibration measured cells differ; HOLD")
        mac_driver._verify_w12_calibration(receipt, expected_code_sha=legacy_pin)
        return "PASS"
    duration, ratio = value.get("duration_seconds"), value.get("w12_w6_wall_ratio")
    ratio_fail = (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration >= 1800
        and isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and math.isfinite(ratio)
        and ratio > 1.30
        and value.get("slowdown_trips") == 0
    )
    slowdown_fail = value.get("slowdown_trips") == 1
    if value.get("status") == "FAIL" and (ratio_fail or slowdown_fail):
        from . import w12_calibration

        cells = value.get("cells")
        if not isinstance(cells, list):
            raise ValueError("future measured-failure cells differ; HOLD")
        if ratio_fail and w12_calibration.wall_ratio(cells) != ratio:
            raise ValueError("future measured-failure cells differ; HOLD")
        return "FAIL"
    raise ValueError("future calibration is HOLD, incomplete or mechanically unsafe; HOLD")


def _f4_d1024_header(control: dict, input_dir: Path) -> tuple[str, set[int]]:
    """Validate main-dispatch identity without requiring future D1024 labels.

    ``preflight_main_dispatch`` also demands the requested dose's life
    allocation be ready, which is circular before D1024 collection.  This is
    its metadata-only prefix, followed by the registered predecessor staging
    validator in ``_future_readiness``.
    """
    import dataclasses

    from research.bl_ppo.exports import digest
    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import (
        B8_SHA,
        DOSES,
        allocation_hash,
        canonical_sha,
        read_ledger,
        require_judgment,
    )

    if control.get("schema") != "bl-residual-control/v1" or control.get("synthetic") is not False:
        raise ValueError("D1024 requires a non-synthetic registered F4 control")
    if not isinstance(control.get("namespace"), str) or not control["namespace"]:
        raise ValueError("D1024 F4 control lacks a namespace")
    inputs = control.get("input_hashes")
    if not isinstance(inputs, dict):
        raise ValueError("D1024 F4 control lacks immutable input hashes")
    for name, expected in inputs.items():
        if name.startswith("train-") and name.endswith(".npz"):
            continue
        if (
            Path(name).name != name
            or not isinstance(expected, str)
            or digest(input_dir / name) != expected
        ):
            raise ValueError("D1024 F4 input hash mismatch")
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("D1024 F4 B8 identity mismatch")
    recipe_sha = canonical_sha(dataclasses.asdict(FitRecipe()))
    if control.get("recipe_sha256") != recipe_sha:
        raise ValueError("D1024 F4 recipe binding mismatch")
    require_judgment(input_dir / "judgment.json", input_dir / "F4_REGISTRATION.md", recipe_sha)
    requested = control.get("doses")
    if (
        requested != [1024]
        or any(type(dose) is not int or dose not in DOSES for dose in requested)
        or allocation_hash(control) != control.get("allocation")
    ):
        raise ValueError("D1024 requires a frozen standalone D1024 allocation")
    skip = read_ledger(
        input_dir / "verified-ledger.json", control["namespace"], control["allocation"]
    )
    if not set(skip) <= {DOSES.index(1024)}:
        raise ValueError("D1024 F4 ledger is outside its allocation")
    if digest(input_dir / "verified-ledger.json") != control.get("ledger_sha256"):
        raise ValueError("D1024 F4 ledger binding mismatch")
    return recipe_sha, set(skip)


def _future_readiness(stage: str, f4_control: Path, f4_inputs: Path) -> dict[str, str]:
    """Use the registered F4 staging validator for the D1024 predecessor.

    This deliberately delegates the substantive checks to ``job`` rather than
    accepting an ad-hoc PASS marker: its immutable input-hash table binds the
    D256 fit receipt and source control, then verifies the fault-free fit and
    frozen recipe/allocation identity before this collection can start.
    """
    if stage != "d1024":
        raise ValueError("future stage readiness has no D4096 admission route")
    if not f4_control.is_file() or not f4_inputs.is_dir():
        raise ValueError("D1024 requires its sealed F4 control and input directory")
    canonical_control = f4_inputs / "control.json"
    if f4_control.resolve() != canonical_control.resolve():
        raise ValueError("D1024 F4 control must be the immutable input directory control.json")
    try:
        control = json.loads(f4_control.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("D1024 F4 control is not JSON") from error
    if not isinstance(control, dict):
        raise ValueError("D1024 requires a non-synthetic registered F4 control")
    from research.bl_residual import job

    recipe_sha, skip = _f4_d1024_header(control, f4_inputs)
    # The registered staging validator supplies the D256 source-control and
    # fault-free predecessor checks. It runs only after the production header.
    job._validate_staging(f4_inputs, control, [1024], recipe_sha, skip)
    return {
        "control_path": str(f4_control.resolve()),
        "control_sha256": _sha(f4_control),
        "inputs_path": str(f4_inputs.resolve()),
        "recipe_sha256": recipe_sha,
    }


def admit_future(
    control_path: Path,
    *,
    speed_equivalence: Path,
    calibration: Path,
    out: Path,
    f4_control: Path,
    f4_inputs: Path,
    old_factory_root: Path | None = None,
) -> dict:
    """Bind a future stage to the new pin, T-EQ and frozen calibration receipt."""
    original = _read(control_path)
    gate = original.get("mac_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "HOLD"
        or gate.get("stage")
        not in {
            "d1024",
            "d4096",
        }
    ):
        raise ValueError("future control is not a sealed D1024/D4096 HOLD")
    if gate["stage"] == "d4096":
        raise ValueError(
            "D4096 remains HOLD pending its registered R6 funding/projection admission"
        )
    pin = _future_source_pin(original)
    legacy_pin = gate.get("w12_calibration_source_commit")
    runner_pin = gate.get("w12_calibration_runner_commit")
    if (
        not isinstance(legacy_pin, str)
        or len(legacy_pin) != 40
        or runner_pin != CALIBRATION_RUNNER_COMMIT
    ):
        raise ValueError("future control lacks frozen calibration pin")
    from . import speed_equiv
    from .mac_package import d256_allocation, future_seed_audit

    if old_factory_root is None:
        speed_equiv.verify_receipt(
            speed_equivalence, legacy_pin, pin, expected_bank_sha=F1_BANK_SHA256
        )
        equiv_status = "PASS"
    else:
        failed = _read(speed_equivalence)
        if (
            failed.get("schema") != "bl-labels-f0-speed-equiv/v1"
            or failed.get("status") != "FAIL"
            or failed.get("old", {}).get("commit") != legacy_pin
            or failed.get("new", {}).get("commit") != pin
            or failed.get("f1_bank_sha256") != F1_BANK_SHA256
            or not old_factory_root.is_dir()
        ):
            raise ValueError("T-EQ failure is not a sealed OLD fallback authority; HOLD")
        equiv_status = "FAIL_OLD_FALLBACK"
    calibration_status = _future_calibration(calibration, legacy_pin, runner_pin)
    audit_path = control_path.with_name(f"{gate['stage']}-seed-audit.json")
    prior = d256_allocation(f"f0-hindsight-labels-{legacy_pin}-mac-d256")
    if gate["stage"] == "d4096":
        prior_control = _read(control_path.with_name("d1024-control.json"))
        prior = [*prior, *prior_control["allocation"]]
    expected_audit = future_seed_audit(original["allocation"], prior_allocation_rows=prior)
    if _read(audit_path) != expected_audit:
        raise ValueError("future seed audit differs from sealed allocation; HOLD")
    readiness = _future_readiness(gate["stage"], f4_control, f4_inputs)
    value = copy.deepcopy(original)
    value["mac_gate"] = {
        **gate,
        "status": "APPROVED",
        "original_control_path": str(control_path.resolve()),
        "original_control_sha256": _verify(control_path),
        "package_root": str(control_path.parent.resolve()),
        "state_dir": str((control_path.parent / gate["state_dir_name"]).resolve()),
        "bindings": {
            "speed_equivalence": {
                "path": str(speed_equivalence.resolve()),
                "sha256": _verify(speed_equivalence),
                "status": equiv_status,
            },
            "calibration": {
                "path": str(calibration.resolve()),
                "sha256": _verify(calibration),
                "status": calibration_status,
            },
            "seed_audit": {"path": str(audit_path.resolve()), "sha256": _verify(audit_path)},
            "f4_readiness": readiness,
        },
    }
    if old_factory_root is not None:
        value["mac_gate"].update(
            collector_source_commit=legacy_pin,
            collector_root=str(old_factory_root.resolve()),
        )
    publish_json(out, value)
    return value


def _require_future_admission(control: dict) -> None:
    """Recheck every future-stage admission dependency before worker start."""
    gate = control["mac_gate"]
    original_path = Path(gate.get("original_control_path", ""))
    if not original_path.is_file() or _verify(original_path) != gate.get("original_control_sha256"):
        raise ValueError("future original control changed after admission")
    original = _read(original_path)
    pin = _future_source_pin(original)
    if control.get("source_commit") != pin:
        raise ValueError("future admitted source pin differs")
    bindings = gate.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("future admission bindings missing")
    speed = bindings.get("speed_equivalence", {})
    calibration = bindings.get("calibration", {})
    audit = bindings.get("seed_audit", {})
    readiness = bindings.get("f4_readiness", {})
    if (
        not isinstance(speed, dict)
        or not isinstance(calibration, dict)
        or not isinstance(audit, dict)
        or not isinstance(readiness, dict)
    ):
        raise ValueError("future admission bindings malformed")
    if _verify(Path(speed.get("path", ""))) != speed.get("sha256"):
        raise ValueError("future T-EQ receipt changed after admission")
    if _verify(Path(calibration.get("path", ""))) != calibration.get("sha256"):
        raise ValueError("future calibration receipt changed after admission")
    if _verify(Path(audit.get("path", ""))) != audit.get("sha256"):
        raise ValueError("future seed audit changed after admission")
    from . import speed_equiv

    legacy_pin = original["mac_gate"]["w12_calibration_source_commit"]
    runner_pin = original["mac_gate"]["w12_calibration_runner_commit"]
    if speed.get("status") == "PASS":
        speed_equiv.verify_receipt(
            Path(speed["path"]), legacy_pin, pin, expected_bank_sha=F1_BANK_SHA256
        )
        collector = {}
    elif speed.get("status") == "FAIL_OLD_FALLBACK":
        failed = _read(Path(speed["path"]))
        root = gate.get("collector_root")
        if (
            failed.get("schema") != "bl-labels-f0-speed-equiv/v1"
            or failed.get("status") != "FAIL"
            or failed.get("old", {}).get("commit") != legacy_pin
            or failed.get("new", {}).get("commit") != pin
            or failed.get("f1_bank_sha256") != F1_BANK_SHA256
            or gate.get("collector_source_commit") != legacy_pin
            or not isinstance(root, str)
            or not Path(root).is_dir()
        ):
            raise ValueError("future OLD fallback differs from its sealed T-EQ failure")
        collector = {
            "collector_source_commit": legacy_pin,
            "collector_root": str(Path(root).resolve()),
        }
    else:
        raise ValueError("future T-EQ status is malformed")
    status = _future_calibration(Path(calibration["path"]), legacy_pin, runner_pin)
    if status != calibration.get("status"):
        raise ValueError("future calibration status changed after admission")
    from .mac_package import d256_allocation, future_seed_audit

    prior = d256_allocation(f"f0-hindsight-labels-{legacy_pin}-mac-d256")
    if original["mac_gate"]["stage"] == "d4096":
        prior_control = _read(original_path.with_name("d1024-control.json"))
        prior = [*prior, *prior_control["allocation"]]
    if _read(Path(audit["path"])) != future_seed_audit(
        original["allocation"], prior_allocation_rows=prior
    ):
        raise ValueError("future seed audit differs after admission")
    readiness_now = _future_readiness(
        original["mac_gate"]["stage"],
        Path(readiness.get("control_path", "")),
        Path(readiness.get("inputs_path", "")),
    )
    if readiness != readiness_now:
        raise ValueError("D1024 registered predecessor readiness changed after admission")
    original_gate = original["mac_gate"]
    expected_gate = {
        **original_gate,
        "status": "APPROVED",
        "original_control_path": str(original_path.resolve()),
        "original_control_sha256": _verify(original_path),
        "package_root": str(original_path.parent.resolve()),
        "state_dir": str((original_path.parent / original_gate["state_dir_name"]).resolve()),
        "bindings": {
            "speed_equivalence": speed,
            "calibration": calibration,
            "seed_audit": audit,
            "f4_readiness": readiness,
        },
        **collector,
    }
    expected = copy.deepcopy(original)
    expected["mac_gate"] = expected_gate
    if control != expected:
        raise ValueError("future admitted control differs from sealed admission")


def ledger_units(attempts: list[Path]) -> list[dict]:
    """Inventory completed immutable triplets without inspecting outcomes."""
    units = []
    seen = set()
    for attempt in attempts:
        if list(attempt.glob("INVALID*.json")):
            raise ValueError("INVALID Mac attempt cannot contribute a ledger")
        if _read(attempt / "runtime-result.json").get("status") == "INVALID":
            raise ValueError("INVALID Mac attempt cannot contribute a ledger")
        for path in sorted((attempt / "lives").glob("*.json")):
            if not Path(str(path) + ".complete").is_file():
                continue
            name = path.stem
            if name in seen:
                raise ValueError("duplicate Mac completed unit across attempts")
            seen.add(name)
            digest = _verify(path)
            units.append(
                {
                    "unit_id": name,
                    "path": "lives/" + path.name,
                    "sha256": digest,
                    "size": path.stat().st_size,
                }
            )
    return units


def publish_ledger(
    pack: Path, attempts: list[Path], out: Path, *, control: dict | None = None
) -> dict:
    """Publish only verified wrappers/path/size/hash; do not calculate any outcome.

    D256 callers must first validate their complete durable attempt chain.
    The default remains the historical Canary-B namespace and allocation.
    """
    from .mac_driver import _verify_resume_metadata

    control = _read(pack / "control.json") if control is None else control
    namespace, allocation = control["namespace"], control["allocation"]
    value = empty_ledger(namespace, allocation)
    value["units"] = ledger_units(attempts)
    for attempt in attempts:
        for path in sorted((attempt / "lives").glob("*.json")):
            if Path(str(path) + ".complete").is_file():
                _materialize_triplet(path, out.parent / "lives" / path.name)
    # Validate before the ledger itself becomes visible as completed evidence.
    temporary = out.parent / (".validation-" + uuid.uuid4().hex + ".json")
    publish_json(temporary, value)
    try:
        names = load_ledger(temporary, namespace, allocation)
        _verify_resume_metadata(temporary, namespace, allocation, names)
    finally:
        for suffix in ("", ".sha256", ".complete"):
            Path(str(temporary) + suffix).unlink(missing_ok=True)
    value["attempt_exports"] = [str(p.resolve()) for p in attempts]
    publish_json(out, value)
    return value


def _attempts(pack: Path, attempts: list[Path], pin: str) -> float:
    if not 1 <= len(attempts) <= 2 or len(set(p.resolve() for p in attempts)) != len(attempts):
        raise ValueError("Canary-B permits one original plus one mechanical completion")
    total = 0.0
    state_dirs: set[Path] = set()
    receipts = []
    for attempt in attempts:
        receipt = _read(attempt / "runtime-result.json")
        elapsed = receipt.get("elapsed_seconds")
        if (
            receipt.get("namespace") != _read(pack / "control.json")["namespace"]
            or receipt.get("workers") != CANARY_WORKERS
            or receipt.get("status") == "INVALID"
            or receipt.get("platform_attestation", {}).get("code_sha") != pin
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("foreign, invalid or unmeasured Canary-B attempt")
        total += elapsed
        receipts.append(receipt)
        if not isinstance(receipt.get("state_dir"), str):
            raise ValueError("Canary-B lacks durable attempt accounting")
        state_dirs.add(Path(receipt["state_dir"]).resolve())
    if len(state_dirs) != 1:
        raise ValueError("Canary-B attempts must share one durable accounting state")
    state_dir = state_dirs.pop()
    starts = sorted(state_dir.glob("attempt-*.start.json"))
    terminal_names = set()
    outputs = set()
    state_total = 0.0
    for start_path in starts:
        start = _read(start_path)
        terminal_path = start_path.with_name(
            start_path.name.replace(".start.json", ".terminal.json")
        )
        terminal = _read(terminal_path)
        if (
            start.get("namespace") != _read(pack / "control.json")["namespace"]
            or start.get("workers") != CANARY_WORKERS
            or start.get("deadline_seconds") != CANARY_WALL_SECONDS
            or start.get("platform_attestation", {}).get("code_sha") != pin
        ):
            raise ValueError("foreign Canary-B durable start identity")
        output = Path(start["output"]).resolve()
        matching = [r for p, r in zip(attempts, receipts) if p.resolve() == output]
        if len(matching) != 1 or any(
            terminal.get(k) != matching[0].get(k)
            for k in (
                "namespace",
                "workers",
                "elapsed_seconds",
                "attempts_ledger",
            )
        ):
            raise ValueError("omitted or mismatched Canary-B attempt accounting")
        elapsed = terminal.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("invalid Canary-B terminal wall charge")
        state_total += elapsed
        outputs.add(output)
        terminal_names.add(terminal_path.name)
    if outputs != {p.resolve() for p in attempts} or len(starts) != len(attempts):
        raise ValueError("omitted or foreign Canary-B attempts")
    if not any(set(r.get("attempts_ledger", [])) == terminal_names for r in receipts):
        raise ValueError("Canary-B does not bind its complete attempt chain")
    if abs(total - state_total) > 1e-6:
        raise ValueError("Canary-B wall charges do not reconcile")
    if total > CANARY_WALL_SECONDS:
        raise ValueError("Canary-B original three-hour wall cap exceeded")
    return CANARY_WORKERS * total / 3600


def look_canary(pack: Path, ledger: Path, attempts: list[Path], out: Path) -> dict:
    pin = _source_pin(pack)
    control = _read(pack / "control.json")
    value = _read(ledger)
    if value.get("attempt_exports") != [str(p.resolve()) for p in attempts]:
        raise ValueError("publish the ledger for these exact attempts before the F0 look")
    names = load_ledger(ledger, control["namespace"], control["allocation"])
    from .mac_driver import _verify_resume_metadata

    _verify_resume_metadata(ledger, control["namespace"], control["allocation"], names)
    if completed_session_parents(ledger, names) != {
        r["allocation_id"] for r in control["allocation"]
    }:
        raise ValueError("HOLD: Canary-B must be complete 36/36 before its single look")
    hours = _attempts(pack, attempts, pin)
    claim_path = pack / "f0-look-claim.json"
    global_claim = GLOBAL_LOOK_ROOT / (
        hashlib.sha256(control["namespace"].encode()).hexdigest() + ".json"
    )
    if claim_path.exists() or global_claim.exists():
        raise ValueError("F0 registered look already claimed; no re-look")
    claim = {
        "schema": "bl-labels-mac-f0-look-claim/v1",
        "nonce": uuid.uuid4().hex,
        "ledger": str(ledger.resolve()),
        "ledger_sha256": _verify(ledger),
        "out": str(out.resolve()),
        "source_commit": pin,
    }
    publish_json(global_claim, claim)
    publish_json(claim_path, claim)
    aggregate_path = out.parent / "registered-aggregate" / "verified-ledger.json"
    # The published ledger materializes exactly verified completed triplets.
    # Keep interrupted raw writes immutable; they cannot enter the registered
    # result set or consume the look through a second scan of partial exports.
    aggregate = aggregate_attempts(
        [ledger.parent], aggregate_path, control["namespace"], control["allocation"]
    )
    analysis = analyze_aggregate(aggregate, hours)
    result = {
        "schema": "bl-labels-mac-f0-look/v1",
        "source_commit": pin,
        "namespace": control["namespace"],
        "claim_sha256": _verify(claim_path),
        "ledger_sha256": _verify(ledger),
        "aggregate_sha256": _verify(aggregate_path),
        "worker_hours": hours,
        "analysis": analysis,
        "status": "PASS" if analysis["verdict"] == "COLLECTED" else analysis["verdict"],
    }
    publish_json(out, result)
    return result


def _require_f0(pack: Path, path: Path, pin: str) -> dict:
    value = _read(path)
    claim = _read(pack / "f0-look-claim.json")
    namespace = _read(pack / "control.json")["namespace"]
    global_claim = GLOBAL_LOOK_ROOT / (hashlib.sha256(namespace.encode()).hexdigest() + ".json")
    if _verify(global_claim) != _verify(pack / "f0-look-claim.json"):
        raise ValueError("F0 canonical global look claim differs")
    if (
        value.get("schema") != "bl-labels-mac-f0-look/v1"
        or value.get("source_commit") != pin
        or value.get("namespace") != _read(pack / "control.json")["namespace"]
        or value.get("status") != "PASS"
        or value.get("analysis", {}).get("verdict") != "COLLECTED"
        or value.get("claim_sha256") != _sha(pack / "f0-look-claim.json")
        or claim.get("out") != str(path.resolve())
    ):
        raise ValueError("D256 HOLD: registered complete Mac F0 look has not passed")
    if value.get("ledger_sha256") != _verify(Path(claim["ledger"])):
        raise ValueError("F0 published ledger changed after its look")
    if value.get("aggregate_sha256") != _verify(
        path.parent / "registered-aggregate/verified-ledger.json"
    ):
        raise ValueError("F0 registered aggregate changed after its look")
    return value


def admit_d256(pack: Path, f1_output: Path, f0_look: Path, approval: Path, out: Path) -> dict:
    pin = _source_pin(pack)
    _f1(pack, f1_output, pin)
    _require_f0(pack, f0_look, pin)
    _approval(approval)
    audit = _seed_audit(pack)
    value = _approved_control(
        pack,
        "d256",
        {
            "f1": {
                "path": str((f1_output / "runtime-result.json").resolve()),
                "sha256": _verify(f1_output / "runtime-result.json"),
            },
            "f0": {"path": str(f0_look.resolve()), "sha256": _verify(f0_look)},
            "f4": {"path": str(approval.resolve()), "sha256": _sha(approval)},
            "seed_audit": {"path": str(audit.resolve()), "sha256": _verify(audit)},
        },
    )
    publish_json(out, value)
    return value


def _seed_audit(pack: Path) -> Path:
    path = pack / "d256-seed-audit.json"
    audit = _read(path)
    allocation = _read(pack / "d256-control.json")["allocation"]
    if (
        audit.get("schema") != "bl-labels-f0-d256-seed-audit/v1"
        or audit.get("seeds") != [r["seed"] for r in allocation]
        or audit.get("outcome_blind") is not True
        or not audit.get("hits_by_root")
        or any(audit["hits_by_root"].values())
    ):
        raise ValueError("D256 HOLD: frozen allocation lacks a clean seed audit")
    return path


def replay_canary_a(pack: Path, out: Path) -> dict:
    """Audit the frozen 64 hash-selected x86 roots without opening label outcomes."""
    from .replay_audit import audit

    manifest = _read(pack / "canary-b-manifest.json")
    bank_path = Path(manifest["f1_bank_path"])
    if bank_digest(bank_path) != F1_BANK_SHA256:
        raise ValueError("foreign frozen Canary-A audit bank")
    bank = load_bank(bank_path)
    sources = {source["name"]: source for source in bank["source_sessions"]}
    records = [
        {**root, "tape": sources[root["source_session"]]["tape"][: root["tape_end"]]}
        for root in bank["roots"]
    ]
    checkpoint_dir = ROOT.parent / "snake-dqn/saved_snakes"
    try:
        result = audit(
            records,
            b8=checkpoint_dir / "b8_s3_t7_20261016.pth",
            frp3=checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth",
        )
    except ValueError as error:
        result = {"schema": "bl-labels-f0-mac-replay/v1", "status": "INVALID", "reason": str(error)}
    result["mixing_status"] = (
        "X86_LABELS_MIXABLE" if result["status"] == "PASS" else "X86_LABELS_NOT_MIXABLE"
    )
    result["scope"] = "64 recorded-action roots; not a gate on Mac-only production"
    publish_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("admit-canary", "ledger", "look-canary", "admit-d256", "replay-canary-a"):
        p = sub.add_parser(command)
        p.add_argument("--pack", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if command in {"admit-canary", "admit-d256"}:
            p.add_argument("--f1-output", type=Path, required=True)
        if command in {"ledger", "look-canary"}:
            p.add_argument("--attempt", type=Path, action="append", required=True)
        if command == "look-canary":
            p.add_argument("--ledger", type=Path, required=True)
        if command == "admit-d256":
            p.add_argument("--f0-look", type=Path, required=True)
            p.add_argument("--f4-approval", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "admit-canary":
        result = admit_canary(a.pack, a.f1_output, a.out)
    elif a.command == "ledger":
        result = publish_ledger(a.pack, a.attempt, a.out)
    elif a.command == "look-canary":
        result = look_canary(a.pack, a.ledger, a.attempt, a.out)
    elif a.command == "replay-canary-a":
        result = replay_canary_a(a.pack, a.out)
    else:
        result = admit_d256(a.pack, a.f1_output, a.f0_look, a.f4_approval, a.out)
    print(
        json.dumps(
            {"command": a.command, "out": str(a.out), "status": result.get("status", "PREPARED")}
        )
    )


if __name__ == "__main__":
    main()
