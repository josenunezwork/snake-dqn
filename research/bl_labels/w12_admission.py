"""Create and verify the one ruled W12 throughput admission record.

Calibration-01 remains a sealed ``FAIL`` under its original per-worker rule.
This module never edits it: it separately binds its bytes and the independent
throughput ruling, then redoes the outcome-blind wall-throughput calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .exports import _verify, publish_json
from .w12_calibration import F1_BANK_SHA256
from .w12_calibration import SCHEMA as CALIBRATION_SCHEMA
from .w12_calibration import wall_ratio

SCHEMA = "bl-labels-f0-w12-admission/v1"
CALIBRATION_RECEIPT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/"
    "w12-calibration-01/receipt.json"
)
ADMISSION_RECEIPT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/" "w12-admission-01/receipt.json"
)
JUDGMENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_W12_THROUGHPUT_20261023.md"
)
CALIBRATION_SHA256 = "082304bc3e94ce48f82313ed285fc7388c099ef4b8d5c1bcc47545cb6fb3cdbc"
JUDGMENT_SHA256 = "03a7afd1a1ab48dfbb4a2a4fd760ea00d2fe5c8bf6132bfc3660f3077b8815ef"
F0_COMMIT = "61900af28932e47e22e332840fc4406559adb062"
MIN_THROUGHPUT_RATIO = 1.20


def sha(path: Path) -> str:
    """Return the exact SHA-256 of an immutable input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"W12 admission {name} is not a finite number")
    return float(value)


def _calibration(path: Path, *, expected_code_sha: str) -> tuple[dict[str, Any], float, float]:
    """Validate calibration-01 itself and recompute its wall throughput."""
    if _verify(path) != CALIBRATION_SHA256:
        raise ValueError("W12 admission calibration receipt hash differs from calibration-01")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("W12 admission calibration receipt is not an object")
    if (
        value.get("schema") != CALIBRATION_SCHEMA
        or value.get("status") != "FAIL"
        or value.get("code_sha") != expected_code_sha
        or value.get("platform") != "darwin-arm64"
        or value.get("bank_sha256") != F1_BANK_SHA256
        or value.get("thermal_not_ok") != 0
        or value.get("slowdown_trips") != 0
        or value.get("outcome_exports") is not False
    ):
        raise ValueError("W12 admission calibration provenance or safety fields differ")
    samples = path.parent / "host-samples.jsonl"
    if (
        not samples.is_file()
        or value.get("host_samples_sha256") != sha(samples)
        or value.get("host_samples_bytes") != samples.stat().st_size
    ):
        raise ValueError("W12 admission calibration host samples differ")
    duration = _number(value.get("duration_seconds"), "duration")
    if duration < 1800:
        raise ValueError("W12 admission calibration duration is below 1800 seconds")
    cells = value.get("cells")
    if not isinstance(cells, list):
        raise ValueError("W12 admission calibration cells are absent")
    ratio = wall_ratio(cells)
    stored_ratio = _number(value.get("w12_w6_wall_ratio"), "stored wall ratio")
    if not math.isclose(ratio, stored_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("W12 admission calibration cells do not match stored wall ratio")
    throughput = 2.0 / ratio
    if not throughput >= MIN_THROUGHPUT_RATIO:
        raise ValueError("W12 admission wall throughput is below the ruled 1.20x criterion")
    return value, ratio, throughput


def build(*, calibration: Path | None = None, judgment: Path | None = None) -> dict[str, Any]:
    """Return the immutable admission payload, without publishing it."""
    calibration = calibration or CALIBRATION_RECEIPT
    judgment = judgment or JUDGMENT
    if calibration.resolve() != CALIBRATION_RECEIPT.resolve():
        raise ValueError("W12 admission must bind the sealed calibration-01 receipt")
    if judgment.resolve() != JUDGMENT.resolve() or sha(judgment) != JUDGMENT_SHA256:
        raise ValueError("W12 admission judgment bytes differ from the independent ruling")
    value, ratio, throughput = _calibration(calibration, expected_code_sha=F0_COMMIT)
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "calibration_receipt": str(calibration.resolve()),
        "calibration_receipt_sha256": _verify(calibration),
        "judgment": str(judgment.resolve()),
        "judgment_sha256": sha(judgment),
        "criterion": {
            "formula": "throughput = 2 / w12_w6_wall_ratio",
            "minimum": MIN_THROUGHPUT_RATIO,
            "equivalent_max_w12_w6_wall_ratio": 2.0 / MIN_THROUGHPUT_RATIO,
        },
        "computed": {
            "duration_seconds": value["duration_seconds"],
            "w12_w6_wall_ratio": ratio,
            "w12_units_per_wall_second_ratio": throughput,
        },
        "provenance": {
            "code_sha": value["code_sha"],
            "platform": value["platform"],
            "bank_sha256": value["bank_sha256"],
            "thermal_not_ok": value["thermal_not_ok"],
            "slowdown_trips": value["slowdown_trips"],
        },
    }


def create(output: Path) -> dict[str, Any]:
    """Publish the create-only admission triplet after every input is rechecked."""
    if output.resolve() != ADMISSION_RECEIPT.resolve():
        raise ValueError("W12 admission must publish only w12-admission-01/receipt.json")
    if output.exists() or any(
        Path(str(output) + suffix).exists() for suffix in (".sha256", ".complete")
    ):
        raise ValueError("W12 admission record already exists; create-only publication refused")
    value = build()
    publish_json(output, value)
    return value


def verify(path: Path, *, expected_code_sha: str | None = None) -> str:
    """Recompute and compare every admission field; return its sealed digest."""
    if path.resolve() != ADMISSION_RECEIPT.resolve():
        raise ValueError("W12 admission must be the canonical w12-admission-01 record")
    record_sha = _verify(path)
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("status") != "PASS"
    ):
        raise ValueError("W12 admission record schema or status differs")
    code_sha = expected_code_sha or F0_COMMIT
    if code_sha != F0_COMMIT:
        raise ValueError("W12 admission is bound only to calibration-01 F0_COMMIT")
    expected = build()
    if value != expected:
        raise ValueError("W12 admission record differs from recomputed ruled evidence")
    return record_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        value = build()
        print(json.dumps({"status": "CHECKED", "throughput": value["computed"]}, sort_keys=True))
        return
    value = create(args.output)
    print(json.dumps({"status": value["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
