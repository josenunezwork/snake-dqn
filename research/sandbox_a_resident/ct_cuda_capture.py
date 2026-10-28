"""Mac-only admission of the completed, excluded-fixture A100 capture.

This is deliberately separate from verify_mechanical_stop: successful canaries
and their fixture summaries cannot authorize a science continuation. The shared
closed diagnostic parsers use an explicit completion policy only in this path.
No torch import or numerical checkpoint decoding is needed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.sandbox_a_resident import ct_resume as admission

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "tests/fixtures/ct-cuda-smoke-08"
PACKET = ROOT / "research/sandbox_a_resident/packet-ct-cuda-smoke-1028-08"
# Immutable first capture predates heartbeat instrumentation. Every later source
# must carry the new cadence proof regardless of its local directory name.
LEGACY_CAPTURE_PIN = "d2302487ebecb1791bec1fb22cd9ab98209ad3d7"


def _read(path: Path) -> Any:
    return admission._load_json(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise admission.CTResumeRefused(message)


def _finite_under(value: Any, bound: float) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value < bound


@admission._evidence_session
def validate_capture(
    output: Path, supervisor_log: Path, *, packet: Path = PACKET, binding: Path = BINDING
) -> dict[str, Any]:
    """Verify every retained byte and diagnostic surface of this fixture-only run."""
    admission._no_symlink_path(output)
    seal = _read(packet / "SEAL.json")
    manifest = _read(output / "manifest.json")
    receipt = _read(output / "receipt.json")
    terminal = _read(output / "terminal.json")
    pin = seal["source_commit"]
    original = _read(binding / "dispatch-original.json")
    live = _read(binding / "dispatch-live.json")
    accounting = _read(binding / "accounting.json")
    expected_live = copy.deepcopy(original)
    expected_live["budget"]["other_assignments"] = accounting["other_assignments"]
    expected_runtime = copy.deepcopy(live)
    expected_runtime["lease_seconds"] = 180.0
    _require(
        original == _read(packet / "dispatch-manifest.json")
        and admission.digest(binding / "dispatch-original.json") == seal["manifest_sha256"]
        and accounting["source_manifest_sha256"] == seal["manifest_sha256"]
        and admission.digest(binding / "dispatch-live.json") == accounting["manifest_sha256"]
        and live == expected_live
        and manifest == expected_runtime,
        "capture dispatch accounting/normalization chain mismatch",
    )
    _require(
        admission.digest(packet / "dispatch-manifest.json") == seal["manifest_sha256"]
        and manifest == expected_runtime
        and receipt["manifest_sha256"] == admission.digest(output / "manifest.json")
        and receipt["job_id"] == manifest["job_id"]
        and receipt["commit"] == manifest["commit"] == pin
        and admission._is_hex(receipt["generation"], 32),
        "capture manifest/source/identity binding mismatch",
    )
    _require(
        receipt["status"] == "PASS"
        and receipt["stop_verified"] is True
        and all(receipt.get(k) is None for k in ("error", "cleanup_error", "failure_class"))
        and terminal == receipt["terminal"]
        and terminal["status"] == "succeeded"
        and terminal["returncode"] == 0
        and terminal["reason"] is None,
        "capture lacks clean completion and verified stop",
    )
    proof = receipt["canary_evidence"]
    _require(
        proof["schema"] == "colab-family-canary/v1"
        and proof["commit"] == pin
        and proof["family"] == manifest["script"]
        and proof["hardware"] == "A100"
        and len(proof["inputs"]) == len(manifest["inputs"])
        and {(row["dest"], row["sha256"]) for row in proof["inputs"]}
        == {(row["dest"], row["sha256"]) for row in manifest["inputs"]}
        and all(row["transport"] == "upload" and row["size"] > 0 for row in proof["inputs"])
        and all(
            admission.digest(ROOT / name) == expected
            for name, expected in proof["transport_identity"].items()
        )
        and receipt["supervisor_sha256"]
        == admission.digest(ROOT / "research/colab_cli/supervisor.py"),
        "capture input/qualified-transport proof mismatch",
    )
    job, generation = receipt["job_id"], receipt["generation"]
    cache = admission._EVIDENCE_CACHE.get()
    assert cache is not None
    plan = _read(output / "staging/source-plan.json")
    prepared = _read(binding / "prepared-source-plan.json")
    _require(
        admission.digest(binding / "prepared-source-plan.json") == seal["source_plan_sha256"]
        and prepared["fallback_sha256"] == seal["source_archive"]["sha256"]
        and {k: v for k, v in plan.items() if k != "fallback_sha256"}
        == {k: v for k, v in prepared.items() if k != "fallback_sha256"},
        "capture source members differ from frozen canary plan",
    )
    cache["source_archive_binding"] = {
        "source_commit": pin,
        "source_tree": seal["source_tree"],
        "source_files": {k: v["sha256"] for k, v in plan["files"].items()},
    }
    admission._verify_staged_source_archive(output / "staging/source.tar.gz", output)
    base = output / "exports/ct-cuda-canary"
    capture = _read(base / "capture.json")
    _require(
        capture["schema"] == "ct-cuda-fixture-capture/v1"
        and capture["science_data"] is False
        and capture["real_cuda"] is True
        and capture["device"] == "cuda"
        and capture["source_commit"] == pin
        and capture["torch"] == "2.9.1+cu126"
        and capture["cuda"] == "12.6"
        and capture["gpu"] == "NVIDIA A100-SXM4-40GB",
        "capture is not the frozen real-CUDA excluded fixture",
    )
    liveness = None
    if pin != LEGACY_CAPTURE_PIN:
        from research.sandbox_a_resident.ct_capture_liveness import measure_liveness

        liveness = measure_liveness(output, supervisor_log)
    probes = capture["probes"]
    _require([p["name"] for p in probes] == ["complete", "signal"], "missing fixture probes")
    for p in probes:
        runtime = p["runtime"]
        _require(
            runtime == _read(base / p["name"] / "fixture-runtime.json")
            and runtime["fixture_only"] is True
            and runtime["device"] == "cuda:0"
            and runtime["gpu"] == capture["gpu"]
            and 0
            < runtime["peak_allocated_bytes"]
            <= runtime["peak_reserved_bytes"]
            < runtime["total_vram_bytes"],
            "fixture CUDA runtime evidence mismatch",
        )
    signal = probes[1]
    marker = _read(base / "signal/sigusr1-marker.json")
    admission._validate_signal_marker(marker)
    _require(
        probes[0]["returncode"] == 0
        and signal["returncode"] == 138
        and signal["marker"] == marker
        and marker["latest_durable_stage"] == 1
        and _finite_under(signal["signal_to_exit_seconds"], 15)
        and 255 * 1024**2 <= signal["checkpoint_bytes"] <= 256 * 1024**2
        and signal["checkpoint_bytes"] == (base / "signal/stage-0001.pt").stat().st_size,
        "CUDA checkpoint/independent SIGUSR1 timing fails 15-second SLA",
    )
    observations = terminal["canary_observations"]
    _require(
        len(observations) == 2
        and all(
            row["device"] == "cuda:0"
            and row["operation"] == "Tensor.argmax"
            and row["cuda_decisions"] > 0
            and row["package"] == "research/sandbox_a_resident"
            and row["root"] == f"/content/snake-cli/{generation}"
            for row in observations
        ),
        "missing real CUDA operation observations",
    )
    files = admission._retained_files(output)
    # Establish exact byte-truncation witnesses before scanning any saved tail.
    for path in files:
        if path.name in {
            "progress-history.log",
            "complete-entrypoint.log",
            "signal-entrypoint.log",
        }:
            text = admission._scan_diagnostic_bytes(path)
            admission._scan_job_tail(
                text, label=str(path), generation=generation, fixture_completion=True
            )
            cache.setdefault("progress_history_lines", set()).update(text.splitlines())
    for path in files:
        if path == output / "manifest.json":
            continue  # Exact original -> live -> normalized binding verified above.
        if admission._publication_sidecar(path):
            continue
        if path == output / "staging/schema-validation.json":
            _require(
                _read(path)
                == {
                    "manifest_sha256": hashlib.sha256(
                        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "required_exports": manifest["required_exports"],
                    "schema": "colab-cli-manifest/v1",
                    "status": "PASS",
                },
                "capture schema-validation binding mismatch",
            )
            continue
        if path.is_relative_to(output / "staging"):
            admission._scan_retained_text(path, output_root=output)
            continue
        if path.is_relative_to(output / "exports"):
            admission._descriptor(path, runtime_path=str(path.relative_to(output / "exports")))
        if admission._verified_stage_payload(path):
            continue
        if path.name.startswith("faulthandler-stacks.log"):
            _require(path.stat().st_size == 0, "non-empty capture faulthandler stack")
        if path.suffix == ".json":
            admission._scan_json_faults(
                _read(path),
                label=str(path),
                expected_job_id=job,
                expected_generation=generation,
                supervisor_transport=True,
                fixture_completion=True,
            )
        else:
            admission._scan_diagnostic_bytes(path)
    exported = {
        str(p.relative_to(output / "exports"))
        for p in files
        if p.is_relative_to(output / "exports") and p.suffix not in {".sha256", ".complete"}
    }
    rows = receipt["exports"]
    _require(
        len(rows) == len(exported) and {r["path"] for r in rows} == exported,
        "capture export inventory mismatch",
    )
    _require(
        all(
            len(inventory) == len(rows)
            and len({row["path"] for row in inventory}) == len(rows)
            and sorted(inventory, key=lambda row: row["path"])
            == sorted(rows, key=lambda row: row["path"])
            for inventory in (terminal["exports"], terminal["export_scanner"]["inventory"])
        ),
        "capture terminal/export scanner disagreement",
    )
    for row in rows:
        path = output / "exports" / row["path"]
        _require(
            path.stat().st_size == row["size"] and admission.digest(path) == row["sha256"],
            "capture export size/SHA-256 mismatch",
        )
    admission._scan_supervisor_log(
        supervisor_log, job_id=job, generation=generation, fixture_completion=True
    )
    return {
        "status": "PASS_CUDA_CAPTURE_ADMISSION",
        "science_data": False,
        "exports": len(rows),
        "files": len(files),
        "checkpoint_bytes": signal["checkpoint_bytes"],
        "marker_seconds": marker["shutdown_elapsed_sec"],
        "signal_to_exit_seconds": signal["signal_to_exit_seconds"],
        "stop_verified": True,
        "source_commit": pin,
        "liveness": liveness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervisor-log", type=Path, required=True)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--binding", type=Path, default=BINDING)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_capture(
                args.output, args.supervisor_log, packet=args.packet, binding=args.binding
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
