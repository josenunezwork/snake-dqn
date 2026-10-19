"""Local S1 checkpoint capacity accounting and receipt-bound export pricing.

The raw-capacity formula is exact for the registered ndarray allocations; it is
not a claim that a small fixture measured a full production state.pt.  NumPy
arrays use torch.save's default protocol-2 Latin-1 pickle and can occupy up to
twice their raw bytes.  Actor/world objects and serializer framing are additional.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = (2_000_000, 4_000_000, 6_000_000, 8_000_000)
CHUNK_BYTES = 1536 * 1024**2
SOURCES = (
    "research/s1_mature_20261018/manifests.py",
    "research/s1_mature_20261018/train.py",
    "research/s1_mature_20261018/checkpoint_sizing.py",
    "research/redesign_m3c_20261008/train_m3c.py",
    "research/redesign_m3c_20261008/async_checkpoint.py",
    "research/redesign_m3c_20261008/actor_snapshot.py",
    "research/redesign_m3c_20261008/execution_snapshot.py",
)
CLI_SOURCES = (
    "research/colab_cli/source.py",
    "research/colab_cli/supervisor.py",
    "research/colab_cli/transfers.py",
    "research/colab_cli/runtime.py",
    "research/colab_cli/bootstrap.sh",
    "research/colab_cli/acceptance.py",
    "research/colab_cli/acceptance_probe.py",
)


def binding(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def capacity(arm: str) -> dict[str, Any]:
    """Price the actual raw ring/replay/label allocations, at any replay fill."""
    from research.redesign_m3c_20261008.train_m3c import ring_capacity, ring_fields
    from research.s1_mature_20261018.manifests import training_command

    command = training_command(arm, 0)
    actors = int(command[command.index("--actors") + 1])
    worlds = int(command[command.index("--worlds") + 1])
    seats = int(command[command.index("--learner-seats") + 1])
    rows, margin = ring_capacity(500_000, actors, worlds * seats, 2)
    fields = [
        {
            "name": name,
            "row_shape": list(shape),
            "dtype": np.dtype(dtype).str,
            "row_bytes": int(np.prod(shape)) * np.dtype(dtype).itemsize,
        }
        for name, shape, dtype in ring_fields(60)
    ]
    row_bytes = sum(item["row_bytes"] for item in fields)
    rings = actors * rows * row_bytes
    replay = 500_000 * (8 + 8 + 4 + 4 + 8)  # replay_state's five saved arrays
    labels = 300_000 * (8 + 8)
    raw = rings + replay + labels
    return {
        "arm": arm,
        "actors": actors,
        "worlds_per_actor": worlds,
        "learner_seats": seats,
        "replay_capacity": 500_000,
        "label_capacity": 300_000,
        "ring_rows_per_actor": rows,
        "ring_margin_rows": margin,
        "ring_fields": fields,
        "ring_row_bytes": row_bytes,
        "ring_array_bytes": rings,
        "replay_array_bytes": replay,
        "label_array_bytes": labels,
        "demonstration_buffer_bytes": 0,
        "demonstration_reason": "bv8 sets demos=None; anchors are on-policy labels",
        "fixed_capacity_array_bytes": raw,
        "fixed_capacity_array_gib": raw / 1024**3,
        "protocol2_array_payload_upper_bound_bytes": 2 * raw,
        "snapshot_targets": list(SNAPSHOTS),
        "fill_independent": True,
        "excluded": "actor simulator/runtime/n-step state, networks, optimizer, RNG, "
        "Python objects and pickle/zip framing",
    }


def _raw_bytes(value: Any) -> int:
    import torch

    if isinstance(value, np.ndarray):
        return value.nbytes
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_raw_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_raw_bytes(item) for item in value)
    return 0


def _load_fixture(archive: Path) -> tuple[dict[str, Any], int]:
    import torch

    from research.redesign_m3c_20261008.train_m3c import ActionWindow

    # Trainer entrypoint pickled this class under __main__; do not patch source.
    main = sys.modules["__main__"]
    old = getattr(main, "ActionWindow", None)
    main.ActionWindow = ActionWindow
    try:
        with tarfile.open(archive, "r:gz") as stream:
            member = next(item for item in stream.getmembers() if item.name.endswith("/state.pt"))
            state = torch.load(stream.extractfile(member), map_location="cpu", weights_only=False)
    finally:
        if old is None:
            del main.ActionWindow
        else:
            main.ActionWindow = old
    return state, member.size


def observe_fixture(archive: Path) -> dict[str, Any]:
    """Inspect our trusted local smoke checkpoint without changing its bytes."""
    state, size = _load_fixture(archive)
    actor_values = list(state["extra"]["actors"].values())
    return {
        "archive": binding(archive),
        "archive_bytes": archive.stat().st_size,
        "state_pt_bytes": size,
        "learner_decisions": state["counters"]["transitions"],
        "raw_array_tensor_bytes": _raw_bytes(state),
        "top_level_raw_array_tensor_bytes": {
            key: _raw_bytes(value) for key, value in state.items()
        },
        "actor_raw_array_tensor_bytes": [_raw_bytes(item) for item in actor_values],
        "actor_network_bytes": [_raw_bytes(item["net"]) for item in actor_values],
        "fixed_noncapacity_raw_bytes": sum(
            _raw_bytes(value)
            for key, value in state.items()
            if key not in ("rings", "replay", "labels", "extra")
        )
        + sum(_raw_bytes(value) for key, value in state["extra"].items() if key != "actors"),
        "recipe": state["extra"]["recipe"],
        "demonstrations_absent": state["extra"]["demos"] is None,
        "limitation": "Observed tiny 2-actor, 8-world fixture; "
        "not production-size or L4 qualification",
    }


class _CountingSink:
    """Torch file-like sink: execute the real serializer without retaining bytes."""

    def __init__(self) -> None:
        self.bytes = 0

    def write(self, value: bytes) -> int:
        self.bytes += len(value)
        return len(value)

    def flush(self) -> None:
        pass


def measure_capacity_serialization(archive: Path, arm: str) -> dict[str, Any]:
    """Serialize every production-sized allocation with real fixture values.

    Arrays are measured individually to keep peak RAM bounded. Summing their
    standalone archive sizes includes extra per-array zip/pickle framing; actor
    object and model state remain accounted for separately. No compression is
    used, exactly as in AsyncCheckpointer's state.pt producer.
    """
    import torch

    state, _ = _load_fixture(archive)
    shape = capacity(arm)
    observed = []
    started = time.monotonic()
    for kind, sample, rows, repeats in (
        ("rings", state["rings"][0]["arrays"], shape["ring_rows_per_actor"], shape["actors"]),
        (
            "replay",
            {k: v for k, v in state["replay"].items() if isinstance(v, np.ndarray)},
            500_000,
            1,
        ),
        (
            "labels",
            {k: v for k, v in state["labels"].items() if isinstance(v, np.ndarray)},
            300_000,
            1,
        ),
    ):
        for name, source in sample.items():
            array = np.resize(source, (rows, *source.shape[1:]))
            sink = _CountingSink()
            torch.save(array, sink)
            observed.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "raw_bytes_per_array": array.nbytes,
                    "serialized_bytes_per_array": sink.bytes,
                    "copies": repeats,
                }
            )
            del array
    return {
        "method": "real torch.save default protocol2, production shapes, "
        "np.resize of observed values",
        "array_inventory": observed,
        "raw_capacity_array_bytes": sum(
            item["raw_bytes_per_array"] * item["copies"] for item in observed
        ),
        "serialized_capacity_array_bytes": sum(
            item["serialized_bytes_per_array"] * item["copies"] for item in observed
        ),
        "elapsed_mac_seconds": time.monotonic() - started,
        "production_trajectory": False,
        "standalone_array_archive_framing_included": True,
        "l4_preparation_time_qualified": False,
    }


def projected_checkpoint(arm: str, observed: dict[str, Any]) -> dict[str, Any]:
    """Combine exact production capacity with disclosed actor/metadata allowances.

    This is a conservative planning calculation, not a measured production file.
    It keeps the unknown full-size L4 preparation qualification explicit.
    """
    result = capacity(arm)
    source_worlds = int(observed["recipe"]["worlds"])
    actor_raw = max(observed["actor_raw_array_tensor_bytes"])
    actor_net = max(observed["actor_network_bytes"])
    actor_projection = actor_net + math.ceil((actor_raw - actor_net) * 128 / source_worlds)
    # Deferred frame packets contain transition/label/statistic arrays, not
    # raster observations. Reserve 1 KiB per driven row for all three packets.
    deferred = result["actors"] * 3 * 128 * result["learner_seats"] * 1024
    raw = (
        result["fixed_capacity_array_bytes"]
        + observed["fixed_noncapacity_raw_bytes"]
        + result["actors"] * actor_projection
        + deferred
    )
    # Explicit planning allowance for object/RNG/pickle framing beyond ndarray
    # payload. It must be validated against a full-size checkpoint before PASS.
    framing = (result["actors"] + 1) * 64 * 1024**2
    planned = 2 * raw + framing
    archive = math.ceil(planned * 1.01) + 1024**2
    return {
        **result,
        "actor_raw_array_tensor_bytes_projection_per_actor": actor_projection,
        "fixed_noncapacity_raw_bytes_observed": observed["fixed_noncapacity_raw_bytes"],
        "deferred_packet_raw_bytes_planning_allowance": deferred,
        "all_state_raw_bytes_projection": raw,
        "python_and_serializer_framing_bytes_planning_allowance": framing,
        "serialized_checkpoint_bytes_planning_upper_bound": planned,
        "serialized_checkpoint_gib_planning_upper_bound": planned / 1024**3,
        "snapshot_archive_bytes_planning_upper_bound": archive,
        "four_snapshot_export_bytes_planning_upper_bound": 4 * archive,
        "assumptions": [
            "Actor non-network arrays scale linearly from observed worlds to 128 worlds",
            "Default protocol-2 NumPy payload expansion is bounded by 2 bytes per raw byte",
            "64 MiB Python/RNG/pickle framing allowance per actor plus learner; provisional",
            "At most three deferred packets per actor with 1 KiB per driven row allowance",
            "No compression credit; gzip/tar/chunk/index framing gets 1 percent plus 1 MiB",
        ],
        "production_size_measured": False,
        "planning_allowances_require_full_size_validation": True,
        "complete_upper_bound": None,
    }


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _bound_receipt(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{name} requires a path and sha256")
    path = Path(value["path"])
    if not path.is_file() or binding(path)["sha256"] != value.get("sha256"):
        raise ValueError(f"{name} receipt absent or changed")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"{name} receipt must be a JSON object")
    return result


def transport_bound(value: dict[str, Any], *, timing_sha256: str) -> dict[str, Any]:
    """Validate measured full-size preparation and CLI receipt transfer throughput.

    Timeout is an outer cap, never evidence of serialization or transfer speed.
    Every registered 2/4/6/8M snapshot and the final weights archive is charged.
    """
    if value.get("schema") != "s1-full-replay-transport/v1" or value.get("status") != "PASS":
        raise ValueError("full-replay transport qualification is absent")
    if value.get("timing_receipt_sha256") != timing_sha256:
        raise ValueError("full-replay qualification timing binding changed")
    expected = {relative: binding(ROOT / relative)["sha256"] for relative in SOURCES}
    if value.get("source_manifest") != expected:
        raise ValueError("full-replay qualification source binding changed")
    acceptance = _bound_receipt(value.get("cli_acceptance_receipt"), "CLI acceptance")
    transfer = acceptance.get("s1_full_replay_transfer")
    if acceptance.get("qualified") is not True or not isinstance(transfer, dict):
        raise ValueError("CLI acceptance receipt lacks measured S1 export throughput")
    if (
        transfer.get("status") != "PASS"
        or transfer.get("hardware") != "L4"
        or transfer.get("protocol") != "s1-chunked-export/v1"
        or transfer.get("verified_on_mac") is not True
        or transfer.get("transport_source_sha256")
        != expected["research/s1_mature_20261018/train.py"]
    ):
        raise ValueError("S1 throughput must cover verified L4 chunked exports on the Mac")
    if transfer.get("transport_source_manifest") != {
        relative: binding(ROOT / relative)["sha256"] for relative in CLI_SOURCES
    }:
        raise ValueError("measured throughput CLI transport source identity changed or is absent")
    rate = _positive(transfer.get("bytes_per_second_lower_bound"), "measured throughput")
    sent = _positive(transfer.get("exported_bytes"), "measured exported bytes")
    seconds = _positive(transfer.get("elapsed_seconds"), "measured transfer seconds")
    if sent < CHUNK_BYTES or rate > sent / seconds:
        raise ValueError("throughput needs >=1 complete 1.5 GiB chunk and a conservative rate")
    arms = {}
    for arm in ("n3", "n1"):
        row = value.get("arms", {}).get(arm, {})
        observation = _bound_receipt(row.get("full_size_checkpoint_measurement"), arm)
        if (
            observation.get("status") != "PASS"
            or observation.get("arm") != arm
            or observation.get("production_shape_validated") is not True
            or observation.get("source_manifest") != expected
            or observation.get("capacity") != capacity(arm)
            or observation.get("deferred_barrier_packets_included") is not True
            or observation.get("hardware") != "L4"
            or observation.get("preparation_scope") != "capture-save-hash-archive-chunk-copy"
        ):
            raise ValueError(f"{arm} lacks production-shape full-state measurement")
        size = _positive(row.get("snapshot_bytes_upper_bound"), f"{arm} snapshot bytes")
        prepared = _positive(row.get("snapshot_prepare_seconds_upper_bound"), f"{arm} prepare")
        # One representative checkpoint cannot bound later protocol-2 byte
        # expansion or actor/Python state growth. Require an independent bound
        # for all registered checkpoints in addition to the timing observation.
        size_proof = _bound_receipt(row.get("full_horizon_size_bound"), f"{arm} size bound")
        if (
            size_proof.get("schema") != "s1-full-horizon-checkpoint-size-bound/v1"
            or size_proof.get("status") != "PASS"
            or size_proof.get("arm") != arm
            or size_proof.get("source_manifest") != expected
            or size_proof.get("snapshot_targets") != list(SNAPSHOTS)
            or size_proof.get("independent_validation") is not True
            or size_proof.get("includes_all_noncapacity_state_and_framing") is not True
        ):
            raise ValueError(f"{arm} lacks a validated full-horizon checkpoint size bound")
        additional_bytes = _positive(
            size_proof.get("additional_serialized_bytes_upper_bound"), f"{arm} additional state"
        )
        minimum_size = capacity(arm)["protocol2_array_payload_upper_bound_bytes"] + additional_bytes
        if size < minimum_size:
            raise ValueError(
                f"{arm} size bound omits full-horizon array expansion or additional state"
            )
        if (
            size < capacity(arm)["fixed_capacity_array_bytes"]
            or size < _positive(observation.get("snapshot_bytes"), f"{arm} observed bytes")
            or prepared < _positive(observation.get("prepare_seconds"), f"{arm} observed prepare")
        ):
            raise ValueError(f"{arm} full-size bound is below the measured production snapshot")
        final_bytes = _positive(row.get("final_archive_bytes_upper_bound"), f"{arm} final bytes")
        final_prepare = _positive(
            row.get("final_archive_prepare_seconds_upper_bound"), f"{arm} final prepare"
        )
        if final_bytes < _positive(
            observation.get("final_archive_bytes"), f"{arm} observed final bytes"
        ) or final_prepare < _positive(
            observation.get("final_archive_prepare_seconds"), f"{arm} final prepare"
        ):
            raise ValueError(f"{arm} final-archive bound is below the measured archive")
        # No compression credit: gzip can expand incompressible data.  One
        # percent plus 1 MiB covers deflate/tar framing and tiny index/sidecars.
        archived = math.ceil(size * 1.01) + 1024**2
        exported = len(SNAPSHOTS) * archived + final_bytes
        export_seconds = exported / rate
        total = len(SNAPSHOTS) * prepared + final_prepare + export_seconds
        arms[arm] = {
            "snapshot_bytes_upper_bound": int(size),
            "minimum_full_horizon_snapshot_bytes": minimum_size,
            "snapshot_archive_bytes_upper_bound": archived,
            "snapshot_count": len(SNAPSHOTS),
            "final_archive_bytes_upper_bound": int(final_bytes),
            "total_export_bytes_upper_bound": exported,
            "export_seconds_upper_bound": export_seconds,
            "preparation_seconds_upper_bound": len(SNAPSHOTS) * prepared + final_prepare,
            "transport_gpu_hours_upper_bound": total / 3600,
            "per_run_gpu_hours_upper_bound": (30 if arm == "n3" else 68) / 60 + total / 3600,
        }
    return {
        **value,
        "measured_export_bytes_per_second_lower_bound": rate,
        "liability_formula": "(4 * (snapshot_bytes * 1.01 + 1 MiB) + final_archive_bytes) / "
        "measured_export_bytes_per_second_lower_bound + preparation_seconds",
        "qualified_arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--measure-capacity-serialization", action="store_true")
    args = parser.parse_args()
    observation = observe_fixture(args.fixture_archive)
    result = {
        "schema": "s1-full-state-capacity/v1",
        "status": "CAPACITY_MEASURED_TRANSPORT_UNQUALIFIED",
        "platform": platform.platform(),
        "source_manifest": {relative: binding(ROOT / relative)["sha256"] for relative in SOURCES},
        "arms": {arm: projected_checkpoint(arm, observation) for arm in ("n3", "n1")},
        "local_serializer_observation": observation,
        "measured_export_bytes_per_second_lower_bound": None,
        "complete_upper_bound": None,
        "scope": "Exact production allocation formula measured on Mac and actual small-fixture "
        "torch serializer observation; no production checkpoint or L4 transfer claim",
    }
    if args.measure_capacity_serialization:
        result["measured_production_capacity_serialization"] = {
            arm: measure_capacity_serialization(args.fixture_archive, arm) for arm in ("n3", "n1")
        }
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({arm: result["arms"][arm]["fixed_capacity_array_bytes"] for arm in ("n3", "n1")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
