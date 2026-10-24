"""Independent, read-only packet verification and create-only Mac evidence.

Run with the project venv and single numerical threads. No provider operations.
The candidate imports come exclusively from an unmodified exact-pin worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def verified(path: Path) -> dict:
    data = path.read_bytes()
    checksum = Path(str(path) + ".sha256").read_text().split()
    assert checksum == [digest(data), path.name], path
    assert Path(str(path) + ".complete").read_bytes() == b"complete\n", path
    return {"path": str(path), "sha256": digest(data), "size": len(data)}


def publish(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    with Path(str(path) + ".sha256").open("x") as stream:
        stream.write(f"{digest(data)}  {path.name}\n")
    with Path(str(path) + ".complete").open("xb") as stream:
        stream.write(b"complete\n")


def run(packet: Path, pin_root: Path, output: Path) -> dict:
    assert platform.system() == "Darwin"
    sys.path.insert(0, str(pin_root))
    import numpy as np
    import torch

    from research.gpu_sim_spike import native_compile, sparse_analyze, vector_exactness_probe
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory, Vector61Params

    torch.set_num_threads(1)
    pin = "b7dfa1fc35321a388813122b3acfe033d1367782"
    manifest, receipt = load(packet / "manifest.json"), load(packet / "receipt.json")
    bench = load(packet / "exports/sparse-bench.json")
    assert manifest["commit"] == receipt["commit"] == bench["commit"] == pin
    assert digest((packet / "manifest.json").read_bytes()) == receipt["manifest_sha256"]
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True
    assert not any(receipt.get(k) for k in ("error", "cleanup_error", "failure_class"))
    assert receipt["terminal"]["returncode"] == 0
    assert receipt["terminal"]["job_id"] == manifest["job_id"] == receipt["job_id"]
    assert manifest["env"]["SPARSE_BENCH_CANARY"] == "1"
    payloads = []
    inventories = [
        receipt["exports"],
        receipt["terminal"]["exports"],
        receipt["terminal"]["export_scanner"]["inventory"],
    ]
    for inventory in inventories:
        assert len(inventory) == len({row["path"] for row in inventory}) == 9
        assert sorted(inventory, key=lambda r: r["path"]) == sorted(
            receipt["exports"], key=lambda r: r["path"]
        )
    for row in receipt["exports"]:
        name = Path(row["path"])
        assert not name.is_absolute() and ".." not in name.parts
        record = verified(packet / "exports" / name)
        assert record["sha256"] == row["sha256"] and record["size"] == row["size"]
        payloads.append(record)
    assert {
        str(p.relative_to(packet / "exports"))
        for p in (packet / "exports").rglob("*")
        if p.is_file() and not p.name.endswith((".sha256", ".complete"))
    } == {row["path"] for row in receipt["exports"]}
    for name, sha in bench["source_sha256"].items():
        assert digest((pin_root / "research/gpu_sim_spike" / name).read_bytes()) == sha, name
    plan = load(packet / "staging/source-plan.json")
    assert plan["commit"] == pin
    archive = packet / "staging/source.tar.gz"
    assert digest(archive.read_bytes()) == plan["fallback_sha256"]
    assert archive.stat().st_size == plan["fallback_size"]
    with tarfile.open(archive) as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        assert {member.name for member in members} == set(plan["files"])
        for member in members:
            data = tar.extractfile(member).read()
            expected = plan["files"][member.name]
            assert digest(data) == expected["sha256"] and len(data) == expected["size"]
            assert data == (pin_root / member.name).read_bytes(), member.name
    inputs = []
    for row in manifest["inputs"]:
        path = Path(row["path"])
        assert digest(path.read_bytes()) == row["sha256"]
        inputs.append({"path": str(path), "sha256": row["sha256"], "size": path.stat().st_size})
        if row["dest"] == "metadata.tar":
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    if member.name.endswith("verified-ledger.json"):
                        ledger = json.loads(tar.extractfile(member).read())
                        assert ledger["units"] == []
    compile_binding = manifest["native_cuda_compile"]["receipt"]
    compile_path = Path(compile_binding["path"])
    assert digest(compile_path.read_bytes()) == compile_binding["sha256"]
    compile_receipt = load(compile_path)
    compile_manifest = load(compile_path.parent / "manifest.json")
    assert compile_receipt["commit"] == compile_manifest["commit"] == pin
    assert compile_receipt["status"] == "PASS" and compile_receipt["stop_verified"] is True
    assert digest((compile_path.parent / "manifest.json").read_bytes()) == (
        compile_receipt["manifest_sha256"]
    )
    assert compile_receipt["terminal"]["returncode"] == 0
    assert not any(compile_receipt.get(k) for k in ("error", "cleanup_error", "failure_class"))
    compile_payloads = []
    for row in compile_receipt["exports"]:
        record = verified(compile_path.parent / "exports" / row["path"])
        assert record["sha256"] == row["sha256"] and record["size"] == row["size"]
        compile_payloads.append(record)
    compile_report = load(compile_path.parent / "exports/native-compile.json")
    hashes, specs = native_compile.inventory(pin_root, pin)
    assert compile_report["source_sha256"] == hashes and compile_report["extensions"] == specs
    assert compile_report["architectures"] == "8.0;8.9" and compile_report["status"] == "PASS"
    assert compile_report["commit"] == pin and compile_report["kernels_launched"] is False
    mac_reduced = sparse_analyze.aggregate([packet / "exports"], pin, canary=True)
    assert mac_reduced["verdict"] == "CANARY_DIAGNOSTIC"
    assert mac_reduced["complete_allocation"]
    publish(output / "mac-canary-reducer.json", mac_reduced)
    fixtures, timings = [], []
    for cell in bench["cells"]:
        assert cell["worlds"] == 1024
        path = packet / "exports" / cell["static"]["feature_oracle"]["recorded_fixture"]["path"]
        probe = vector_exactness_probe.probe(path, "cpu")
        assert probe["status"] == "PASS"
        publish(output / f"mac-replay-{cell['mix']}.json", probe)
        with np.load(path, allow_pickle=False) as saved:
            cfg = BatchSimConfig(**json.loads(str(saved["config_json"])))
            params_dict = json.loads(str(saved["params_json"]))
            params_dict["actions"] = tuple(map(tuple, params_dict["actions"]))
            params = Vector61Params(**params_dict)
            snakes = saved["alive"].shape[0]
            ref = SimpleNamespace(
                cfg=cfg, E=1, S=snakes, s=cfg.segment_size, cap=saved["bodies"].shape[1]
            )
            for name in ("bodies", "head_ptr", "seg_count", "length", "alive", "direction"):
                setattr(ref, name, saved[name][None].copy())
            ref.food_cells = [
                [tuple(map(int, xy)) for xy in saved["food_cells"][: int(saved["food_count"])]]
            ]
            memory = Vector61Memory(1, snakes)
            memory.prev_dist[:] = saved["prev_dist"][None]
            memory.prev_id[:] = saved["prev_id"][None]
            rows = np.array([(0, slot) for slot in range(snakes)], dtype=np.int64)
            features = Vector61Featurizer(params).featurize(ref, rows, memory, update_memory=True)
        first_sha = digest(features.tobytes())
        repeated_sha = digest(np.tile(features, (1024, 1)).tobytes())
        for backend in ("dense", "static"):
            oracle = cell[backend]["feature_oracle"]
            assert oracle["match"] is True and oracle["rows"] == 6144 and oracle["columns"] == 61
            assert oracle["first_mismatch"] is None
            assert len(oracle["column_mismatches"]) == 61
            assert all(
                c["mismatch_count"] == 0 and c["first_mismatch"] is None
                for c in oracle["column_mismatches"]
            )
            assert oracle["actual_sha256"] == oracle["expected_sha256"] == repeated_sha
            samples = cell[backend]["samples"]
            assert len(samples) == 1
            for sample in samples:
                assert math.isclose(
                    sample["complete_world_steps_per_second"],
                    1024 / sample["tick_seconds"],
                    rel_tol=1e-15,
                )
        dense = statistics.median(s["tick_seconds"] for s in cell["dense"]["samples"])
        static = statistics.median(s["tick_seconds"] for s in cell["static"]["samples"])
        assert dense / static == cell["speedup"]
        timings.append(
            {
                "mix": cell["mix"],
                "dense_tick_seconds": dense,
                "static_tick_seconds": static,
                "speedup": dense / static,
                "numerically_at_least_2x": dense / static >= 2,
                "scored_main_measurement": False,
            }
        )
        fixtures.append(
            {
                "mix": cell["mix"],
                **verified(path),
                "rows": snakes,
                "columns": 61,
                "cpu_first_world_sha256": first_sha,
                "cpu_repeated_e1024_sha256": repeated_sha,
                "probe_records": len(probe["records"]),
                "status": "PASS",
            }
        )
    result = {
        "status": "PASS",
        "scope": "Mac packet integrity and recorded component fixtures",
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "pin": pin,
        "pin_root": str(pin_root),
        "exports": payloads,
        "source_plan_files_verified": len(plan["files"]),
        "package_sources_verified": len(bench["source_sha256"]),
        "inputs": inputs,
        "compile_receipt": compile_binding,
        "compile_exports": compile_payloads,
        "compile_only_not_runtime_parity": True,
        "fixtures": fixtures,
        "timings": timings,
        "mac_reducer_verdict": mac_reduced["verdict"],
        "per_stage_flood_timing_seconds": None,
        "limitations": [
            "Mac cannot rerun native CUDA; native actual feature matrices are not exported.",
            "Each saved state is one independent world, replicated at E1024.",
            "Canary has one repeat and zero warmups; registered main decision is absent.",
            "Component identity does not qualify engine mechanics or closed-loop policies.",
        ],
    }
    publish(output / "independent-verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--pin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.packet, args.pin_root, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "source_plan_files_verified",
                    "package_sources_verified",
                    "fixtures",
                    "timings",
                )
            },
            indent=2,
        )
    )
