"""Admitted, exploratory single-lineage preview; never an official promotion gate.

Importing this file performs no numerical imports or artifact reads. Each child
loads one frozen policy and uses it for exactly two E32/H3000 profiles.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import threading
import traceback
from typing import Any, Mapping

STUDY_ID = "task-aligned-first-lineage-preview-20260924"
LINEAGE = 2026096101
HORIZON = 3000
CAPS = {
    "native_frames": 192000,
    "hero_transitions": 192000,
    "opponent_transitions": 480000,
    "model_forwards": 6000,
    "model_forward_rows": 192000,
    "checkpoint_loads": 1,
    "checkpoint_files_hashed": 1,
    "optimizer_updates": 0,
    "sampled_transition_draws": 0,
    "completed_rollouts": 0,
    "preview_calls": 0,
    "episode_starts": 64,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def publish(path: Path, value: Any) -> None:
    """Create only, including on failure; no prior result is replaced."""
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class Heartbeat:
    def __init__(self, path: Path, role: str) -> None:
        self.path = path
        self.role = role
        self.last = 0.0
        self.phase = "startup"
        self.progress: dict[str, Any] = {}
        self.meter: Any = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self) -> None:
        while not self.stop_event.wait(5):
            self.write(force=True)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=6)

    def write(self, phase: str | None = None, force: bool = False) -> None:
        if phase is not None:
            self.phase = phase
        now = time.monotonic()
        if not force and now - self.last < 5:
            return
        self.last = now
        value = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "pid": os.getpid(),
            "role": self.role,
            "phase": self.phase,
            "time_unix": time.time(),
            "progress": self.progress,
            "counters": self.meter.as_dict() if self.meter is not None else {},
        }
        temp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        with self.lock:
            with temp.open("w") as handle:
                json.dump(value, handle, sort_keys=True)
            os.replace(temp, self.path)


def validate_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("study_id") != STUDY_ID or value.get("schema_version") != 1:
        raise ValueError("Preview study/schema binding differs")
    if value.get("seed") != LINEAGE or value.get("horizon") != HORIZON:
        raise ValueError("Preview is fixed to lineage6101 and H3000")
    profiles = value["profiles"]
    if set(profiles) != {"solo", "mixed"} or profiles["solo"]["opponents"] != []:
        raise ValueError("Exactly solo and mixed profiles are admitted")
    if profiles["mixed"]["opponents"] != [
        "greedy_food",
        "greedy_food",
        "random_safe",
        "random_safe",
        "random_safe",
    ]:
        raise ValueError("Mixed roster differs from the frozen two-greedy/three-random task")
    all_seeds = []
    for profile in profiles.values():
        seeds = profile["world_seeds"]
        if len(seeds) != 32 or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Each profile needs exactly32 nonnegative integer world seeds")
        all_seeds.extend(seeds)
    if len(set(all_seeds)) != 64:
        raise ValueError("Preview world seeds must be distinct across profiles")
    if value["parent"]["seed"] != LINEAGE:
        raise ValueError("Parent lineage differs")
    for field in ("source_files", "authority_files"):
        if not value[field]:
            raise ValueError(f"Empty {field} binding")
        for raw_path, expected in value[field].items():
            path = Path(raw_path)
            if not path.is_absolute() or digest(path) != expected:
                raise ValueError(f"Frozen {field} differs: {raw_path}")
    original = value["original_intent"]
    original_path = Path(original["path"])
    if digest(original_path) != original["sha256"]:
        raise ValueError("Original study intent changed")
    source = json.loads(original_path.read_text())
    if source["evaluation"]["source_config"] != value["source_config"]:
        raise ValueError("Preview source configuration differs from original intent")
    namespace = "task-aligned-20260924/preview-first-lineage-v1"
    for profile, entry in profiles.items():
        expected = [
            int.from_bytes(
                hashlib.sha256(f"{namespace}|{profile}|{index}".encode()).digest()[:8], "big"
            )
            for index in range(32)
        ]
        if entry["world_seeds"] != expected:
            raise ValueError("Preview seed derivation differs from fixed fresh namespace")
    binding = value["completed_training_worlds"]
    ledger_path = Path(binding["path"])
    if digest(ledger_path) != binding["sha256"]:
        raise ValueError("Completed6101 training world ledger changed")
    ledger = json.loads(ledger_path.read_text())["worlds"]
    training = {seed for lanes in ledger.values() for episodes in lanes for seed in episodes}
    held = {seed for seeds in source["evaluation"]["world_seeds"].values() for seed in seeds}
    preview = set(all_seeds)
    if preview.intersection(training) or preview.intersection(held):
        raise ValueError("Preview worlds overlap completed6101 training or official held IDs")
    own_path = str(Path(__file__).resolve())
    if value["source_files"].get(own_path) != digest(Path(own_path)):
        raise ValueError("Preview executable is not in the frozen source closure")
    return {
        "namespace": namespace,
        "preview_world_count": len(preview),
        "completed6101_training_world_count": len(training),
        "official_held_world_count": len(held),
        "training_overlap": [],
        "official_held_overlap": [],
        "completed_training_worlds": dict(binding),
        "scope": "Exact IDs versus completed6101 training and original official held IDs only; no future-lineage or all-history geometry claim",
    }


def reduce_trace(trace: Any, world_seed: int, profile: str) -> tuple[dict, dict]:
    frames = [dataclasses.asdict(frame) for frame in trace.frames]
    if not 0 < len(frames) <= HORIZON:
        raise ValueError("Empty or overlong evaluation trace")
    dead = False
    for row in frames:
        if dead or any(type(row[key]) is not bool for key in ("alive", "boost", "encounter")):
            raise ValueError("Trace continues after death or has invalid frame flags")
        if not math.isfinite(row["mass"]) or row["mass"] < 0 or (not row["alive"] and row["mass"]):
            raise ValueError("Invalid living mass")
        if type(row["ambient_food"]) is not int or row["ambient_food"] < 0:
            raise ValueError("Invalid ambient-food count")
        dead = not row["alive"]
    if not dead and len(frames) != HORIZON:
        raise ValueError("Live trace truncated before full horizon")
    if trace.counters["native_frames"] != len(frames) or trace.counters["hero_transitions"] != len(
        frames
    ):
        raise ValueError("Physical frame count differs from trace")
    mass = sum(row["mass"] for row in frames)
    alive = sum(row["alive"] for row in frames)
    boost = sum(row["boost"] for row in frames)
    # Match the sealed screen's post-initialization encounter convention.
    encounter = sum(row["encounter"] for row in frames[16:])
    case = {
        "world_seed": world_seed,
        "profile": profile,
        "horizon": HORIZON,
        "initial_sha256": canonical(trace.initial_descriptor),
        "observed_frames": len(frames),
        "living_mass_sum": mass,
        "mass_integral": mass / HORIZON,
        "ambient_food": sum(row["ambient_food"] for row in frames),
        "alive_frames": alive,
        "survival_fraction": alive / HORIZON,
        "boost_frames": boost,
        "boost_fraction": boost / HORIZON,
        "encounter_frames": encounter,
        "endpoint": int(not dead),
        "counters": dict(trace.counters),
    }
    raw = {
        "world_seed": world_seed,
        "initial_descriptor": dict(trace.initial_descriptor),
        "counters": dict(trace.counters),
        "frames": frames,
    }
    return case, raw


def run(intent_path: Path, role: str, out: Path, heartbeat_path: Path) -> int:
    # Existing result directories may hold earlier evidence: never retry them.
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Output directory must be empty; preview retries are not allowed")
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    beat = Heartbeat(heartbeat_path, role)
    report: dict[str, Any] = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "role": role,
        "seed": LINEAGE,
        "lineage": LINEAGE,
        "exploratory": True,
        "promotion_eligible": False,
        "selection_eligible": False,
        "profiles": {},
        "counter_caps": CAPS,
        "started_unix": time.time(),
        "load_attempted": False,
    }
    raw_profiles: dict[str, Any] = {}
    meter = None
    try:
        beat.write("startup", force=True)
        intent = json.loads(intent_path.read_text())
        report["intent_sha256"] = digest(intent_path)
        report["intent_path"] = str(intent_path.resolve())
        report["seed_exclusion"] = validate_intent(intent)
        publish(out / "seed-exclusion.json", report["seed_exclusion"])
        report["bindings"] = {
            key: intent[key] for key in ("original_intent", "source_files", "authority_files")
        }
        beat.write("validated_importing", force=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import torch
        from research.task_aligned_20260924 import adapter
        from research.task_aligned_20260924.evaluation import EpisodeRequest

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 2 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("CPU thread contract failed")
        report["runtime"] = {"device": "cpu", "intraop_threads": 2, "interop_threads": 1}

        class BoundedCounters(adapter.PhysicalCounters):
            def __setattr__(self, name: str, value: Any) -> None:
                if name in CAPS and (type(value) is not int or value < 0 or value > CAPS[name]):
                    raise RuntimeError(f"Preview physical counter cap: {name}={value}")
                super().__setattr__(name, value)

        meter = BoundedCounters()
        beat.meter = meter
        report["load_attempted"] = True
        beat.write("loading", force=True)
        if role == "parent":
            binding = dict(intent["parent"])
            binding["report_path"] = Path(binding["report_path"])
            binding["checkpoint_path"] = Path(binding["checkpoint_path"])
            loaded = adapter.load_parent(adapter.ParentSpec(**binding), device="cpu")
            policy_sha = binding["checkpoint_sha256"]
        else:
            loaded = adapter.load_candidate(
                Path(intent["candidate"]["path"]), intent["candidate"]["sha256"], device="cpu"
            )
            policy_sha = intent["candidate"]["sha256"]
        report["loaded_counters"] = loaded.counters.as_dict()
        for name, value in loaded.counters.as_dict().items():
            setattr(meter, name, getattr(meter, name) + value)
        if role == "candidate" and (
            loaded.native["completed_rollouts"] != 31005
            or loaded.native["planned_rollouts"] != 31005
            or loaded.native["update_counter"] != 35613
        ):
            raise ValueError("Preview candidate is not the fixed completed31005 endpoint")
        if loaded.metadata["seed"] != LINEAGE or loaded.config != intent["source_config"]:
            raise ValueError("Loaded lineage/source configuration differs")
        report.update(
            policy_sha256=policy_sha,
            checkpoint=loaded.metadata,
            source_config=loaded.config,
            contract=adapter.evaluation_contract(loaded.config),
        )
        report["contract_sha256"] = canonical(report["contract"])
        if report["contract_sha256"] != intent["evaluation_contract_sha256"]:
            raise ValueError("Loaded serving contract differs from frozen preview intent")
        beat.write("loaded", force=True)

        for profile in ("solo", "mixed"):
            entry = intent["profiles"][profile]
            requests = [
                EpisodeRequest(seed, profile, tuple(entry["opponents"]), HORIZON)
                for seed in entry["world_seeds"]
            ]
            beat.progress = {"profile": profile}
            beat.write("evaluating", force=True)

            traces, shared = adapter.run_batch_episodes(
                loaded.model,
                loaded.config,
                requests,
                device="cpu",
                meter=meter,
            )
            if len(traces) != 32:
                raise RuntimeError("Preview batch omitted worlds")
            cases, raw = [], []
            for request, trace in zip(requests, traces):
                case, item = reduce_trace(trace, request.world_seed, profile)
                cases.append(case)
                raw.append(item)
            metrics = (
                "mass_integral",
                "ambient_food",
                "alive_frames",
                "survival_fraction",
                "boost_frames",
                "boost_fraction",
                "encounter_frames",
                "endpoint",
            )
            report["profiles"][profile] = {
                "cases": cases,
                "shared_counters": {**shared, "model_forward_rows": shared["model_forwards"] * 32},
                "means": {name: sum(row[name] for row in cases) / 32 for name in metrics},
            }
            raw_profiles[profile] = raw
            publish(
                out / f"frames-{profile}.json",
                {
                    "schema_version": 1,
                    "role": role,
                    "seed": LINEAGE,
                    "profile": profile,
                    "cases": raw,
                },
            )
            report["profiles"][profile]["frames_artifact"] = {
                "path": f"frames-{profile}.json",
                "sha256": digest(out / f"frames-{profile}.json"),
            }
            beat.write("profile_complete", force=True)
        payload = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "role": role,
            "seed": LINEAGE,
            "profiles": raw_profiles,
        }
        publish(out / "frames.json", payload)
        report["frames_sha256"] = digest(out / "frames.json")
        report["counters"] = meter.as_dict()
        report["status"] = "PASS"
        report["finished_unix"] = time.time()
        publish(out / "report.json", report)
        beat.write("complete", force=True)
        return 0
    except BaseException as exc:
        report.update(
            status="FAIL",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            finished_unix=time.time(),
            counters=meter.as_dict() if meter is not None else {},
        )
        if report["load_attempted"] and "loaded_counters" not in report:
            report["checkpoint_load_count_unknown_on_loader_failure"] = True
        publish(
            out / "prefix.json",
            {
                "schema_version": 1,
                "role": role,
                "completed_profiles": raw_profiles,
                "last_decision": beat.progress,
                "counters": report["counters"],
                "limitation": "Incomplete batch frames are unavailable if the immutable adapter raises",
            },
        )
        if not (out / "report.json").exists():
            publish(out / "report.json", report)
        beat.write("failed", force=True)
        return 1
    finally:
        beat.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--role", choices=("parent", "candidate"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    args = parser.parse_args()
    return run(args.intent, args.role, args.out, args.heartbeat)


if __name__ == "__main__":
    raise SystemExit(main())
