"""Independent saved-output audit for the exploratory first-lineage preview.

Only the standard library is imported. No checkpoint payload, historical game
trajectory, simulator, neural model, or official decision analyzer is opened.
The CLI is numerical saved-data work and runs only after separate admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

STUDY = "task-aligned-first-lineage-preview-20260924"
SEED = 2026096101
HORIZON = 3000
PROFILES = ("solo", "mixed")
CORE = (
    "native_frames",
    "hero_transitions",
    "model_forwards",
    "checkpoint_loads",
    "optimizer_updates",
)
PHYSICAL = (
    "native_frames",
    "hero_transitions",
    "opponent_transitions",
    "model_forwards",
    "model_forward_rows",
    "checkpoint_loads",
    "checkpoint_files_hashed",
    "optimizer_updates",
    "sampled_transition_draws",
    "completed_rollouts",
    "preview_calls",
    "episode_starts",
)
ROLE_CAPS = dict(zip(PHYSICAL, (192000, 192000, 480000, 6000, 192000, 1, 1, 0, 0, 0, 0, 64)))
METRICS = (
    "mass_integral",
    "ambient_food",
    "alive_frames",
    "survival_fraction",
    "boost_frames",
    "boost_fraction",
    "encounter_frames",
    "endpoint",
)
NAMESPACE = "task-aligned-20260924/preview-first-lineage-v1"


def check(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def integer(value: Any, label: str) -> int:
    check(type(value) is int and value >= 0, f"Invalid nonnegative integer: {label}")
    return value


def finite(value: Any, label: str) -> float:
    check(type(value) in (int, float) and math.isfinite(value), f"Nonfinite number: {label}")
    return float(value)


def equal_number(actual: Any, expected: float, label: str) -> None:
    actual = finite(actual, label)
    check(
        math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"Saved arithmetic differs: {label}",
    )


def counters(value: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, int]:
    check(set(value) == set(names), "Counter schema differs")
    return {name: integer(value[name], name) for name in names}


def publish(path: Path, payload: Any) -> str:
    """Create once, including failure evidence; existing artifacts are preserved."""
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


class Reader:
    """Read each explicitly named file once and reuse its authenticated bytes."""

    def __init__(self, heartbeat: Path) -> None:
        self.heartbeat = heartbeat
        self.bytes: dict[str, bytes] = {}
        self.hashes: dict[str, str] = {}
        self.parsed: dict[str, Any] = {}
        self.last_pulse = 0.0

    def pulse(self, phase: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_pulse < 2:
            return
        self.last_pulse = now
        temporary = self.heartbeat.with_name(self.heartbeat.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "study_id": STUDY,
                    "role": "audit",
                    "phase": phase,
                    "pid": os.getpid(),
                    "monotonic": now,
                    "time_unix": time.time(),
                    "files_read": len(self.bytes),
                }
            )
            + "\n"
        )
        os.replace(temporary, self.heartbeat)

    def read_bytes(self, path: str | Path, expected: str | None = None) -> bytes:
        file = Path(path).resolve()
        check(
            file.suffix.lower() not in {".pth", ".pt", ".npz", ".npy", ".jsonl", ".sqlite"},
            f"Audit must not open checkpoint/trajectory payload: {file}",
        )
        key = str(file)
        if key not in self.bytes:
            self.pulse(f"authenticate:{file.name}")
            self.bytes[key] = file.read_bytes()
            self.hashes[key] = hashlib.sha256(self.bytes[key]).hexdigest()
        if expected is not None:
            check(self.hashes[key] == expected, f"Bound file hash differs: {file}")
        return self.bytes[key]

    def json(self, path: str | Path, expected: str | None = None) -> Any:
        key = str(Path(path).resolve())
        data = self.read_bytes(path, expected)
        if key not in self.parsed:
            self.parsed[key] = json.loads(data)
        return self.parsed[key]


def authenticate_intent(intent: Mapping[str, Any], reader: Reader) -> tuple[dict, dict]:
    check(intent["study_id"] == STUDY and intent["schema_version"] == 1, "Intent study/schema")
    check(
        intent["seed"] == SEED and intent["horizon"] == HORIZON and intent["batch_worlds"] == 32,
        "Intent lineage/horizon/batch scope",
    )
    for field in ("source_files", "authority_files"):
        check(bool(intent[field]), f"Empty {field}")
        for path, digest in intent[field].items():
            check(Path(path).is_absolute(), "Bound source/authority path must be absolute")
            reader.read_bytes(path, digest)
    own = str(Path(__file__).resolve())
    check(
        own in intent["source_files"] and reader.hashes[own] == intent["source_files"][own],
        "Auditor is absent from frozen source closure",
    )
    original_binding = intent["original_intent"]
    original = reader.json(original_binding["path"], original_binding["sha256"])
    check(
        original["evaluation"]["source_config"] == intent["source_config"],
        "Original world/input source configuration differs",
    )
    check(intent["parent"] == original["parents"][str(SEED)], "Frozen parent binding differs")
    profiles = intent["profiles"]
    check(
        set(profiles) == set(PROFILES) and profiles["solo"]["opponents"] == [],
        "Preview profiles differ",
    )
    check(
        profiles["mixed"]["opponents"]
        == ["greedy_food", "greedy_food", "random_safe", "random_safe", "random_safe"],
        "Mixed opponent roster differs",
    )
    check(intent["world_seed_namespace"] == NAMESPACE, "Preview namespace differs")
    seeds = set()
    for profile in PROFILES:
        expected = [
            int.from_bytes(
                hashlib.sha256(f"{NAMESPACE}|{profile}|{i}".encode()).digest()[:8], "big"
            )
            for i in range(32)
        ]
        check(profiles[profile]["world_seeds"] == expected, "Preview seed derivation differs")
        seeds.update(expected)
    check(len(seeds) == 64, "Duplicate preview worlds")
    ledger_binding = intent["completed_training_worlds"]
    ledger = reader.json(ledger_binding["path"], ledger_binding["sha256"])["worlds"]
    training = {
        integer(seed, "training ID")
        for lanes in ledger.values()
        for episodes in lanes
        for seed in episodes
    }
    held = {
        integer(seed, "official held ID")
        for group in original["evaluation"]["world_seeds"].values()
        for seed in group
    }
    check(not seeds & training and not seeds & held, "Preview training/official-held ID overlap")
    exclusion = {
        "namespace": NAMESPACE,
        "preview_world_count": 64,
        "completed6101_training_world_count": len(training),
        "official_held_world_count": len(held),
        "training_overlap": [],
        "official_held_overlap": [],
        "completed_training_worlds": dict(ledger_binding),
        "scope": "Exact IDs versus completed6101 training and original official held IDs only; no future-lineage or all-history geometry claim",
    }
    return original, exclusion


def authenticate_candidate(intent: Mapping[str, Any], reader: Reader) -> dict[str, Any]:
    """Bind a completed old endpoint using its explicitly authorized JSON receipts only."""
    eligibility = intent["candidate_eligibility"]
    documents = {}
    for key in ("report", "completed_receipt", "supervisor"):
        path = eligibility[key]
        check(path in intent["authority_files"], f"Candidate {key} lacks authority binding")
        documents[key] = reader.json(path, intent["authority_files"][path])
    report, completed, supervisor = (
        documents[k] for k in ("report", "completed_receipt", "supervisor")
    )
    report_path = Path(eligibility["report"])
    original_sha = intent["original_intent"]["sha256"]
    check(report["status"] == "PASS" and report["lineage"] == SEED, "Candidate training report")
    check(
        report["completed_rollouts"]
        == report["selected_rollouts"]
        == eligibility["completed_rollouts"]
        == 31005,
        "Candidate is not the fixed complete rollout endpoint",
    )
    check(
        report["final_update"] == eligibility["global_update"] == 35613,
        "Candidate global update differs",
    )
    endpoint = report["final_checkpoint"]
    check(
        (report_path.parent / endpoint["path"]).resolve()
        == Path(intent["candidate"]["path"]).resolve()
        and endpoint["sha256"] == intent["candidate"]["sha256"],
        "Candidate checkpoint identity",
    )
    check(
        completed["intent_sha256"] == original_sha
        and completed["report_sha256"] == reader.hashes[str(report_path.resolve())]
        and completed["supervisor_sha256"]
        == reader.hashes[str(Path(eligibility["supervisor"]).resolve())],
        "Candidate completion receipt identity",
    )
    result = supervisor["result"]
    check(
        result.get("cause") is None
        and result["confirmed_exit"] is True
        and result["observed_returncode"] == 0
        and result["termination"] == "natural_exit",
        "Candidate training exit was not cleanly confirmed",
    )
    return {
        "checkpoint_sha256": endpoint["sha256"],
        "completed_rollouts": 31005,
        "global_update": 35613,
        "checkpoint_payload_read": False,
        "authority_report_sha256": completed["report_sha256"],
    }


def receipt(
    intent: Mapping[str, Any], identity: str, root: Path, role: str, reader: Reader
) -> tuple[dict, dict]:
    directory = root / "stages" / role
    done = reader.json(directory / "completed.json")
    check(
        done["schema_version"] == 1
        and done["stage_id"] == role
        and done["status"] == "COMPLETE"
        and done["intent_sha256"] == identity,
        "Preview completion identity differs",
    )
    check(
        done["confirmed_exit"] is True
        and done["observed_returncode"] == 0
        and done["termination"] == "natural_exit",
        "Unconfirmed/nonzero preview exit",
    )
    stage = next(row for row in intent["stages"] if row["id"] == role)
    check(
        0 <= finite(done["elapsed_seconds"], "elapsed") <= stage["cap_seconds"], "Preview stage cap"
    )
    check(
        done["source_files_sha256"] == intent["source_files"]
        and done["authority_files_sha256"] == intent["authority_files"],
        "Receipt source authority differs",
    )
    started = reader.json(directory / "started.json", done["started_sha256"])
    admitted = reader.json(directory / "admitted.json", done["admitted_sha256"])
    values = {
        "run": str(root),
        "stage_out": str(directory / "output"),
        "heartbeat": str(directory / "output/heartbeat.json"),
        "intent": intent["_intent_path"],
    }
    check(
        started["stage_id"] == role
        and started["intent_sha256"] == identity
        and started["cap_seconds"] == stage["cap_seconds"]
        and started["command"] == [part.format_map(values) for part in stage["command"]],
        "Started preview command/authority differs",
    )
    limits = intent["resource_limits"]
    check(
        admitted["available_bytes"] >= limits["admission_available_gib"] * 1024**3
        and admitted["preview_rss_bytes"] <= limits["sidecar_rss_gib"] * 1024**3
        and admitted["combined_rss_bytes"] <= limits["combined_rss_gib"] * 1024**3,
        "Admission resource receipt violates bounds",
    )
    primary = intent["primary"]
    for name in ("controller", "child"):
        check(
            any(
                row["pid"] == primary[f"{name}_pid"]
                and abs(row["create_time"] - primary[f"{name}_create_time"]) < 0.001
                for row in admitted["primary_tree"]
            ),
            f"Admission primary {name} identity differs",
        )
    check(
        -1
        <= finite(admitted["primary_heartbeat_age_seconds"], "primary heartbeat age")
        <= limits["primary_heartbeat_seconds"],
        "Admission primary heartbeat was stale",
    )
    resources = done["resource_summary"]
    check(
        integer(resources["samples"], "resource samples") > 0
        and resources["minimum_available_bytes"] >= limits["abort_available_gib"] * 1024**3
        and resources["maximum_preview_rss_bytes"] <= limits["sidecar_rss_gib"] * 1024**3
        and resources["maximum_combined_rss_bytes"] <= limits["combined_rss_gib"] * 1024**3,
        "Completed resource summary violates bounds",
    )
    report = reader.json(directory / "output/report.json", done["report_sha256"])
    check(done["counters"] == report["counters"], "Completion counters differ from saved report")
    return report, done


def reduce_case(
    raw: Mapping[str, Any], saved: Mapping[str, Any], world: int, profile: str
) -> dict[str, Any]:
    """Independently reduce one compact physical stream, without imported helpers."""
    check(raw["world_seed"] == saved["world_seed"] == world, "Case world pairing/order differs")
    descriptor = raw["initial_descriptor"]
    check(
        {"world_seed", "hero", "opponents", "food", "rng_sha256"} <= set(descriptor)
        and descriptor["world_seed"] == world,
        "Initial descriptor incomplete",
    )
    check(len(descriptor["opponents"]) == (0 if profile == "solo" else 5), "Initial roster differs")
    initial_hash = canonical(descriptor)
    check(saved["initial_sha256"] == initial_hash, "Initial descriptor identity differs")
    frames = raw["frames"]
    check(isinstance(frames, list) and 0 < len(frames) <= HORIZON, "Empty or overlong frame stream")
    mass = 0.0
    food = alive = boosted = encounters = 0
    for index, frame in enumerate(frames):
        check(set(frame) == {"alive", "mass", "ambient_food", "boost", "encounter"}, "Frame schema")
        check(
            all(type(frame[k]) is bool for k in ("alive", "boost", "encounter")),
            "Frame boolean type",
        )
        check(index == len(frames) - 1 or frame["alive"], "Frames continued after terminal death")
        value = finite(frame["mass"], "mass")
        check(
            value >= 0 and (frame["alive"] or value == 0), "Dead mass is nonzero or mass negative"
        )
        mass += value
        food += integer(frame["ambient_food"], "ambient food")
        alive += frame["alive"]
        boosted += frame["boost"]
        encounters += int(index >= 16 and frame["encounter"])
    check(not frames[-1]["alive"] or len(frames) == HORIZON, "Live episode stopped before H3000")
    expected = {
        "world_seed": world,
        "profile": profile,
        "horizon": HORIZON,
        "initial_sha256": initial_hash,
        "observed_frames": len(frames),
        "living_mass_sum": mass,
        "mass_integral": mass / HORIZON,
        "ambient_food": food,
        "alive_frames": alive,
        "survival_fraction": alive / HORIZON,
        "boost_frames": boosted,
        "boost_fraction": boosted / HORIZON,
        "encounter_frames": encounters,
        "endpoint": int(frames[-1]["alive"]),
    }
    for key, value in expected.items():
        if type(value) in (int, float):
            equal_number(saved[key], value, key)
        else:
            check(saved[key] == value, f"Saved case differs: {key}")
    physical = counters(raw["counters"], CORE)
    check(physical == counters(saved["counters"], CORE), "Saved per-case counters differ")
    check(
        physical
        == {
            "native_frames": len(frames),
            "hero_transitions": len(frames),
            "model_forwards": 0,
            "checkpoint_loads": 0,
            "optimizer_updates": 0,
        },
        "Batched work was allocated incorrectly to an individual world",
    )
    return {**expected, "initial_descriptor": descriptor}


def audit_role(
    intent: Mapping[str, Any],
    identity: str,
    root: Path,
    role: str,
    exclusion: Mapping[str, Any],
    reader: Reader,
) -> dict[str, Any]:
    report, completed = receipt(intent, identity, root, role, reader)
    check(
        report["schema_version"] == 1
        and report["study_id"] == STUDY
        and report["role"] == role
        and report["seed"] == report["lineage"] == SEED
        and report["status"] == "PASS"
        and report["intent_sha256"] == identity,
        "Evaluation report identity/status differs",
    )
    check(
        report["exploratory"] is True
        and report["promotion_eligible"] is False
        and report["selection_eligible"] is False,
        "Preview claim boundary differs",
    )
    check(
        report["runtime"] == {"device": "cpu", "intraop_threads": 2, "interop_threads": 1},
        "CPU serving contract differs",
    )
    check(
        report["bindings"]
        == {key: intent[key] for key in ("original_intent", "source_files", "authority_files")},
        "Evaluation authority binding differs",
    )
    check(
        report["seed_exclusion"] == exclusion, "Saved seed exclusion differs from independent check"
    )
    expected_policy = (
        intent["parent"]["checkpoint_sha256"] if role == "parent" else intent["candidate"]["sha256"]
    )
    check(
        report["policy_sha256"] == expected_policy and report["checkpoint"]["seed"] == SEED,
        "Loaded policy identity differs",
    )
    check(
        report["source_config"] == intent["source_config"]
        and report["contract_sha256"] == intent["evaluation_contract_sha256"]
        and canonical(report["contract"]) == intent["evaluation_contract_sha256"],
        "Loaded source or exact serving contract differs",
    )
    document = reader.json(root / "stages" / role / "output/frames.json", report["frames_sha256"])
    check(
        document["schema_version"] == 1
        and document["study_id"] == STUDY
        and document["role"] == role
        and document["seed"] == SEED,
        "Authoritative frame payload identity differs",
    )
    check(set(document["profiles"]) == set(report["profiles"]) == set(PROFILES), "Missing profile")
    loaded = counters(report["loaded_counters"], PHYSICAL)
    check(
        loaded
        == {key: int(key in ("checkpoint_loads", "checkpoint_files_hashed")) for key in PHYSICAL},
        "Hidden operations occurred during policy loading",
    )
    observed = dict(loaded)
    result_profiles = {}
    for profile in PROFILES:
        reader.pulse(f"reduce:{role}:{profile}", force=True)
        raw_cases, summaries = document["profiles"][profile], report["profiles"][profile]["cases"]
        check(len(raw_cases) == len(summaries) == 32, "Profile omitted or duplicated worlds")
        reduced = []
        for raw, summary, world in zip(
            raw_cases, summaries, intent["profiles"][profile]["world_seeds"]
        ):
            reduced.append(reduce_case(raw, summary, world, profile))
            reader.pulse(f"reduce:{role}:{profile}")
        shared = counters(
            report["profiles"][profile]["shared_counters"], (*CORE, "model_forward_rows")
        )
        forward_calls = max(row["observed_frames"] for row in reduced)
        expected_shared = {key: 0 for key in (*CORE, "model_forward_rows")}
        expected_shared.update(model_forwards=forward_calls, model_forward_rows=32 * forward_calls)
        check(shared == expected_shared, "Batched forward count/row attribution differs")
        frame_count = sum(row["observed_frames"] for row in reduced)
        observed["native_frames"] += frame_count
        observed["hero_transitions"] += frame_count
        observed["model_forwards"] += forward_calls
        observed["model_forward_rows"] += 32 * forward_calls
        observed["episode_starts"] += 32
        means = {metric: sum(row[metric] for row in reduced) / 32 for metric in METRICS}
        check(
            set(report["profiles"][profile]["means"]) == set(METRICS), "Mean metric schema differs"
        )
        for metric, mean in means.items():
            equal_number(report["profiles"][profile]["means"][metric], mean, f"mean:{metric}")
        result_profiles[profile] = {"cases": reduced, "means": means}
    physical = counters(report["counters"], PHYSICAL)
    for name in PHYSICAL:
        check(physical[name] <= ROLE_CAPS[name], f"Role physical cap: {name}")
        if name != "opponent_transitions":
            check(physical[name] == observed[name], f"Role counter arithmetic: {name}")
    mixed_frames = sum(row["observed_frames"] for row in result_profiles["mixed"]["cases"])
    check(physical["opponent_transitions"] <= 5 * mixed_frames, "Opponent transition bound")
    return {
        "profiles": result_profiles,
        "counters": physical,
        "report_sha256": completed["report_sha256"],
        "frames_sha256": report["frames_sha256"],
        "policy_sha256": expected_policy,
    }


def run(intent_path: Path, out: Path, heartbeat: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    check(not any(out.iterdir()), "Audit output must be empty; no retry is permitted")
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    reader = Reader(heartbeat)
    identity = None
    try:
        intent = reader.json(intent_path)
        identity = reader.hashes[str(intent_path.resolve())]
        root = Path(intent.get("run_root", str(intent_path.parent / "run"))).resolve()
        check(out.resolve() == root / "stages/audit/output", "Audit output outside frozen stage")
        original, exclusion = authenticate_intent(intent, reader)
        eligibility = authenticate_candidate(intent, reader)
        runtime_intent = {**intent, "_intent_path": str(intent_path.resolve())}
        roles = {
            role: audit_role(runtime_intent, identity, root, role, exclusion, reader)
            for role in ("parent", "candidate")
        }
        totals = {key: sum(roles[role]["counters"][key] for role in roles) for key in PHYSICAL}
        for name, cap in intent["physical_caps"].items():
            actual = totals["episode_starts"] if name == "hero_episodes" else totals[name]
            check(actual <= integer(cap, name), f"Combined physical cap: {name}")
        check(
            totals["checkpoint_loads"] == 2
            and totals["episode_starts"] == 128
            and totals["optimizer_updates"] == 0,
            "Preview fixed work scope differs",
        )
        comparisons = {}
        for profile in PROFILES:
            parent = roles["parent"]["profiles"][profile]["cases"]
            candidate = roles["candidate"]["profiles"][profile]["cases"]
            for old, new in zip(parent, candidate):
                check(
                    old["world_seed"] == new["world_seed"]
                    and old["initial_sha256"] == new["initial_sha256"]
                    and old["initial_descriptor"] == new["initial_descriptor"],
                    "Controllers did not begin from exactly paired native worlds",
                )
            metrics = {}
            for metric in METRICS:
                deltas = [new[metric] - old[metric] for old, new in zip(parent, candidate)]
                metrics[metric] = {
                    "parent_mean": sum(row[metric] for row in parent) / 32,
                    "candidate_mean": sum(row[metric] for row in candidate) / 32,
                    "mean_delta": sum(deltas) / 32,
                    "candidate_wins": sum(d > 0 for d in deltas),
                    "ties": sum(d == 0 for d in deltas),
                    "candidate_losses": sum(d < 0 for d in deltas),
                    "world_deltas": deltas,
                }
            comparisons[profile] = {
                "world_seeds": intent["profiles"][profile]["world_seeds"],
                "n_pairs": 32,
                "metrics": metrics,
            }
        comparison = {
            "schema_version": 1,
            "study_id": STUDY,
            "seed": SEED,
            "horizon": HORIZON,
            "status": "EXPLORATORY_ONLY",
            "intent_sha256": identity,
            "profiles": comparisons,
            "parent_sha256": roles["parent"]["policy_sha256"],
            "candidate_sha256": roles["candidate"]["policy_sha256"],
            "promotion_eligible": False,
            "selection_eligible": False,
            "training_decision": None,
            "interpretation": "One fixed trained lineage on new paired worlds. Descriptive differences only; no official gate, population claim, checkpoint selection, or change to ongoing training.",
        }
        comparison_sha = publish(out / "comparison.json", comparison)
        report = {
            "schema_version": 1,
            "study_id": STUDY,
            "role": "audit",
            "status": "PASS",
            "intent_sha256": identity,
            "seed": SEED,
            "exploratory": True,
            "promotion_eligible": False,
            "selection_eligible": False,
            "counters": dict.fromkeys(PHYSICAL, 0),
            "audited_evaluation_counters": totals,
            "candidate_eligibility": eligibility,
            "seed_exclusion": exclusion,
            "inputs": {
                role: {
                    key: roles[role][key]
                    for key in ("report_sha256", "frames_sha256", "policy_sha256")
                }
                for role in roles
            },
            "source_files_sha256": intent["source_files"],
            "authority_files_sha256": intent["authority_files"],
            "artifacts": [{"path": "comparison.json", "sha256": comparison_sha}],
            "physical_file_reads": len(reader.bytes),
            "json_documents_parsed": len(reader.parsed),
            "read_once_scope": "Each named input file read once; authoritative frames.json parsed once per role. Prefix copies and checkpoint payloads were not opened.",
            "limitations": [
                "Opponent transition count is hash-bound and bounded by five per mixed native frame; hero-only evidence does not independently reconstruct individual opponent transitions.",
                "Policy loading and inference identities rely on frozen evaluator code and authenticated receipts; audit performs zero neural queries.",
                "Sustained CPU and continuous process liveness rely on the frozen supervisor and its completion receipt; resource sample logs are not replayed by this audit.",
            ],
        }
        publish(out / "report.json", report)
        reader.pulse("complete", force=True)
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "study_id": STUDY,
            "role": "audit",
            "status": "FAIL",
            "intent_sha256": identity,
            "error": f"{type(exc).__name__}: {exc}",
            "retry_authorized": False,
            "promotion_eligible": False,
            "selection_eligible": False,
            "counters": dict.fromkeys(PHYSICAL, 0),
            "physical_file_reads": len(reader.bytes),
            "json_documents_parsed": len(reader.parsed),
        }
        if not (out / "report.json").exists():
            publish(out / "report.json", failure)
        reader.pulse("failed", force=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    args = parser.parse_args()
    return run(args.intent, args.out, args.heartbeat)


if __name__ == "__main__":
    raise SystemExit(main())
