#!/usr/bin/env python3
"""T23 R0 paired, terminal-hero H5000 evaluator.

This is deliberately a thin, CPU-only wrapper around ``run_simd_eval``.  It is
not a new gate: it materializes the T23-owned world bank, evaluates the same
worlds with one hero at a time, and writes independently resumable shard
receipts.  The analysis program is the only decision authority.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "t23-eval-shard/v1"
HORIZON = 5000
TRAINING_SEEDS = (7101, 7102, 7103, 7104, 7105)
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
COLAB_ROOT = Path("/content/drive/MyDrive/t16_colab/t23-r0")
MAC_ROOT = Path(
    "/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/"
    "t16_colab/t23-r0"
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    """Bind the evaluation implementation independently of machine-local paths."""
    paths = {
        p
        for directory in ("src/core", "src/model", "src/evaluation", "src/simd_env")
        for p in (REPO / directory).glob("*.py")
    }
    paths.update(
        REPO / relative
        for relative in (
            "src/scripts/tournament_eval.py",
            "src/scripts/eval_stats.py",
            "research/apex_safety_20260926/dev_screen.py",
            "research/redesign_scope_20261007/grid_h5000_identity.py",
            "research/t23_r0_screen_20261017/bank.py",
            "research/t23_r0_screen_20261017/evaluate.py",
        )
    )
    return {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(paths)}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(canonical(payload) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_bank(bank_path: Path, bank: Mapping[str, Any], profile_digest: str) -> None:
    """Fail closed unless the generated bank and its collision audit bind together."""
    audit_path = bank_path.with_name("SEED_AUDIT.json")
    audit = _json(audit_path)
    if bank.get("schema") != "t23-r0-world-bank/v1" or audit.get("pass") is not True:
        raise ValueError("T23 world bank/audit is not a passing registered artifact")
    if audit.get("world_bank_semantic_sha256") != bank.get("allocation_semantic_sha256"):
        raise ValueError("seed audit is not bound to this world bank allocation")
    file_binding = audit.get("world_bank_file")
    if not isinstance(file_binding, Mapping) or file_binding.get("sha256") != sha256_file(
        bank_path
    ):
        raise ValueError("seed audit does not bind the bytes of this world bank file")
    profile = bank.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("world bank profile binding missing")
    if profile.get("base_profile_digest") != profile_digest:
        raise ValueError("world bank base profile differs from active strict profile")
    required = {
        "engine": "grid",
        "ego2s_forward": "rowwise",
        "ego2s_mask": "advisory",
        "veto": False,
    }
    if any(profile.get(key) != value for key, value in required.items()):
        raise ValueError("world bank engine semantics differ from T23 preregistration")


def _find_values(value: Any, keys: set[str]) -> list[Any]:
    """Find explicit provenance values without trusting a checkpoint filename."""
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in keys:
                found.append(item)
            found.extend(_find_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_values(item, keys))
    return found


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def training_provenance(root: Path, seed: int) -> dict[str, Any]:
    """Fail-closed training receipt used by every candidate arm.

    T23 has external training artifacts, whose field names have evolved.  The
    checks therefore accept documented aliases, but require a real summary,
    matching seed, an observed eight-million transition value, a clean status,
    and no recorded tripwire/halt indication.
    """
    seed_dir = root / f"seed-{seed}"
    summary_path = seed_dir / "summary.json"
    manifest_path = seed_dir / "manifest.jsonl"
    telemetry_path = seed_dir / "log.jsonl"
    run_log = root / "run.log"
    problems: list[str] = []
    tripwires: list[str] = []
    summary: dict[str, Any] = {}
    if not summary_path.is_file():
        problems.append("missing summary.json")
    else:
        try:
            summary = _json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"unreadable summary.json: {exc}")
    args = summary.get("args") if isinstance(summary.get("args"), Mapping) else {}
    if str(args.get("seed")) != str(seed):
        problems.append("summary seed identity missing or mismatched")
    # Do not recurse into args: args.transitions is the requested budget, not
    # evidence that this run reached it.
    transition_values = [
        summary[key]
        for key in ("transitions", "learner_transitions", "transition_count", "actual_transitions")
        if key in summary
    ]
    transitions = [
        int(v) for v in transition_values if isinstance(v, (int, float, str)) and str(v).isdigit()
    ]
    if not any(value >= 8_000_000 for value in transitions):
        problems.append("summary lacks completed 8000000 transition budget")
    if summary.get("stopped_for_time") is not False:
        problems.append("summary stopped_for_time must be false")
    actor_exitcodes = summary.get("actor_exitcodes")
    if not isinstance(actor_exitcodes, list) or len(actor_exitcodes) != 6 or any(actor_exitcodes):
        problems.append("summary requires six zero actor_exitcodes")
    halt = summary.get("halt")
    if halt not in (None, False, "", "none", "None"):
        tripwires.append(f"trainer halt: {halt}")
    if not run_log.is_file():
        problems.append("missing root run.log GPU/exit provenance")
    else:
        log = run_log.read_text(errors="replace")
        start = re.search(rf"seed\s+{seed}\s+start\b", log)
        exit_match = re.search(rf"seed\s+{seed}\s+exit\s+0\b", log)
        if start is None or exit_match is None or exit_match.start() < start.start():
            problems.append("run.log lacks seed clean exit")
        else:
            # Interleaved runtimes are legal, so associate this seed with the
            # nearest preceding runtime header rather than any A100 string.
            headers = list(re.finditer(r"^commit .* gpu .* start .*?$", log, re.MULTILINE))
            if not headers or any(
                not re.search(r"A100-SXM4-80GB", h.group(0), re.IGNORECASE) for h in headers
            ):
                problems.append("run.log has a missing or non-A100-SXM4-80GB runtime header")
    raw_lines = (
        telemetry_path.read_text(errors="replace").splitlines() if telemetry_path.is_file() else []
    )
    telemetry = _load_json_lines(telemetry_path)
    if len(telemetry) != len(raw_lines):
        problems.append("malformed telemetry JSONL")
    # Existing trainer action collapse halts.  Its absence is only valid when the
    # completed summary attests a clean finish; boost/nonfinite checks are audited
    # here because T23's launcher predates their online sentinels.
    if not telemetry:
        problems.append("missing or unreadable log.jsonl offline tripwire telemetry")
    required_windows = {
        "transitions",
        "td",
        "anchor",
        "qmax",
        "greedy_boost_rate",
        "greedy_boost_rate_lt_700",
    }
    if telemetry and any(not required_windows.issubset(row) for row in telemetry):
        problems.append("telemetry omits required boost-window fields")
    for row in telemetry:
        for key in ("loss", "td", "anchor", "qmax", "grad_norm"):
            if key in row and (
                not isinstance(row[key], (int, float)) or not math.isfinite(float(row[key]))
            ):
                tripwires.append(f"nonfinite telemetry {key}")
        for key, limit in (("greedy_boost_rate", 0.02), ("greedy_boost_rate_lt_700", 0.01)):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(float(row[key])):
                problems.append(f"telemetry {key} is not finite numeric")
            elif not 0.0 <= float(row[key]) <= 1.0:
                problems.append(f"telemetry {key} outside [0,1]")
            elif float(row[key]) > limit:
                tripwires.append(f"tripwire {key}>{limit}")
    positions = [
        int(row["transitions"]) for row in telemetry if str(row.get("transitions", "")).isdigit()
    ]
    if (
        not positions
        or positions[0] > 200_000
        or positions[-1] < 8_000_000
        or any(b <= a or b - a > 200_000 for a, b in zip(positions, positions[1:]))
    ):
        problems.append("telemetry lacks complete monotonic <=200k windows through 8M")
    if not manifest_path.is_file():
        problems.append("missing checkpoint manifest.jsonl")
    return {
        "seed": seed,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
        "run_log_path": str(run_log),
        "run_log_sha256": sha256_file(run_log) if run_log.is_file() else None,
        "tripwire_triggered": bool(tripwires),
        "tripwire_reasons": tripwires,
        "clean": not problems,
        "problems": problems,
        "telemetry_path": str(telemetry_path),
        "telemetry_sha256": sha256_file(telemetry_path) if telemetry_path.is_file() else None,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
    }


def checkpoint_metadata(path: Path, *, required_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"checkpoint missing: {path}")
    digest = sha256_file(path)
    if required_sha256 is not None and digest != required_sha256:
        raise ValueError(f"checkpoint SHA mismatch for {path}: {digest} != {required_sha256}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def checkpoint_training_metadata(
    path: Path, provenance: Mapping[str, Any], seed: int, minimum: int
) -> dict[str, Any]:
    """Bind checkpoint bytes, manifest record, and embedded seed/transition facts."""
    info = checkpoint_metadata(path)
    matches = []
    for row in _load_json_lines(Path(str(provenance["manifest_path"]))):
        names = {str(v) for v in _find_values(row, {"file", "path", "name", "checkpoint"})}
        hashes = {str(v) for v in _find_values(row, {"sha256", "checkpoint_sha256"})}
        steps = [
            int(v)
            for v in _find_values(row, {"transitions", "actual_transitions"})
            if isinstance(v, (str, int, float)) and str(v).isdigit()
        ]
        if path.name in names and info["sha256"] in hashes and any(v >= minimum for v in steps):
            matches.append(row)
    if not matches:
        raise ValueError(f"checkpoint has no matching manifest provenance: {path.name}")
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint lacks metadata mapping")
    meta = state.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("checkpoint lacks meta provenance")
    if int(meta.get("seed", -1)) != seed or int(meta.get("transitions", -1)) < minimum:
        raise ValueError("checkpoint embedded seed/transition metadata mismatches T23 declaration")
    if (
        meta.get("init_sha256") != B8_SHA256
        or meta.get("m3c", {}).get("namespace") != "t23-r0-screen/v1"
    ):
        raise ValueError("checkpoint embedded init/namespace does not bind T23 R0")
    return {**info, "manifest_matches": len(matches), "minimum_transitions": minimum}


def _bank_worlds(bank: Mapping[str, Any], training_seed: int, phase: str) -> list[dict[str, Any]]:
    worlds_value = bank.get("worlds", [])
    if isinstance(worlds_value, Mapping):
        worlds_value = worlds_value.get(str(training_seed), worlds_value.get(training_seed, {}))
        if isinstance(worlds_value, Mapping):
            worlds_value = worlds_value.get(phase, [])
    worlds = worlds_value
    selected = [
        dict(world)
        for world in worlds
        if int(world.get("training_seed", training_seed)) == training_seed
        and str(world.get("bank_phase", phase)) == phase
    ]
    if not selected:
        raise ValueError(f"bank has no {phase!r} worlds for training seed {training_seed}")
    counts = {mix: sum(str(w.get("mix")) == mix for w in selected) for mix in MIX_COUNTS}
    expected = MIX_COUNTS if phase == "initial" else MIX_COUNTS
    if counts != expected:
        raise ValueError(
            f"world-bank counts for seed {training_seed}: {counts}, expected {expected}"
        )
    identities = [str(w.get("world_semantic_sha256", "")) for w in selected]
    if any(len(item) != 64 for item in identities) or len(set(identities)) != len(identities):
        raise ValueError("world bank has missing or duplicate world semantic hashes")
    return sorted(selected, key=lambda item: (str(item["mix"]), int(item["world_seed"])))


def _opponents(world: Mapping[str, Any], lookup: Mapping[str, Any]) -> list[tuple[str, str]]:
    slots = world.get("slots", world.get("opponent_slots", world.get("opponents")))
    if not isinstance(slots, list):
        raise ValueError("world needs slots/opponents list")
    out: list[tuple[str, str]] = []
    for slot in slots:
        if isinstance(slot, (list, tuple)) and len(slot) == 2:
            kind, ref = str(slot[0]), slot[1]
            if kind == "scripted":
                out.append(("scripted", str(ref)))
            elif kind == "checkpoint" and str(ref) in lookup:
                spec = lookup[str(ref)]
                out.append((str(spec[0]), str(spec[1])))
            else:
                raise ValueError(f"invalid bank opponent {slot!r}")
            continue
        if (
            not isinstance(slot, Mapping)
            or int(slot.get("slot", -1)) == 0
            or slot.get("kind") == "hero"
        ):
            continue
        kind = str(slot.get("kind", "checkpoint"))
        ref = slot.get("ref", slot.get("member_sha256", slot.get("sha256")))
        if kind == "scripted":
            out.append(("scripted", str(ref)))
        elif kind in {"checkpoint", "member"}:
            if str(ref) not in lookup:
                raise ValueError(f"bank opponent SHA absent from pinned lookup: {ref}")
            spec = lookup[str(ref)]
            out.append((str(spec[0]), str(spec[1])))
        else:
            raise ValueError(f"unknown opponent kind {kind!r}")
    if not out:
        raise ValueError("world has no fixed opponents")
    return out


class SurvivalObserver:
    """Detached accounting diagnostic: alive frames and one terminal death only."""

    def __init__(self, seeds: Sequence[int]) -> None:
        self.alive_frames = {int(seed): 0 for seed in seeds}
        self.death_frame: dict[int, int | None] = dict.fromkeys(self.alive_frames)

    def __call__(self, snapshot: Mapping[str, Any]) -> None:
        for index, seed in enumerate(snapshot["seeds"]):
            post, pre = snapshot["post"], snapshot["pre"]
            number = int(seed)
            self.alive_frames[number] += int(post["alive"][index])
            if pre["alive"][index] and not post["alive"][index]:
                if self.death_frame[number] is not None:
                    raise RuntimeError("terminal hero died twice")
                self.death_frame[number] = int(snapshot["frame"]) + 1


def evaluate_world(
    world: Mapping[str, Any],
    hero_checkpoint: Path,
    hero_carrier: tuple[str, str],
    context: Mapping[str, Any],
    *,
    observer_factory: Callable[[Sequence[int]], SurvivalObserver] = SurvivalObserver,
    vector61_trace: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one exact bank world.  Kept public for Mac/Colab parity tooling."""
    from src.simd_env.eval_engine import run_simd_eval

    seed = int(world["world_seed"])
    roster = _opponents(world, context["lookup"])
    observer = observer_factory([seed])
    records = run_simd_eval(
        hero_carrier,
        roster,
        HORIZON,
        [seed],
        profile=context["profile"],
        opponent_specs_by_world={seed: roster},
        mix_id=str(world["mix"]),
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        hero_ego2s=str(hero_checkpoint),
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
        vector61_backend="numba",
        scripted_anchors="fast",
        frame_observer=observer,
        vector61_trace=vector61_trace,
    )
    if len(records) != 1:
        raise RuntimeError("one-world evaluator received unexpected record count")
    record = dict(records[0])
    denom = record.get("denominators", {})
    alive = int(denom.get("alive_frames", -1))
    if alive != observer.alive_frames[seed] or int(denom.get("scored_frames", -1)) != HORIZON:
        raise RuntimeError("H5000 accumulator/observer accounting mismatch")
    if int(record.get("deaths", -1)) != int(observer.death_frame[seed] is not None):
        raise RuntimeError("terminal hero death accounting mismatch")
    return {
        "schema": "t23-eval-row/v1",
        "world_seed": seed,
        "world_id": str(world.get("world_id", world["world_semantic_sha256"])),
        "mix": str(world["mix"]),
        "world_semantic_sha256": str(world["world_semantic_sha256"]),
        "roster_semantic_sha256": str(world["roster_semantic_sha256"]),
        "mass_integral": float(record["mass_integral"]),
        "alive_frame_fraction": float(record["survival_fraction"]),
        "no_death_episode_survival": bool(int(record["deaths"]) == 0),
        "no_death_episode": bool(int(record["deaths"]) == 0),
        "deaths": int(record["deaths"]),
        "death_frame": observer.death_frame[seed],
        "horizon": HORIZON,
        "frames_stepped": HORIZON,
        "record": record,
    }


def _context(workers: int) -> dict[str, Any]:
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    return gid._context(max(1, workers))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shard_worlds(
    worlds: Sequence[dict[str, Any]], shard: int, shards: int
) -> list[dict[str, Any]]:
    if not 0 <= shard < shards:
        raise ValueError("--shard must be in [0, --shards)")
    return [world for index, world in enumerate(worlds) if index % shards == shard]


def _result_path(
    root: Path, training_seed: int, checkpoint_label: str, phase: str, shard: int, shards: int
) -> Path:
    return (
        root
        / "evaluation"
        / f"seed-{training_seed}"
        / phase
        / checkpoint_label
        / f"shard-{shard}-of-{shards}.json"
    )


def valid_completed_shard(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        row = _json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    rows = row.get("rows")
    return (
        row.get("schema") == SCHEMA
        and row.get("status") == "completed"
        and row.get("clean_exit") is True
        and row.get("identity") == dict(expected)
        and isinstance(rows, list)
        and len(rows) == 2 * int(expected["world_count"])
        and row.get("rows_sha256") == hashlib.sha256(canonical(rows).encode()).hexdigest()
        and all(item.get("schema") == "t23-eval-row/v1" for item in rows)
    )


def _run_checkpoint(
    *,
    root: Path,
    bank: Mapping[str, Any],
    context: Mapping[str, Any],
    training_seed: int,
    checkpoint: Path,
    checkpoint_label: str,
    b8: Path,
    phase: str,
    shard: int,
    shards: int,
    provenance: Mapping[str, Any],
) -> Path:
    worlds = _shard_worlds(_bank_worlds(bank, training_seed, phase), shard, shards)
    candidate_meta = checkpoint_training_metadata(
        checkpoint, provenance, training_seed, int(checkpoint_label.removesuffix("k")) * 1000
    )
    b8_meta = checkpoint_metadata(b8, required_sha256=B8_SHA256)
    path = _result_path(root, training_seed, checkpoint_label, phase, shard, shards)
    identity = {
        "training_seed": training_seed,
        "checkpoint_label": checkpoint_label,
        "checkpoint_sha256": candidate_meta["sha256"],
        "b8_sha256": b8_meta["sha256"],
        "phase": phase,
        "shard": shard,
        "shards": shards,
        "world_count": len(worlds),
        "world_hashes": [world["world_semantic_sha256"] for world in worlds],
        "bank_sha256": hashlib.sha256(canonical(bank).encode()).hexdigest(),
        "evaluator_source_sha256": sha256_file(Path(__file__)),
        "profile_digest": str(context["profile"].digest),
        "training_summary_sha256": provenance.get("summary_sha256"),
        "training_manifest_sha256": provenance.get("manifest_sha256"),
        "training_telemetry_sha256": provenance.get("telemetry_sha256"),
        "training_run_log_sha256": provenance.get("run_log_sha256"),
        "training_tripwire_triggered": provenance.get("tripwire_triggered"),
        "evaluation_dependencies_sha256": hashlib.sha256(
            canonical(source_manifest()).encode()
        ).hexdigest(),
    }
    if valid_completed_shard(path, identity):
        return path
    if path.exists():
        try:
            previous = _json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            previous = {}
        if previous.get("status") == "completed":
            raise RuntimeError(
                f"refusing to replace completed receipt with mismatched identity: {path}"
            )
    if not provenance["clean"]:
        _atomic_json(
            path,
            {
                "schema": SCHEMA,
                "status": "invalid",
                "clean_exit": False,
                "identity": identity,
                "training_provenance": provenance,
                "tripwire_evidence": {
                    "passed": False,
                    "triggered": any("tripwire" in p for p in provenance["problems"]),
                    "problems": provenance["problems"],
                },
                "rows": [],
            },
        )
        return path
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    # B8 is an ego2s/raster checkpoint, not a Vector61 carrier.  Its Q values
    # still drive the hero through hero_ego2s; FRP3 only carries the runtime.
    hero_carrier = context["lookup"][gid.FRP3_S12[1]]
    rows: list[dict[str, Any]] = []
    profile_digest = str(context["profile"].digest)
    source_hash = sha256_file(Path(__file__))
    for arm, arm_checkpoint, arm_meta in (
        ("candidate", checkpoint, candidate_meta),
        ("b8", b8, b8_meta),
    ):
        for world in worlds:
            value = evaluate_world(world, arm_checkpoint, hero_carrier, context)
            value.update(
                {
                    "arm": arm,
                    "training_seed": training_seed,
                    "checkpoint": arm_meta,
                    "checkpoint_step": int(checkpoint_label.removesuffix("k")) * 1000,
                    "clean_exit": True,
                    "candidate_checkpoint_sha256": arm_meta["sha256"],
                    "b8_checkpoint_sha256": b8_meta["sha256"],
                    "world_bank_sha256": identity["bank_sha256"],
                    "world_profile_sha256": profile_digest,
                    "initial_world_sha256": str(world.get("initial_world_sha256", "")),
                    "evaluator_source_sha256": source_hash,
                    "opponent_specs_sha256": str(world["roster_semantic_sha256"]),
                }
            )
            rows.append(value)
    _atomic_json(
        path,
        {
            "schema": SCHEMA,
            "status": "completed",
            "clean_exit": True,
            "training_seed": training_seed,
            "shard": {"index": shard, "count": shards},
            "identity": identity,
            "rows": rows,
            "rows_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "training_provenance": provenance,
            "tripwire_evidence": {
                "passed": not bool(provenance["tripwire_triggered"]),
                "triggered": bool(provenance["tripwire_triggered"]),
                "problems": list(provenance["problems"]),
            },
            "engine": {
                "cpu_only": True,
                "sim_engine": "grid",
                "hero": "ego2s",
                "ego2s_forward": "rowwise",
                "ego2s_mask": "advisory",
                "veto": False,
                "horizon": HORIZON,
            },
            "bank_sha256": hashlib.sha256(canonical(bank).encode()).hexdigest(),
            "profile_sha256": profile_digest,
            "git_commit": _git_commit(),
            "platform": platform.platform(),
        },
    )
    return path


def _run_job(payload: Mapping[str, Any]) -> str:
    """Process-pool entrypoint.  Each job owns both paired arms for its worlds."""
    context = _context(1)
    return str(
        _run_checkpoint(
            root=Path(str(payload["root"])),
            bank=payload["bank"],
            context=context,
            training_seed=int(payload["training_seed"]),
            checkpoint=Path(str(payload["checkpoint"])),
            checkpoint_label=str(payload["checkpoint_label"]),
            b8=Path(str(payload["b8"])),
            phase=str(payload["phase"]),
            shard=int(payload["shard"]),
            shards=int(payload["shards"]),
            provenance=payload["provenance"],
        )
    )


def _authorize_extension(
    root: Path, verdict_path: Path, jobs: Sequence[tuple[int, Path, str]]
) -> None:
    """Freeze exact initial candidate bytes before any one-time extension work."""
    verdict = _json(verdict_path)
    frozen = verdict.get("frozen_checkpoint_sha256s")
    if (
        verdict.get("verdict") != "EXTEND"
        or verdict.get("extension_round") != 0
        or not isinstance(frozen, Mapping)
    ):
        raise ValueError("extension verdict lacks the registered round-0 freeze contract")
    for seed, checkpoint, label in jobs:
        key = f"{seed}:{int(label.removesuffix('k')) * 1000}"
        if frozen.get(key) != sha256_file(checkpoint):
            raise ValueError(f"extension checkpoint differs from frozen initial identity: {key}")
    payload = {
        "prior_verdict": "EXTEND",
        "prior_result": verdict,
        "prior_result_sha256": hashlib.sha256(canonical(verdict).encode()).hexdigest(),
        "frozen_checkpoint_sha256s": dict(frozen),
    }
    path = root / "evaluation" / "extension_authorization.json"
    if path.exists() and _json(path) != payload:
        raise ValueError("existing extension authorization differs from requested frozen verdict")
    if not path.exists():
        _atomic_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=COLAB_ROOT if Path("/content").exists() else MAC_ROOT
    )
    parser.add_argument("--bank", type=Path, default=Path(__file__).with_name("WORLD_BANK.json"))
    parser.add_argument(
        "--b8-checkpoint", type=Path, default=os.environ.get("SNAKE_T23_B8_CHECKPOINT")
    )
    parser.add_argument("--shard", required=True, help="zero-based i/N (for example 0/3)")
    parser.add_argument("--shards", type=int, help="legacy total; omit when --shard is i/N")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phase", choices=("initial", "extension"), default="initial")
    parser.add_argument(
        "--prior-verdict", type=Path, help="required frozen initial verdict for extension"
    )
    parser.add_argument("--training-seeds", default=",".join(map(str, TRAINING_SEEDS)))
    parser.add_argument(
        "--include-descriptive", action="store_true", help="also evaluate 2M and 4M"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.b8_checkpoint is None:
        raise SystemExit("--b8-checkpoint or SNAKE_T23_B8_CHECKPOINT is required")
    if args.phase == "extension":
        if args.prior_verdict is None or not args.prior_verdict.is_file():
            raise SystemExit(
                "extension requires --prior-verdict from the completed initial analysis"
            )
        verdict = _json(args.prior_verdict)
        if str(verdict.get("verdict", "")).upper() != "EXTEND":
            raise SystemExit("extension requires an initial EXTEND verdict")
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if "/" in str(args.shard):
        index, total = str(args.shard).split("/", 1)
        args.shard, parsed_total = int(index), int(total)
        if args.shards is not None and args.shards != parsed_total:
            raise SystemExit("--shards conflicts with --shard i/N")
        args.shards = parsed_total
    else:
        args.shard = int(args.shard)
    if args.shards is None:
        raise SystemExit("--shard must be i/N (or supply legacy --shards)")
    from research.t23_r0_screen_20261017 import bank as bank_module

    bank = bank_module.load_world_bank(args.bank)
    _validate_bank(args.bank, bank, str(_context(1)["profile"].digest))
    seeds = tuple(int(item) for item in args.training_seeds.split(",") if item)
    if any(seed not in TRAINING_SEEDS for seed in seeds):
        raise SystemExit("T23 accepts only preregistered training seeds 7101-7105")
    jobs: list[tuple[int, Path, str]] = []
    for seed in seeds:
        jobs.append((seed, args.root / f"seed-{seed}" / "ckpt_8000k.pth", "8000k"))
        if args.include_descriptive:
            jobs.extend(
                (seed, args.root / f"seed-{seed}" / f"ckpt_{step}k.pth", f"{step}k")
                for step in (2000, 4000)
            )
    provenance = {seed: training_provenance(args.root, seed) for seed in seeds}
    if args.phase == "extension":
        assert args.prior_verdict is not None
        _authorize_extension(args.root, args.prior_verdict, jobs)
    payloads = [
        {
            "root": str(args.root),
            "bank": bank,
            "training_seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_label": label,
            "b8": str(args.b8_checkpoint),
            "phase": args.phase,
            "shard": args.shard,
            "shards": args.shards,
            "provenance": provenance[seed],
        }
        for seed, checkpoint, label in jobs
    ]
    if args.workers == 1:
        outputs = [Path(_run_job(payload)) for payload in payloads]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            outputs = [Path(item) for item in pool.map(_run_job, payloads)]
    print(canonical({"schema": SCHEMA, "outputs": [str(path) for path in outputs]}))
    return 0 if all(_json(path).get("clean_exit") is True for path in outputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
