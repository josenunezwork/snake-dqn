"""Byte-bound evidence contract for the strict promotion decision.

This module is deliberately an artifact consumer.  It does not execute worlds,
construct policies, or accept caller-computed statistics.  It opens checkpoint
metadata only to validate the candidate's serving and training lineage.  A
request is frozen before final worlds, a raw artifact is bound after those
worlds, and the final receipt is derived by reopening every referenced byte and
recomputing the decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from src.core.runtime_contract import (
    EffectiveWorldConfig,
    PQN_TRAIN_RESET_STRATEGIES,
    RuntimeModeContract,
    canonical_digest,
)
from src.core.seeding import derive_seed
from src.evaluation.anchors import SCRIPTED_ANCHOR_KINDS, SCRIPTED_ANCHOR_VERSION
from src.evaluation.artifacts import EVALUATOR_ARTIFACT_VERSION
from src.evaluation.protocol import (
    PROMOTION_V2_EVALUATOR,
    PROMOTION_V2_WATCH_RECT,
    EvaluationProfile,
)
from src.model.obs_spec import RASTER31V3_CONTRACT
from src.scripts.eval_stats import (
    PAIRED_DELTA_PILOT_FAMILY_ALPHA,
    PAIRED_DELTA_PILOT_METHOD,
    PAIRED_DELTA_PILOT_POWER,
    paired_delta_pilot_size,
    strict_promotion_decision,
)
from web.backend.session import (
    V3_ACTION_MASK_CONTRACT,
    _deployment_target_manifest,
    _validate_v3_serving_checkpoint,
)

STRICT_REQUEST_VERSION = "strict-promotion-request/v2"
STRICT_RAW_WORLD_VERSION = "strict-raw-worlds/v1"
STRICT_PILOT_VERSION = "strict-paired-pilot/v1"
STRICT_CALIBRATION_VERSION = "strict-calibration/v1"
STRICT_SERVING_BUNDLE_VERSION = "strict-serving-bundle/v1"
STRICT_SERVING_EPISODE_VERSION = "strict-serving-episode/v3"
STRICT_FINAL_RECEIPT_VERSION = "strict-promotion-final-receipt/v2"
STRICT_MIXES = ("frozen", "scripted", "mixed")
SEED_NAMESPACES = (
    "training",
    "development",
    "shakedown",
    "pilot",
    "final",
    "serving",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_ACCEPTED_V3_ACTION_MASK_CONTRACT = MappingProxyType(dict(V3_ACTION_MASK_CONTRACT))
_REQUIRED_SOURCE_CLOSURE = frozenset(
    {
        "src/scripts/tournament_eval.py",
        "src/scripts/eval_cli.py",
        "src/scripts/eval_stats.py",
        "src/core/config_loader.py",
        "src/core/game_config.py",
        "src/core/runtime_contract.py",
        "src/core/mechanics_constants.py",
        "src/core/reward_events.py",
        "src/evaluation/strict_promotion.py",
        "src/evaluation/artifacts.py",
        "src/evaluation/protocol.py",
        "src/evaluation/metrics.py",
        "src/evaluation/anchors.py",
        "src/evaluation/serving_episode.py",
        "src/core/seeding.py",
        "src/game/game_state_factory.py",
        "src/game/game_state.py",
        "src/game/game_logic.py",
        "src/game/food_manager.py",
        "src/game/ai_snake.py",
        "src/game/snake.py",
        "src/game/scripted_snake.py",
        "src/game/snake_factory.py",
        "src/game/snake_reward.py",
        "src/game/snake_state.py",
        "src/training/behavior_probes.py",
        "src/training/action_mask.py",
        "src/training/apex_policy.py",
        "src/training/base_dqn_policy.py",
        "src/training/checkpoint_contract.py",
        "src/training/pqn_lifecycle.py",
        "src/training/td_targets.py",
        "src/model/apex_network.py",
        "src/model/checkpoint_manager.py",
        "src/model/inference_agent.py",
        "src/model/obs_spec.py",
        "src/model/raster_network.py",
        "src/simd_env/eval_engine.py",
        "src/simd_env/live_adapter.py",
        "src/simd_env/batch_sim.py",
        "src/simd_env/featurizer.py",
        "web/backend/raster_policy.py",
        "web/backend/session.py",
        "web/backend/serialize.py",
        "web/backend/checkpoints.py",
        "web/backend/state_labels.py",
        "src/scripts/serving_spot_check.py",
    }
)


class StrictPromotionArtifactError(ValueError):
    """An artifact cannot participate in strict promotion authority."""


@dataclass(frozen=True)
class _OpenedJSON:
    value: Mapping[str, Any]
    byte_sha256: str
    path: str


@dataclass(frozen=True)
class StrictRequestToken:
    """Opaque, recursively frozen identity for a validated pre-world request."""

    request_path: str
    request_byte_sha256: str
    request_semantic_digest: str
    request: Mapping[str, Any]
    required_final_worlds: int
    artifact_byte_sha256: Mapping[str, str]
    checkpoint_role_sha256: Mapping[str, Any]


@dataclass(frozen=True)
class StrictRawArtifactToken:
    """Byte identity for a validated post-world raw artifact."""

    raw_world_artifact_path: str
    raw_world_artifact_byte_sha256: str
    request_semantic_digest: str


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrictPromotionArtifactError(f"value is not finite canonical JSON: {exc}") from exc


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in items:
        if key in output:
            raise StrictPromotionArtifactError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _nonfinite(value: str) -> None:
    raise StrictPromotionArtifactError(f"non-finite JSON constant {value}")


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _open_json(path: str | Path, label: str) -> _OpenedJSON:
    """Read one regular file consistently and reject replacement during the read."""
    source = Path(path).expanduser().resolve(strict=True)
    try:
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StrictPromotionArtifactError(f"{label} is not a regular file")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        path_after = source.stat()
    except OSError as exc:
        raise StrictPromotionArtifactError(f"cannot read {label}: {exc}") from exc
    if not _same_file(before, after) or not _same_file(after, path_after):
        raise StrictPromotionArtifactError(f"{label} changed while it was opened")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictPromotionArtifactError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise StrictPromotionArtifactError(f"{label} must contain a JSON object")
    return _OpenedJSON(value, hashlib.sha256(payload).hexdigest(), str(source))


def _hash_path(path: str | Path, label: str) -> str:
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StrictPromotionArtifactError(f"{label} is not a regular file")
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        path_after = source.stat()
    except OSError as exc:
        raise StrictPromotionArtifactError(f"cannot read {label}: {exc}") from exc
    if not _same_file(before, after) or not _same_file(after, path_after):
        raise StrictPromotionArtifactError(f"{label} changed while it was opened")
    return digest.hexdigest()


def _read_path_bytes(path: str | Path, label: str) -> tuple[bytes, str, str]:
    """Read one regular file once and retain the bytes used for deserialization."""
    source = Path(path).expanduser().resolve(strict=True)
    try:
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StrictPromotionArtifactError(f"{label} is not a regular file")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        path_after = source.stat()
    except OSError as exc:
        raise StrictPromotionArtifactError(f"cannot read {label}: {exc}") from exc
    if not _same_file(before, after) or not _same_file(after, path_after):
        raise StrictPromotionArtifactError(f"{label} changed while it was opened")
    return payload, hashlib.sha256(payload).hexdigest(), str(source)


def _exact(value: Any, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    expected = set(keys)
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise StrictPromotionArtifactError(
            f"{label} has missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StrictPromotionArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictPromotionArtifactError(f"{label} must be a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise StrictPromotionArtifactError(f"{label} must be a positive finite number")
    return parsed


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictPromotionArtifactError(f"{label} must be finite numeric data")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise StrictPromotionArtifactError(f"{label} must be finite numeric data")
    return parsed


def _uint32(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise StrictPromotionArtifactError(f"{label} must be an unsigned 32-bit integer")
    return value


def _checkpoint_agent(value: Any, label: str) -> Mapping[str, Any]:
    agent = _exact(value, {"kind", "sha256"}, label)
    if agent["kind"] != "checkpoint":
        raise StrictPromotionArtifactError(f"{label} must be a checkpoint identity")
    _sha(agent["sha256"], f"{label}.sha256")
    return agent


def scripted_agent(name: str) -> dict[str, str]:
    """Return the canonical immutable identity for one scripted anchor."""
    if name not in SCRIPTED_ANCHOR_KINDS:
        raise StrictPromotionArtifactError(f"unknown scripted anchor {name!r}")
    digest = canonical_digest({"scripted_anchor": name, "anchor_version": SCRIPTED_ANCHOR_VERSION})
    return {
        "kind": "scripted",
        "name": name,
        "version": SCRIPTED_ANCHOR_VERSION,
        "sha256": digest,
    }


def _scripted_agent(value: Any, label: str) -> Mapping[str, Any]:
    agent = _exact(value, {"kind", "name", "version", "sha256"}, label)
    if agent != scripted_agent(agent.get("name")):
        raise StrictPromotionArtifactError(f"{label} is not a canonical scripted anchor")
    return agent


def _profile(value: Any) -> tuple[Mapping[str, Any], EvaluationProfile]:
    raw = _exact(value, {"descriptor", "digest"}, "profile")
    try:
        profile = EvaluationProfile.from_descriptor(raw["descriptor"])
    except (TypeError, ValueError) as exc:
        raise StrictPromotionArtifactError(f"invalid evaluation profile: {exc}") from exc
    if profile.digest != _sha(raw["digest"], "profile.digest"):
        raise StrictPromotionArtifactError("profile digest does not match its descriptor")
    if (
        profile.name != PROMOTION_V2_WATCH_RECT
        or profile.evaluator_version != PROMOTION_V2_EVALUATOR
        or profile.scored_horizon != 5000
        or profile.observation_progress_horizon != 5000
        or profile.learn
        or profile.legacy_diagnostic
    ):
        raise StrictPromotionArtifactError("strict authority requires the fixed promotion profile")
    return raw, profile


def _source_closure(value: Any) -> Mapping[str, Any]:
    raw = _exact(
        value,
        {"evaluator_sha256", "source_manifest", "closure_sha256", "git_commit"},
        "evaluator_source",
    )
    _sha(raw["evaluator_sha256"], "evaluator_source.evaluator_sha256")
    if not isinstance(raw["git_commit"], str) or _GIT_SHA.fullmatch(raw["git_commit"]) is None:
        raise StrictPromotionArtifactError("evaluator source needs an exact Git commit")
    manifest = raw["source_manifest"]
    if not isinstance(manifest, Mapping) or not manifest:
        raise StrictPromotionArtifactError("evaluator source manifest must be non-empty")
    for source_path, digest in manifest.items():
        if not isinstance(source_path, str) or not source_path:
            raise StrictPromotionArtifactError("evaluator source paths must be non-empty")
        pure = PurePosixPath(source_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise StrictPromotionArtifactError("evaluator source paths must stay inside the repo")
        _sha(digest, f"evaluator source {source_path}")
    if not _REQUIRED_SOURCE_CLOSURE <= set(manifest):
        missing = sorted(_REQUIRED_SOURCE_CLOSURE - set(manifest))
        raise StrictPromotionArtifactError(f"evaluator source closure is incomplete: {missing!r}")
    closure = canonical_digest(dict(manifest))
    if _sha(raw["closure_sha256"], "evaluator_source.closure_sha256") != closure:
        raise StrictPromotionArtifactError("source closure digest does not match its manifest")
    return raw


def _seed_namespaces(value: Any) -> Mapping[str, Any]:
    raw = _exact(value, SEED_NAMESPACES, "seed_namespaces")
    seen: set[int] = set()
    for name in SEED_NAMESPACES:
        seeds = raw[name]
        if not isinstance(seeds, list) or not seeds:
            raise StrictPromotionArtifactError(f"seed namespace {name!r} must be non-empty")
        parsed = [_uint32(seed, f"{name} seed") for seed in seeds]
        if len(set(parsed)) != len(parsed) or seen.intersection(parsed):
            raise StrictPromotionArtifactError("seed namespaces must be unique and disjoint")
        seen.update(parsed)
    if len(raw["pilot"]) < 2:
        raise StrictPromotionArtifactError("pilot requires at least two unique worlds")
    if len(raw["serving"]) != 50:
        raise StrictPromotionArtifactError("serving namespace requires exactly 50 seeds")
    return raw


def _bands(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise StrictPromotionArtifactError("behavioral bands must be non-empty")
    bands: list[Mapping[str, Any]] = []
    names: set[str] = set()
    for entry in value:
        band = _exact(
            entry,
            {
                "name",
                "metric",
                "subject",
                "mix_scope",
                "reducer",
                "denominator",
                "lower",
                "upper",
                "bound_source_sha256",
                "calibration_method",
                "lower_offset",
                "upper_offset",
            },
            "behavioral band",
        )
        if (
            not isinstance(band["name"], str)
            or not band["name"]
            or band["name"] in names
            or not isinstance(band["metric"], str)
            or not band["metric"]
        ):
            raise StrictPromotionArtifactError("behavioral band names/metrics must be unique")
        if (
            band["subject"] != "candidate"
            or band["reducer"] != "mean"
            or band["denominator"] != "candidate_final_world_records"
            or band["calibration_method"] != "reference-mean-plus-offsets/v1"
            or not isinstance(band["mix_scope"], list)
            or not band["mix_scope"]
            or len(band["mix_scope"]) != len(set(band["mix_scope"]))
            or not set(band["mix_scope"]) <= set(STRICT_MIXES)
        ):
            raise StrictPromotionArtifactError("behavioral band execution contract is invalid")
        _sha(band["bound_source_sha256"], "behavioral band bound source")
        lower_offset = _finite(band["lower_offset"], "behavioral band lower offset")
        upper_offset = _finite(band["upper_offset"], "behavioral band upper offset")
        if lower_offset > upper_offset:
            raise StrictPromotionArtifactError("behavioral band offsets are reversed")
        lower = _finite(band["lower"], f"band {band['name']} lower")
        upper = _finite(band["upper"], f"band {band['name']} upper")
        if lower > upper:
            raise StrictPromotionArtifactError("behavioral band lower bound exceeds upper bound")
        names.add(band["name"])
        bands.append(band)
    return bands


def _semantic_contract(
    value: Any, label: str, expected_descriptor: Mapping[str, Any]
) -> Mapping[str, Any]:
    raw = _exact(value, {"descriptor", "digest"}, label)
    if not isinstance(raw["descriptor"], Mapping) or not raw["descriptor"]:
        raise StrictPromotionArtifactError(f"{label} descriptor must be non-empty")
    # Contract constants can contain tuples, while their JSON representation
    # necessarily contains arrays. Compare their canonical JSON forms so the
    # accepted typed constant remains the authority without rejecting its
    # lossless on-disk representation.
    expected_json = json.loads(_canonical_bytes(dict(expected_descriptor)))
    if raw["descriptor"] != expected_json:
        raise StrictPromotionArtifactError(f"{label} is not the accepted typed contract")
    if canonical_digest(dict(raw["descriptor"])) != _sha(raw["digest"], f"{label}.digest"):
        raise StrictPromotionArtifactError(f"{label} digest does not match its descriptor")
    return raw


def _serving_contract(value: Any, request: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _exact(
        value,
        {
            "observation",
            "action",
            "deployment_profile_digest",
            "source_closure_sha256",
            "plan",
            "candidate_contracts",
        },
        "serving contract",
    )
    _semantic_contract(
        raw["observation"],
        "serving observation contract",
        RASTER31V3_CONTRACT.semantic_dict(),
    )
    _semantic_contract(raw["action"], "serving action contract", _ACCEPTED_V3_ACTION_MASK_CONTRACT)
    if (
        raw["deployment_profile_digest"] != request["profile"]["digest"]
        or raw["source_closure_sha256"] != request["evaluator_source"]["closure_sha256"]
    ):
        raise StrictPromotionArtifactError("serving contract differs from deployment/source")
    plan = _exact(
        raw["plan"],
        {
            "schema_version",
            "frame_limit",
            "watch_completion",
            "play_completion",
            "episodes",
        },
        "serving plan",
    )
    if (
        plan["schema_version"] != "strict-serving-plan/v1"
        or plan["frame_limit"] != request["profile"]["descriptor"]["scored_horizon"]
        or plan["watch_completion"] != "external-horizon"
        or plan["play_completion"] != "human-terminal-or-external-horizon"
        or not isinstance(plan["episodes"], list)
        or len(plan["episodes"]) != 50
    ):
        raise StrictPromotionArtifactError(
            "serving readiness requires the frozen versioned episode plan"
        )
    plan_ids: set[str] = set()
    plan_seeds: set[int] = set()
    mode_counts = {"watch": 0, "play": 0}
    for value in plan["episodes"]:
        episode = _exact(value, {"episode_id", "serving_seed", "mode"}, "serving plan episode")
        episode_id = episode["episode_id"]
        seed = _uint32(episode["serving_seed"], "serving plan seed")
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or episode_id in plan_ids
            or seed in plan_seeds
            or episode["mode"] not in mode_counts
        ):
            raise StrictPromotionArtifactError("serving plan episodes are duplicate/invalid")
        plan_ids.add(episode_id)
        plan_seeds.add(seed)
        mode_counts[episode["mode"]] += 1
    if plan_seeds != set(request["seed_namespaces"]["serving"]) or mode_counts != {
        "watch": 1,
        "play": 49,
    }:
        raise StrictPromotionArtifactError(
            "serving plan must cover all seeds with one Watch and 49 Play episodes"
        )
    candidate_contracts = _exact(
        raw["candidate_contracts"],
        {
            "model_head_digest",
            "run_provenance_digest",
            "effective_world_digest",
            "source_runtime",
            "episode_lifecycle",
            "policy_source",
        },
        "candidate serving contracts",
    )
    for name in ("model_head_digest", "run_provenance_digest", "effective_world_digest"):
        _sha(candidate_contracts[name], f"candidate serving {name}")
    source_runtime = _exact(
        candidate_contracts["source_runtime"],
        {"descriptor", "digest"},
        "candidate source runtime",
    )
    episode_lifecycle = _exact(
        candidate_contracts["episode_lifecycle"],
        {"descriptor", "digest", "compatibility"},
        "candidate episode lifecycle",
    )
    policy_source = _exact(
        candidate_contracts["policy_source"],
        {"descriptor", "digest"},
        "candidate policy source",
    )
    for identity, label in (
        (episode_lifecycle, "candidate episode lifecycle"),
        (policy_source, "candidate policy source"),
    ):
        if not isinstance(identity["descriptor"], Mapping) or not identity["descriptor"]:
            raise StrictPromotionArtifactError(f"{label} descriptor must be non-empty")
        if canonical_digest(dict(identity["descriptor"])) != _sha(
            identity["digest"], f"{label} digest"
        ):
            raise StrictPromotionArtifactError(f"{label} digest is invalid")
    if episode_lifecycle["compatibility"] is not None and not isinstance(
        episode_lifecycle["compatibility"], Mapping
    ):
        raise StrictPromotionArtifactError(
            "candidate episode lifecycle compatibility must be an object or null"
        )
    descriptor = _exact(
        source_runtime["descriptor"],
        {
            "mode",
            "training",
            "respawn",
            "hero_terminal",
            "population_floor",
            "reset_strategy",
        },
        "candidate source runtime descriptor",
    )
    source_world = request["profile"]["descriptor"]["world"]
    expected_population_floor = (
        source_world["mechanics_version"] == 2 and source_world["num_snakes"] >= 3
    )
    expected_source_runtime = {
        "mode": "pqn_train",
        "training": True,
        "respawn": False,
        "hero_terminal": True,
        "population_floor": expected_population_floor,
    }
    if (
        not isinstance(descriptor["mode"], str)
        or not descriptor["mode"]
        or not isinstance(descriptor["reset_strategy"], str)
        or not descriptor["reset_strategy"]
        or any(
            not isinstance(descriptor[name], bool)
            for name in ("training", "respawn", "hero_terminal", "population_floor")
        )
        or canonical_digest(dict(descriptor))
        != _sha(source_runtime["digest"], "candidate source runtime digest")
        or candidate_contracts["effective_world_digest"]
        != request["profile"]["descriptor"]["world_digest"]
        or {name: descriptor[name] for name in expected_source_runtime}
        != expected_source_runtime
        or descriptor["reset_strategy"]
        not in PQN_TRAIN_RESET_STRATEGIES
    ):
        raise StrictPromotionArtifactError("candidate serving metadata is inconsistent")
    return raw


def _member_ids(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise StrictPromotionArtifactError(f"{label} must contain eligible members")
    members = [_sha(value, f"{label} member") for value in values]
    if len(members) != len(set(members)):
        raise StrictPromotionArtifactError(f"{label} contains duplicate members")
    return members


def materialize_rosters(
    *,
    final_world_seeds: Sequence[int],
    roster_width: int,
    checkpoint_pool: Sequence[Mapping[str, Any]],
    scripted_anchors: Sequence[Mapping[str, Any]],
    mixed_slot_rules: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize deterministic per-slot-balanced rosters before final worlds."""
    seeds = [_uint32(seed, "final world seed") for seed in final_world_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise StrictPromotionArtifactError("final world seeds must be non-empty and unique")
    if isinstance(roster_width, bool) or not isinstance(roster_width, int) or roster_width < 2:
        raise StrictPromotionArtifactError("strict roster width must be at least two")
    checkpoints = [
        _checkpoint_agent(dict(agent), "checkpoint pool member") for agent in checkpoint_pool
    ]
    scripts = [_scripted_agent(dict(agent), "scripted anchor") for agent in scripted_anchors]
    checkpoint_ids = [agent["sha256"] for agent in checkpoints]
    if not checkpoint_ids or len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise StrictPromotionArtifactError("checkpoint pool content must be non-empty and unique")
    scripted_ids = [agent["sha256"] for agent in scripts]
    greedy = scripted_agent("greedy_food")["sha256"]
    random_safe = scripted_agent("random_safe")["sha256"]
    if greedy not in scripted_ids or random_safe not in scripted_ids:
        raise StrictPromotionArtifactError(
            "strict rosters require canonical greedy_food and random_safe anchors"
        )
    if len(mixed_slot_rules) != roster_width:
        raise StrictPromotionArtifactError("mixed roster needs one explicit rule per slot")
    rules: dict[int, list[str]] = {}
    allowed = set(checkpoint_ids) | set(scripted_ids)
    for value in mixed_slot_rules:
        rule = _exact(value, {"slot", "eligible_member_sha256s"}, "mixed slot rule")
        slot = rule["slot"]
        if isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= roster_width:
            raise StrictPromotionArtifactError("mixed slot rule has an invalid slot")
        if slot in rules:
            raise StrictPromotionArtifactError("mixed slot rules contain a duplicate slot")
        eligible = _member_ids(rule["eligible_member_sha256s"], "mixed slot rule")
        if not set(eligible) <= allowed:
            raise StrictPromotionArtifactError("mixed slot rule names an undeclared member")
        rules[slot] = eligible
    if set(rules) != set(range(1, roster_width + 1)):
        raise StrictPromotionArtifactError("mixed slot rules do not cover the roster")
    canonical_mixed = {
        slot: (checkpoint_ids if slot % 2 else [random_safe]) for slot in range(1, roster_width + 1)
    }
    if rules != canonical_mixed:
        raise StrictPromotionArtifactError(
            "mixed roster rules must alternate checkpoint slots and canonical random_safe slots"
        )

    eligibility = {
        "frozen": {slot: checkpoint_ids for slot in range(1, roster_width + 1)},
        "scripted": {slot: [greedy] for slot in range(1, roster_width + 1)},
        "mixed": rules,
    }
    rows: list[dict[str, Any]] = []
    for mix in STRICT_MIXES:
        for world_index, seed in enumerate(seeds):
            slots = []
            mixed_checkpoint_ordinal = 0
            for slot in range(1, roster_width + 1):
                eligible = eligibility[mix][slot]
                if mix == "frozen":
                    phase = slot - 1
                elif mix == "mixed" and slot % 2:
                    # Phase checkpoint members by checkpoint-slot ordinal.
                    # Using the absolute slot number collapses all odd slots
                    # onto one member for a two-member pool.
                    phase = mixed_checkpoint_ordinal
                    mixed_checkpoint_ordinal += 1
                else:
                    phase = 0
                member = eligible[(world_index + phase) % len(eligible)]
                slots.append(
                    {
                        "slot": slot,
                        "eligible_member_sha256s": list(eligible),
                        "member_sha256": member,
                    }
                )
            rows.append({"mix": mix, "world_seed": seed, "slots": slots})
    return rows


def _roster_design(
    value: Any,
    *,
    world_roster_width: int,
    checkpoint_pool: Sequence[Mapping[str, Any]],
    scripted_anchors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    raw = _exact(value, {"mixes", "roster_width", "mixed_slot_rules"}, "roster_design")
    if raw["mixes"] != list(STRICT_MIXES) or raw["roster_width"] != world_roster_width:
        raise StrictPromotionArtifactError("roster design does not match the fixed mixes/world")
    # Materializer validates the detailed rules; a one-world call is enough here.
    materialize_rosters(
        final_world_seeds=[0],
        roster_width=world_roster_width,
        checkpoint_pool=checkpoint_pool,
        scripted_anchors=scripted_anchors,
        mixed_slot_rules=raw["mixed_slot_rules"],
    )
    return raw


def validate_materialized_rosters(
    rows: Any,
    *,
    final_world_seeds: Sequence[int],
    roster_design: Mapping[str, Any],
    checkpoint_pool: Sequence[Mapping[str, Any]],
    scripted_anchors: Sequence[Mapping[str, Any]],
) -> None:
    """Validate exact coverage and balance every eligible member in every slot."""
    if not isinstance(rows, list):
        raise StrictPromotionArtifactError("materialized rosters must be a list")
    expected = materialize_rosters(
        final_world_seeds=final_world_seeds,
        roster_width=roster_design["roster_width"],
        checkpoint_pool=checkpoint_pool,
        scripted_anchors=scripted_anchors,
        mixed_slot_rules=roster_design["mixed_slot_rules"],
    )
    if len(rows) != len(expected):
        raise StrictPromotionArtifactError("materialized rosters have incomplete cardinality")
    expected_keys = {(row["mix"], row["world_seed"]) for row in expected}
    actual: dict[tuple[str, int], Mapping[str, Any]] = {}
    counts: dict[tuple[str, int], dict[str, int]] = {}
    expected_eligibility: dict[tuple[str, int], list[str]] = {}
    expected_assignments: dict[tuple[str, int, int], str] = {}
    for template in expected:
        for slot in template["slots"]:
            key = (template["mix"], slot["slot"])
            expected_eligibility[key] = slot["eligible_member_sha256s"]
            counts[key] = {member: 0 for member in slot["eligible_member_sha256s"]}
            expected_assignments[(template["mix"], template["world_seed"], slot["slot"])] = slot[
                "member_sha256"
            ]
    for row_value in rows:
        row = _exact(row_value, {"mix", "world_seed", "slots"}, "materialized roster")
        key = (row["mix"], row["world_seed"])
        if key not in expected_keys or key in actual:
            raise StrictPromotionArtifactError("materialized roster key is duplicate/undeclared")
        if not isinstance(row["slots"], list) or len(row["slots"]) != roster_design["roster_width"]:
            raise StrictPromotionArtifactError("materialized roster width is incorrect")
        seen_slots: set[int] = set()
        for slot_value in row["slots"]:
            slot = _exact(
                slot_value,
                {"slot", "eligible_member_sha256s", "member_sha256"},
                "materialized roster slot",
            )
            slot_number = slot["slot"]
            eligibility = _member_ids(slot["eligible_member_sha256s"], "roster slot")
            eligibility_key = (row["mix"], slot_number)
            if (
                isinstance(slot_number, bool)
                or not isinstance(slot_number, int)
                or slot_number in seen_slots
                or eligibility_key not in expected_eligibility
                or eligibility != expected_eligibility[eligibility_key]
            ):
                raise StrictPromotionArtifactError("roster slot eligibility differs from design")
            member = _sha(slot["member_sha256"], "roster slot member")
            if member not in eligibility:
                raise StrictPromotionArtifactError("roster assignment is not eligible for its slot")
            if member != expected_assignments[(row["mix"], row["world_seed"], slot_number)]:
                raise StrictPromotionArtifactError(
                    "roster assignment differs from the canonical slot-phased rotation"
                )
            counts[eligibility_key][member] += 1
            seen_slots.add(slot_number)
        actual[key] = row
    if set(actual) != expected_keys:
        raise StrictPromotionArtifactError("materialized rosters do not cover every final world")
    for key, member_counts in counts.items():
        if max(member_counts.values()) - min(member_counts.values()) > 1:
            raise StrictPromotionArtifactError(f"roster assignments are unbalanced at {key!r}")


def _request(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], EvaluationProfile]:
    raw = _exact(
        value,
        {
            "schema_version",
            "engine",
            "candidate",
            "incumbent",
            "checkpoint_pool",
            "scripted_anchors",
            "evaluator_source",
            "profile",
            "seed_namespaces",
            "artifact_bindings",
            "calibration",
            "serving_contract",
            "roster_design",
            "materialized_rosters",
        },
        "strict promotion request",
    )
    if raw["schema_version"] != STRICT_REQUEST_VERSION:
        raise StrictPromotionArtifactError("unsupported strict request version")
    if raw["engine"] != "live":
        raise StrictPromotionArtifactError("strict promotion is limited to the live engine")
    candidate = _checkpoint_agent(raw["candidate"], "candidate")
    incumbent = _checkpoint_agent(raw["incumbent"], "incumbent")
    if candidate["sha256"] == incumbent["sha256"]:
        raise StrictPromotionArtifactError("candidate and incumbent checkpoint bytes must differ")
    if not isinstance(raw["checkpoint_pool"], list) or not raw["checkpoint_pool"]:
        raise StrictPromotionArtifactError("checkpoint_pool must be non-empty")
    pool = [_checkpoint_agent(agent, "checkpoint pool member") for agent in raw["checkpoint_pool"]]
    pool_ids = [agent["sha256"] for agent in pool]
    if len(pool_ids) != len(set(pool_ids)):
        raise StrictPromotionArtifactError("duplicate checkpoint pool content is forbidden")
    if not isinstance(raw["scripted_anchors"], list) or not raw["scripted_anchors"]:
        raise StrictPromotionArtifactError("scripted_anchors must be non-empty")
    scripts = [_scripted_agent(agent, "scripted anchor") for agent in raw["scripted_anchors"]]
    if len({agent["sha256"] for agent in scripts}) != len(scripts):
        raise StrictPromotionArtifactError("scripted anchor identities must be unique")
    source = _source_closure(raw["evaluator_source"])
    del source
    _, profile = _profile(raw["profile"])
    namespaces = _seed_namespaces(raw["seed_namespaces"])
    bindings = _exact(
        raw["artifact_bindings"], {"e0", "pilot", "calibration", "serving"}, "artifact bindings"
    )
    for name, digest in bindings.items():
        _sha(digest, f"artifact binding {name}")
    calibration = _exact(
        raw["calibration"],
        {"absolute_delta_ni", "mde_by_mix", "behavioral_bands"},
        "calibration declaration",
    )
    _positive(calibration["absolute_delta_ni"], "absolute_delta_ni")
    if not isinstance(calibration["mde_by_mix"], Mapping) or set(calibration["mde_by_mix"]) != set(
        STRICT_MIXES
    ):
        raise StrictPromotionArtifactError("calibration MDEs must name the three strict mixes")
    for mix in STRICT_MIXES:
        _positive(calibration["mde_by_mix"][mix], f"MDE for {mix}")
    _bands(calibration["behavioral_bands"])
    _serving_contract(raw["serving_contract"], raw)
    design = _roster_design(
        raw["roster_design"],
        world_roster_width=profile.world.num_snakes - 1,
        checkpoint_pool=pool,
        scripted_anchors=scripts,
    )
    validate_materialized_rosters(
        raw["materialized_rosters"],
        final_world_seeds=namespaces["final"],
        roster_design=design,
        checkpoint_pool=pool,
        scripted_anchors=scripts,
    )
    return raw, profile


def _validate_snapshot(snapshot: Mapping[str, Any], label: str, kind: str) -> str:
    raw = _exact(
        snapshot,
        {
            "artifact_kind",
            "source_path",
            "source_identity",
            "sha256",
            "size_bytes",
            "snapshot_path",
        },
        label,
    )
    if raw["artifact_kind"] != kind:
        raise StrictPromotionArtifactError(f"{label} has the wrong artifact kind")
    digest = _sha(raw["sha256"], f"{label}.sha256")
    if _hash_path(raw["snapshot_path"], f"{label} snapshot bytes") != digest:
        raise StrictPromotionArtifactError(f"{label} snapshot bytes changed")
    try:
        size = Path(raw["snapshot_path"]).resolve(strict=True).stat().st_size
    except OSError as exc:
        raise StrictPromotionArtifactError(f"cannot stat {label}: {exc}") from exc
    if isinstance(raw["size_bytes"], bool) or raw["size_bytes"] != size:
        raise StrictPromotionArtifactError(f"{label} size does not match snapshot bytes")
    identity = _exact(
        raw["source_identity"], {"device", "inode", "size_bytes", "mtime_ns"}, f"{label} identity"
    )
    if identity["size_bytes"] != raw["size_bytes"]:
        raise StrictPromotionArtifactError(f"{label} source identity size is inconsistent")
    return digest


def _validate_e0(
    opened: _OpenedJSON, request: Mapping[str, Any], profile: EvaluationProfile
) -> str:
    raw = _exact(
        opened.value,
        {
            "artifact_version",
            "checkpoint_digest_algorithm",
            "checkpoint_snapshots",
            "config",
            "evaluator",
            "source_agents",
            "evaluation_profile",
        },
        "E0 receipt",
    )
    if (
        raw["artifact_version"] != EVALUATOR_ARTIFACT_VERSION
        or raw["checkpoint_digest_algorithm"] != "sha256"
    ):
        raise StrictPromotionArtifactError("E0 receipt uses an unsupported artifact contract")
    if not isinstance(raw["checkpoint_snapshots"], list):
        raise StrictPromotionArtifactError("E0 checkpoint snapshots must be a list")
    digests = [
        _validate_snapshot(snapshot, "E0 checkpoint", "checkpoints")
        for snapshot in raw["checkpoint_snapshots"]
    ]
    required = {
        request["candidate"]["sha256"],
        request["incumbent"]["sha256"],
        *(agent["sha256"] for agent in request["checkpoint_pool"]),
    }
    if len(digests) != len(set(digests)) or set(digests) != required:
        raise StrictPromotionArtifactError(
            "E0 checkpoint bytes do not exactly match declared roles"
        )
    config = raw["config"]
    if not isinstance(config, Mapping) or "effective" not in config:
        raise StrictPromotionArtifactError("E0 config snapshot/effective config is incomplete")
    config_snapshot = dict(config)
    config_snapshot.pop("effective")
    _validate_snapshot(config_snapshot, "E0 config", "configs")
    e0_profile = _exact(raw["evaluation_profile"], {"descriptor", "digest"}, "E0 profile")
    if e0_profile != request["profile"] or e0_profile["digest"] != profile.digest:
        raise StrictPromotionArtifactError("E0 profile differs from the strict request")
    evaluator = _exact(
        raw["evaluator"], {"source_path", "sha256", "source_manifest", "git"}, "E0 evaluator"
    )
    declared = request["evaluator_source"]
    if (
        _hash_path(evaluator["source_path"], "E0 evaluator bytes") != declared["evaluator_sha256"]
        or evaluator["sha256"] != declared["evaluator_sha256"]
        or evaluator["source_manifest"] != declared["source_manifest"]
    ):
        raise StrictPromotionArtifactError("E0 evaluator identity differs from the strict request")
    git = _exact(evaluator["git"], {"commit", "dirty", "diff_sha256"}, "E0 evaluator git")
    if git != {"commit": declared["git_commit"], "dirty": False, "diff_sha256": _EMPTY_SHA256}:
        raise StrictPromotionArtifactError("strict evaluator source must be a clean exact commit")
    evaluator_path = Path(evaluator["source_path"]).resolve(strict=True)
    repository = evaluator_path.parents[2]
    project_python_closure = {
        str(path.relative_to(repository))
        for root in (repository / "src", repository / "web" / "backend")
        for path in root.rglob("*.py")
        if path.is_file()
    }
    if not project_python_closure <= set(declared["source_manifest"]):
        missing = sorted(project_python_closure - set(declared["source_manifest"]))
        raise StrictPromotionArtifactError(
            f"E0 omits project Python source from the conservative closure: {missing!r}"
        )
    for relative, digest in declared["source_manifest"].items():
        if _hash_path(repository / relative, f"evaluator source {relative}") != digest:
            raise StrictPromotionArtifactError(f"evaluator source changed: {relative}")
    if not isinstance(raw["source_agents"], list):
        raise StrictPromotionArtifactError("E0 source_agents must be a list")
    by_source_path = {
        str(Path(snapshot["source_path"]).expanduser().resolve()): snapshot["sha256"]
        for snapshot in raw["checkpoint_snapshots"]
    }
    actual_role_order: list[str] = []
    for value in raw["source_agents"]:
        agent = _exact(value, {"kind", "reference"}, "E0 source agent")
        if agent["kind"] == "checkpoint":
            try:
                reference = str(Path(agent["reference"]).expanduser().resolve())
            except (TypeError, OSError) as exc:
                raise StrictPromotionArtifactError("E0 checkpoint reference is invalid") from exc
            if reference not in by_source_path:
                raise StrictPromotionArtifactError("E0 source agent lacks an immutable snapshot")
            actual_role_order.append(by_source_path[reference])
        elif agent["kind"] == "scripted":
            actual_role_order.append(scripted_agent(agent["reference"])["sha256"])
        else:
            raise StrictPromotionArtifactError("E0 source agent kind is unsupported")
    expected_role_order = [
        request["incumbent"]["sha256"],
        *(agent["sha256"] for agent in request["checkpoint_pool"]),
        request["candidate"]["sha256"],
    ]
    if actual_role_order != expected_role_order:
        raise StrictPromotionArtifactError("E0 ordered source roles differ from the strict request")
    candidate_snapshot = next(
        snapshot
        for snapshot in raw["checkpoint_snapshots"]
        if snapshot["sha256"] == request["candidate"]["sha256"]
    )
    return str(candidate_snapshot["snapshot_path"])


def _candidate_checkpoint_contracts(
    candidate_path: str | Path,
    request: Mapping[str, Any],
) -> None:
    """Validate frozen candidate contracts from the actual E0 checkpoint bytes."""
    payload, digest, stable_path = _read_path_bytes(candidate_path, "E0 candidate checkpoint")
    if digest != request["candidate"]["sha256"]:
        raise StrictPromotionArtifactError("E0 candidate checkpoint bytes changed")
    try:
        import torch

        metadata = torch.load(BytesIO(payload), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise StrictPromotionArtifactError(
            f"cannot deserialize E0 candidate checkpoint {stable_path}: {exc}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise StrictPromotionArtifactError("E0 candidate checkpoint metadata must be a mapping")
    try:
        validated_metadata = _validate_v3_serving_checkpoint(
            dict(metadata),
            digest,
            stable_path,
        )
        runtime_descriptor = validated_metadata["runtime_contract"]
        world_descriptor = validated_metadata["effective_world"]
        source_runtime = RuntimeModeContract(**dict(runtime_descriptor))
        source_world = EffectiveWorldConfig(**dict(world_descriptor))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise StrictPromotionArtifactError(
            f"E0 candidate lifecycle metadata is invalid: {exc}"
        ) from exc
    actual = {
        "model_head_digest": metadata.get("model_head_digest"),
        "run_provenance_digest": metadata.get("run_provenance_digest"),
        "effective_world_digest": metadata.get("effective_world_digest"),
        "source_runtime": {
            "descriptor": dict(runtime_descriptor),
            "digest": metadata.get("runtime_contract_digest"),
        },
        "episode_lifecycle": {
            "descriptor": validated_metadata["_validated_lifecycle_fields"][
                "episode_lifecycle_contract"
            ],
            "digest": validated_metadata["_validated_lifecycle_fields"][
                "episode_lifecycle_contract_digest"
            ],
            "compatibility": validated_metadata["_validated_lifecycle_fields"][
                "episode_lifecycle_compatibility"
            ],
        },
        "policy_source": {
            "descriptor": validated_metadata["_validated_lifecycle_fields"][
                "policy_source_contract"
            ],
            "digest": validated_metadata["_validated_lifecycle_fields"][
                "policy_source_contract_digest"
            ],
        },
    }
    frozen = request["serving_contract"]["candidate_contracts"]
    if (
        actual != frozen
        or source_runtime.digest != metadata.get("runtime_contract_digest")
        or source_world.digest != metadata.get("effective_world_digest")
        or source_world.digest != request["profile"]["descriptor"]["world_digest"]
    ):
        raise StrictPromotionArtifactError(
            "E0 candidate checkpoint contracts differ from the strict request"
        )


def _pilot(opened: _OpenedJSON, request: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _exact(
        opened.value,
        {
            "schema_version",
            "candidate_sha256",
            "incumbent_sha256",
            "profile_digest",
            "source_closure_sha256",
            "metric_version",
            "sizing_method",
            "declared_per_mix_power",
            "family_alpha",
            "mde_by_mix",
            "pairs",
        },
        "pilot artifact",
    )
    expected_identity = (
        request["candidate"]["sha256"],
        request["incumbent"]["sha256"],
        request["profile"]["digest"],
        request["evaluator_source"]["closure_sha256"],
    )
    if (
        raw["schema_version"] != STRICT_PILOT_VERSION
        or (
            raw["candidate_sha256"],
            raw["incumbent_sha256"],
            raw["profile_digest"],
            raw["source_closure_sha256"],
        )
        != expected_identity
    ):
        raise StrictPromotionArtifactError("pilot artifact identity differs from the request")
    if raw["metric_version"] != request["profile"]["descriptor"]["metric_version"]:
        raise StrictPromotionArtifactError("pilot metric differs from the promotion profile")
    if (
        raw["sizing_method"] != PAIRED_DELTA_PILOT_METHOD
        or raw["declared_per_mix_power"] != PAIRED_DELTA_PILOT_POWER
        or raw["family_alpha"] != PAIRED_DELTA_PILOT_FAMILY_ALPHA
    ):
        raise StrictPromotionArtifactError("pilot uses an unsupported sizing declaration")
    if raw["mde_by_mix"] != request["calibration"]["mde_by_mix"]:
        raise StrictPromotionArtifactError("pilot MDE differs from predeclared calibration")
    if not isinstance(raw["pairs"], list):
        raise StrictPromotionArtifactError("pilot pairs must be a list")
    expected_keys = {
        (mix, seed) for mix in STRICT_MIXES for seed in request["seed_namespaces"]["pilot"]
    }
    values: dict[tuple[str, int], float] = {}
    expected_rosters = {
        (row["mix"], row["world_seed"]): row
        for row in materialize_rosters(
            final_world_seeds=request["seed_namespaces"]["pilot"],
            roster_width=request["roster_design"]["roster_width"],
            checkpoint_pool=request["checkpoint_pool"],
            scripted_anchors=request["scripted_anchors"],
            mixed_slot_rules=request["roster_design"]["mixed_slot_rules"],
        )
    }
    for value in raw["pairs"]:
        pair = _exact(
            value,
            {
                "mix",
                "world_seed",
                "world_identity",
                "candidate_mass_integral",
                "incumbent_mass_integral",
            },
            "pilot pair",
        )
        key = (pair["mix"], _uint32(pair["world_seed"], "pilot world seed"))
        if key not in expected_keys or key in values:
            raise StrictPromotionArtifactError("pilot pairs contain duplicate/undeclared worlds")
        identity = _exact(
            pair["world_identity"],
            {"seed_namespace", "seed", "mix_id", "ordered_slot_content_hashes", "roster_id"},
            "pilot world identity",
        )
        if identity != _expected_world_identity(expected_rosters[key]):
            raise StrictPromotionArtifactError("pilot world identity is invalid")
        candidate_mass = _finite(pair["candidate_mass_integral"], "pilot candidate mass")
        incumbent_mass = _finite(pair["incumbent_mass_integral"], "pilot incumbent mass")
        if candidate_mass < 0 or incumbent_mass < 0:
            raise StrictPromotionArtifactError("pilot mass integral must be non-negative")
        values[key] = candidate_mass - incumbent_mass
    if set(values) != expected_keys:
        raise StrictPromotionArtifactError("pilot must cover every declared pilot world/mix")
    deltas = {
        mix: [values[(mix, seed)] for seed in request["seed_namespaces"]["pilot"]]
        for mix in STRICT_MIXES
    }
    try:
        return paired_delta_pilot_size(deltas, dict(raw["mde_by_mix"]))
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise StrictPromotionArtifactError(f"invalid pilot sizing data: {exc}") from exc


def _calibration(opened: _OpenedJSON, request: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = _exact(
        opened.value,
        {
            "schema_version",
            "reference_source_sha256",
            "reference_checkpoint_sha256",
            "profile_digest",
            "source_closure_sha256",
            "seed_namespace",
            "seeds",
            "effect_derivation_method",
            "absolute_delta_ni_fraction",
            "mde_fraction_by_mix",
            "reference_records",
            "absolute_delta_ni",
            "mde_by_mix",
            "behavioral_bands",
        },
        "calibration artifact",
    )
    if (
        raw["schema_version"] != STRICT_CALIBRATION_VERSION
        or raw["seed_namespace"] != "development"
        or raw["seeds"] != request["seed_namespaces"]["development"]
        or raw["profile_digest"] != request["profile"]["digest"]
        or raw["source_closure_sha256"] != request["evaluator_source"]["closure_sha256"]
        or raw["reference_checkpoint_sha256"] != request["incumbent"]["sha256"]
    ):
        raise StrictPromotionArtifactError("calibration identity/seeds differ from the request")
    reference_source = _sha(raw["reference_source_sha256"], "calibration reference source")
    if raw["effect_derivation_method"] != "positive-reference-mean-fraction/v1":
        raise StrictPromotionArtifactError("calibration effect derivation method is unsupported")
    ni_fraction = _positive(raw["absolute_delta_ni_fraction"], "NI reference fraction")
    if not isinstance(raw["mde_fraction_by_mix"], Mapping) or set(
        raw["mde_fraction_by_mix"]
    ) != set(STRICT_MIXES):
        raise StrictPromotionArtifactError("calibration MDE fractions must name strict mixes")
    mde_fractions = {
        mix: _positive(raw["mde_fraction_by_mix"][mix], f"MDE fraction for {mix}")
        for mix in STRICT_MIXES
    }
    if not isinstance(raw["reference_records"], list):
        raise StrictPromotionArtifactError("calibration reference records must be a list")
    expected_keys = {
        (mix, seed) for mix in STRICT_MIXES for seed in request["seed_namespaces"]["development"]
    }
    reference: dict[tuple[str, int], Mapping[str, Any]] = {}
    development_rosters = {
        (row["mix"], row["world_seed"]): row
        for row in materialize_rosters(
            final_world_seeds=request["seed_namespaces"]["development"],
            roster_width=request["roster_design"]["roster_width"],
            checkpoint_pool=request["checkpoint_pool"],
            scripted_anchors=request["scripted_anchors"],
            mixed_slot_rules=request["roster_design"]["mixed_slot_rules"],
        )
    }
    for value in raw["reference_records"]:
        record = _exact(
            value,
            {"mix", "world_seed", "checkpoint_sha256", "record"},
            "calibration reference record",
        )
        key = (record["mix"], _uint32(record["world_seed"], "calibration world seed"))
        if key not in expected_keys or key in reference:
            raise StrictPromotionArtifactError(
                "calibration records contain duplicate/undeclared worlds"
            )
        if record["checkpoint_sha256"] != request["incumbent"]["sha256"]:
            raise StrictPromotionArtifactError("calibration record is not incumbent-bound")
        reference[key] = _validate_e1_record(
            record["record"], request["profile"], development_rosters[key]
        )
    if set(reference) != expected_keys:
        raise StrictPromotionArtifactError(
            "calibration must cover every declared development world"
        )
    means = {
        mix: sum(
            float(reference[(mix, seed)]["mass_integral"])
            for seed in request["seed_namespaces"]["development"]
        )
        / len(request["seed_namespaces"]["development"])
        for mix in STRICT_MIXES
    }
    if any(mean_value <= 0 for mean_value in means.values()):
        raise StrictPromotionArtifactError(
            "fraction-derived effects require positive aggregate reference means"
        )
    expected_source = canonical_digest({"records": raw["reference_records"]})
    if reference_source != expected_source:
        raise StrictPromotionArtifactError(
            "calibration source hash does not bind its actual reference records"
        )
    derived_ni = means["scripted"] * ni_fraction
    derived_mdes = {mix: means[mix] * mde_fractions[mix] for mix in STRICT_MIXES}
    declared = request["calibration"]
    if (
        _positive(raw["absolute_delta_ni"], "calibration absolute_delta_ni")
        != declared["absolute_delta_ni"]
        or raw["mde_by_mix"] != declared["mde_by_mix"]
        or raw["behavioral_bands"] != declared["behavioral_bands"]
        or not math.isclose(raw["absolute_delta_ni"], derived_ni, rel_tol=0.0, abs_tol=1e-12)
        or any(
            not math.isclose(raw["mde_by_mix"][mix], derived_mdes[mix], rel_tol=0.0, abs_tol=1e-12)
            for mix in STRICT_MIXES
        )
    ):
        raise StrictPromotionArtifactError("calibration values differ from the frozen request")
    bands = _bands(raw["behavioral_bands"])
    for band in bands:
        if band["bound_source_sha256"] != raw["reference_source_sha256"]:
            raise StrictPromotionArtifactError("behavioral band does not bind calibration source")
        selected = [
            _metric_value(record, band["metric"])
            for (mix, _), record in reference.items()
            if mix in band["mix_scope"]
        ]
        if any(value is None for value in selected):
            raise StrictPromotionArtifactError(
                "calibration behavioral source is unavailable/unimplemented"
            )
        mean_value = sum(selected) / len(selected)
        if not math.isclose(
            band["lower"], mean_value + band["lower_offset"], rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(
            band["upper"], mean_value + band["upper_offset"], rel_tol=0.0, abs_tol=1e-12
        ):
            raise StrictPromotionArtifactError("behavioral band bounds are not calibration-derived")
    return raw


def _serving_runtime_contract(
    value: Any, request: Mapping[str, Any], runtime_mode: str
) -> Mapping[str, Any]:
    """Validate the complete contract emitted by the raster31v3 serving path."""
    contract = _exact(
        value,
        {
            "obs_contract_digest",
            "model_head_digest",
            "effective_world_digest",
            "runtime_contract_digest",
            "action_mask_contract_digest",
            "run_provenance_digest",
            "deployment_profile",
            "deployment_target_manifest",
            "deployment_target_manifest_digest",
            "checkpoint_sha256",
            "episode_lifecycle_contract",
            "episode_lifecycle_contract_digest",
            "episode_lifecycle_compatibility",
            "policy_source_contract",
            "policy_source_contract_digest",
        },
        "serving runtime contract",
    )
    profile = request["profile"]["descriptor"]
    frozen = request["serving_contract"]
    candidate_contracts = frozen["candidate_contracts"]
    lifecycle_fields = {
        "episode_lifecycle_contract": candidate_contracts["episode_lifecycle"]["descriptor"],
        "episode_lifecycle_contract_digest": candidate_contracts["episode_lifecycle"]["digest"],
        "episode_lifecycle_compatibility": candidate_contracts["episode_lifecycle"][
            "compatibility"
        ],
        "policy_source_contract": candidate_contracts["policy_source"]["descriptor"],
        "policy_source_contract_digest": candidate_contracts["policy_source"]["digest"],
    }
    for name in (
        "obs_contract_digest",
        "model_head_digest",
        "effective_world_digest",
        "runtime_contract_digest",
        "action_mask_contract_digest",
        "run_provenance_digest",
        "deployment_target_manifest_digest",
        "checkpoint_sha256",
    ):
        _sha(contract[name], f"serving runtime {name}")
    if (
        contract["obs_contract_digest"] != frozen["observation"]["digest"]
        or contract["action_mask_contract_digest"] != frozen["action"]["digest"]
        or contract["model_head_digest"] != candidate_contracts["model_head_digest"]
        or contract["run_provenance_digest"] != candidate_contracts["run_provenance_digest"]
        or contract["effective_world_digest"] != candidate_contracts["effective_world_digest"]
        or contract["runtime_contract_digest"] != candidate_contracts["source_runtime"]["digest"]
        or contract["checkpoint_sha256"] != request["candidate"]["sha256"]
        or contract["deployment_profile"] != profile["name"]
        or any(contract[name] != expected for name, expected in lifecycle_fields.items())
    ):
        raise StrictPromotionArtifactError(
            "serving runtime contract differs from the frozen candidate/profile"
        )
    target = _exact(
        contract["deployment_target_manifest"],
        {
            "checkpoint_sha256",
            "deployment_profile",
            "source_world",
            "source_world_digest",
            "deployed_world",
            "deployed_world_digest",
            "source_normalization",
            "source_normalization_digest",
            "deployed_normalization",
            "deployed_normalization_digest",
            "source_runtime",
            "source_runtime_digest",
            "deployed_runtime",
            "deployed_runtime_digest",
            "episode_horizon_frames",
            "evaluation_horizon_frames",
            "evaluation_horizon_owner",
            "serving_enforced",
            "distribution_differences",
            "episode_lifecycle_contract",
            "episode_lifecycle_contract_digest",
            "episode_lifecycle_compatibility",
            "policy_source_contract",
            "policy_source_contract_digest",
        },
        "serving deployment target manifest",
    )
    if canonical_digest(dict(target)) != contract["deployment_target_manifest_digest"]:
        raise StrictPromotionArtifactError("serving deployment manifest digest is invalid")
    target_source_runtime = _exact(
        target["source_runtime"],
        {
            "mode",
            "training",
            "respawn",
            "hero_terminal",
            "population_floor",
            "reset_strategy",
        },
        "serving source runtime",
    )
    if (
        target_source_runtime != candidate_contracts["source_runtime"]["descriptor"]
        or target["source_runtime_digest"] != candidate_contracts["source_runtime"]["digest"]
    ):
        raise StrictPromotionArtifactError(
            "serving source runtime differs from the frozen checkpoint runtime"
        )
    try:
        world = EffectiveWorldConfig(**dict(profile["world"]))
        runtime = RuntimeModeContract(**dict(target_source_runtime))
        expected_target = _deployment_target_manifest(
            world,
            runtime,
            request["candidate"]["sha256"],
            runtime_mode,
            lifecycle_fields=lifecycle_fields,
        )
    except (TypeError, ValueError) as exc:
        raise StrictPromotionArtifactError(f"serving deployment target is invalid: {exc}") from exc
    if target != expected_target:
        raise StrictPromotionArtifactError(
            "serving deployment target differs from the frozen episode mode/profile"
        )
    if (
        contract["runtime_contract_digest"] != target["source_runtime_digest"]
        or target["evaluation_horizon_frames"] != profile["scored_horizon"]
    ):
        raise StrictPromotionArtifactError(
            "serving runtime/deployment lifecycle differs from the frozen profile"
        )
    return contract


def _serving(opened: _OpenedJSON, request: Mapping[str, Any]) -> list[str]:
    raw = _exact(
        opened.value,
        {
            "schema_version",
            "candidate_sha256",
            "profile_digest",
            "action_contract_digest",
            "observation_contract_digest",
            "source_closure_sha256",
            "episodes",
        },
        "serving bundle",
    )
    contract = request["serving_contract"]
    action_digest = contract["action"]["digest"]
    observation_digest = contract["observation"]["digest"]
    if (
        raw["schema_version"] != STRICT_SERVING_BUNDLE_VERSION
        or raw["candidate_sha256"] != request["candidate"]["sha256"]
        or raw["profile_digest"] != request["profile"]["digest"]
        or raw["action_contract_digest"] != action_digest
        or raw["observation_contract_digest"] != observation_digest
        or raw["source_closure_sha256"] != request["evaluator_source"]["closure_sha256"]
        or not isinstance(raw["episodes"], list)
        or len(raw["episodes"]) != 50
    ):
        raise StrictPromotionArtifactError("serving bundle identity/cardinality is invalid")
    expected_seeds = set(request["seed_namespaces"]["serving"])
    plan_episodes = request["serving_contract"]["plan"]["episodes"]
    schedule = {(row["episode_id"], row["serving_seed"]): row for row in plan_episodes}
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    receipt_hashes: list[str] = []
    for episode_index, value in enumerate(raw["episodes"]):
        episode = _exact(
            value,
            {"episode_id", "serving_seed", "receipt_path", "receipt_sha256"},
            "serving episode reference",
        )
        if not isinstance(episode["episode_id"], str) or not episode["episode_id"]:
            raise StrictPromotionArtifactError("serving episode_id must be non-empty")
        seed = _uint32(episode["serving_seed"], "serving seed")
        if episode["episode_id"] in seen_ids or seed in seen_seeds or seed not in expected_seeds:
            raise StrictPromotionArtifactError(
                "serving episodes need unique IDs and declared seeds"
            )
        schedule_key = (episode["episode_id"], seed)
        if schedule_key not in schedule:
            raise StrictPromotionArtifactError("serving episode differs from the frozen schedule")
        if schedule_key != (
            plan_episodes[episode_index]["episode_id"],
            plan_episodes[episode_index]["serving_seed"],
        ):
            raise StrictPromotionArtifactError("serving bundle order differs from the frozen plan")
        runtime_mode = schedule[schedule_key]["mode"]
        receipt = _open_json(episode["receipt_path"], "serving episode receipt")
        if receipt.byte_sha256 != _sha(episode["receipt_sha256"], "serving receipt hash"):
            raise StrictPromotionArtifactError("serving episode receipt bytes changed")
        evidence = _exact(
            receipt.value,
            {
                "schema_version",
                "episode_id",
                "serving_seed",
                "mode",
                "checkpoint_path",
                "candidate_sha256",
                "profile_digest",
                "source_closure_sha256",
                "frame_limit",
                "start_frame",
                "frames_completed",
                "end_frame",
                "completion",
                "hero_lifecycle",
                "seed_application",
                "initial_layout",
                "initial_layout_digest",
                "participants",
                "dispatch_observation",
                "obs_spec",
                "serving_contract",
                "status",
                "error",
            },
            "serving episode receipt",
        )
        if (
            evidence["schema_version"] != STRICT_SERVING_EPISODE_VERSION
            or evidence["episode_id"] != episode["episode_id"]
            or evidence["serving_seed"] != seed
            or evidence["mode"] != runtime_mode
            or evidence["candidate_sha256"] != request["candidate"]["sha256"]
            or evidence["profile_digest"] != request["profile"]["digest"]
            or evidence["source_closure_sha256"] != request["evaluator_source"]["closure_sha256"]
            or evidence["obs_spec"] != "raster31v3"
            or evidence["status"] != "completed"
            or evidence["error"] is not None
        ):
            raise StrictPromotionArtifactError("serving episode has no matching executed evidence")
        if not isinstance(evidence["checkpoint_path"], str) or not evidence["checkpoint_path"]:
            raise StrictPromotionArtifactError("serving checkpoint path must be non-empty")
        if (
            _hash_path(evidence["checkpoint_path"], "serving checkpoint bytes")
            != request["candidate"]["sha256"]
        ):
            raise StrictPromotionArtifactError("serving episode checkpoint bytes differ")
        horizon = request["profile"]["descriptor"]["scored_horizon"]
        counters = {
            name: evidence[name]
            for name in ("frame_limit", "start_frame", "frames_completed", "end_frame")
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counters.values()
        ):
            raise StrictPromotionArtifactError("serving frame counters must be integers")
        completed = counters["frames_completed"]
        if (
            counters["frame_limit"] != horizon
            or counters["start_frame"] != 0
            or not 1 <= completed <= horizon
            or counters["end_frame"] != completed
        ):
            raise StrictPromotionArtifactError(
                "serving episode frame counters do not prove a bounded completed episode"
            )
        lifecycle = _exact(
            evidence["hero_lifecycle"],
            {
                "snake_id",
                "initial_alive",
                "final_alive",
                "alive_to_dead_transitions",
                "dead_to_alive_transitions",
                "frame_rewind_count",
                "game_instance_change_count",
            },
            "serving hero lifecycle",
        )
        death_count = lifecycle["alive_to_dead_transitions"]
        if (
            lifecycle["initial_alive"] is not True
            or not isinstance(lifecycle["final_alive"], bool)
            or isinstance(death_count, bool)
            or not isinstance(death_count, int)
            or death_count not in {0, 1}
            or lifecycle["final_alive"] != (death_count == 0)
            or lifecycle["dead_to_alive_transitions"] != 0
            or lifecycle["frame_rewind_count"] != 0
            or lifecycle["game_instance_change_count"] != 0
        ):
            raise StrictPromotionArtifactError(
                "serving episode has invalid terminal/reset lifecycle evidence"
            )
        completion = _exact(
            evidence["completion"],
            {"owner", "reason", "backend_run_over", "backend_run_frames"},
            "serving completion",
        )
        backend_frames = completion["backend_run_frames"]
        if (
            not isinstance(completion["backend_run_over"], bool)
            or isinstance(backend_frames, bool)
            or not isinstance(backend_frames, int)
            or not 0 <= backend_frames <= completed
        ):
            raise StrictPromotionArtifactError("serving backend completion evidence is invalid")
        if runtime_mode == "watch":
            complete = (
                completion
                == {
                    "owner": "external-evaluator",
                    "reason": "external-horizon",
                    "backend_run_over": False,
                    "backend_run_frames": 0,
                }
                and completed == horizon
            )
        else:
            complete = (
                completion["owner"] == "GameSession._step_play"
                and completion["reason"] == "human-terminal"
                and completion["backend_run_over"] is True
                and backend_frames == completed - 1
                and death_count == 1
                and lifecycle["final_alive"] is False
            ) or (
                completion["owner"] == "external-evaluator"
                and completion["reason"] == "external-horizon"
                and completion["backend_run_over"] is False
                and backend_frames == completed
                and completed == horizon
                and death_count == 0
                and lifecycle["final_alive"] is True
            )
        if not complete:
            raise StrictPromotionArtifactError(
                "serving episode has no valid terminal or external-horizon completion"
            )
        seed_application = _exact(
            evidence["seed_application"],
            {
                "api",
                "requested_seed",
                "effective_seed",
                "global_stream_seed",
                "evaluated_build_hook",
            },
            "serving seed application",
        )
        if seed_application != {
            "api": "src.core.seeding.initialize_run_seed/v1",
            "requested_seed": seed,
            "effective_seed": seed,
            "global_stream_seed": derive_seed(seed, "global"),
            "evaluated_build_hook": (
                "GameSession.__init__" if runtime_mode == "watch" else "GameSession.set_mode(play)"
            ),
        }:
            raise StrictPromotionArtifactError(
                "serving episode does not evidence application of its frozen seed"
            )
        participants = evidence["participants"]
        world_slots = list(range(request["profile"]["descriptor"]["world"]["num_snakes"]))
        expected_candidate_slots = world_slots if runtime_mode == "watch" else world_slots[1:]
        if not isinstance(participants, list) or len(participants) != len(world_slots):
            raise StrictPromotionArtifactError(
                "serving episode participants do not cover the deployed roster"
            )
        participant_ids: set[int] = set()
        parsed_participants: dict[int, Mapping[str, Any]] = {}
        for expected_slot, value in enumerate(participants):
            participant = _exact(
                value,
                {
                    "slot",
                    "snake_id",
                    "controller",
                    "controller_class",
                    "policy_binding",
                    "checkpoint_sha256",
                },
                "serving participant",
            )
            snake_id = _uint32(participant["snake_id"], "serving participant snake_id")
            expected_controller = (
                "human" if runtime_mode == "play" and expected_slot == 0 else "candidate_ai"
            )
            expected_checkpoint = (
                None if expected_controller == "human" else request["candidate"]["sha256"]
            )
            expected_class = "HumanSnake" if expected_controller == "human" else "AISnake"
            expected_binding = None if expected_controller == "human" else "session_shared_policy"
            if (
                participant["slot"] != expected_slot
                or snake_id in participant_ids
                or participant["controller"] != expected_controller
                or participant["controller_class"] != expected_class
                or participant["policy_binding"] != expected_binding
                or participant["checkpoint_sha256"] != expected_checkpoint
            ):
                raise StrictPromotionArtifactError(
                    "serving episode participants do not prove actual human/AI dispatch"
                )
            participant_ids.add(snake_id)
            parsed_participants[expected_slot] = participant
        if lifecycle["snake_id"] != parsed_participants[0]["snake_id"]:
            raise StrictPromotionArtifactError("serving hero lifecycle identifies the wrong slot")
        descriptor = _exact(
            evidence["initial_layout"],
            {"mode", "frame", "ordered_snakes", "food_positions", "ordered_slot_to_snake_id"},
            "serving initial layout",
        )
        if (
            descriptor["mode"] != runtime_mode
            or descriptor["frame"] != 0
            or canonical_digest(dict(descriptor))
            != _sha(evidence["initial_layout_digest"], "serving initial layout digest")
            or not isinstance(descriptor["ordered_snakes"], list)
            or len(descriptor["ordered_snakes"]) != len(world_slots)
            or descriptor["ordered_slot_to_snake_id"]
            != [parsed_participants[slot]["snake_id"] for slot in world_slots]
            or not isinstance(descriptor["food_positions"], list)
        ):
            raise StrictPromotionArtifactError("serving initial layout evidence is invalid")
        heads: set[tuple[int, int]] = set()
        for expected_slot, value in enumerate(descriptor["ordered_snakes"]):
            snake = _exact(
                value,
                {
                    "slot",
                    "snake_id",
                    "controller_class",
                    "head",
                    "segments",
                    "direction",
                    "length",
                    "alive",
                    "auto_respawn",
                },
                "serving initial snake",
            )
            participant = parsed_participants[expected_slot]
            head = snake["head"]
            direction = snake["direction"]
            if (
                snake["slot"] != expected_slot
                or snake["snake_id"] != participant["snake_id"]
                or snake["controller_class"] != participant["controller_class"]
                or snake["alive"] is not True
                or snake["auto_respawn"] is not (expected_slot != 0)
                or not isinstance(head, list)
                or len(head) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in head)
                or tuple(head) in heads
                or not isinstance(direction, list)
                or len(direction) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in direction)
                or sum(abs(item) for item in direction) != 1
                or not isinstance(snake["segments"], list)
                or not snake["segments"]
                or any(
                    not isinstance(segment, list)
                    or len(segment) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in segment)
                    for segment in snake["segments"]
                )
                or snake["segments"][0] != head
                or isinstance(snake["length"], bool)
                or not isinstance(snake["length"], int)
                or snake["length"] != len(snake["segments"])
            ):
                raise StrictPromotionArtifactError("serving initial snake evidence is invalid")
            heads.add(tuple(head))
        food_positions: set[tuple[int, int]] = set()
        if descriptor["food_positions"] != sorted(descriptor["food_positions"]):
            raise StrictPromotionArtifactError("serving initial food evidence is unsorted")
        for position in descriptor["food_positions"]:
            if (
                not isinstance(position, list)
                or len(position) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in position)
                or tuple(position) in food_positions
            ):
                raise StrictPromotionArtifactError("serving initial food evidence is invalid")
            food_positions.add(tuple(position))
        dispatch_observation = _exact(
            evidence["dispatch_observation"],
            {
                "human_control_events",
                "human_update_calls",
                "candidate_action_context_calls",
                "context_id_mismatch_count",
                "unknown_context_id_count",
            },
            "serving dispatch observation",
        )
        if any(
            isinstance(dispatch_observation[name], bool)
            or not isinstance(dispatch_observation[name], int)
            or dispatch_observation[name] != 0
            for name in ("context_id_mismatch_count", "unknown_context_id_count")
        ):
            raise StrictPromotionArtifactError("serving action-context identity mismatched")
        human_events = dispatch_observation["human_control_events"]
        human_updates = dispatch_observation["human_update_calls"]
        if (
            not isinstance(human_events, list)
            or isinstance(human_updates, bool)
            or not isinstance(human_updates, int)
        ):
            raise StrictPromotionArtifactError("serving human dispatch evidence is invalid")
        if runtime_mode == "watch":
            if human_events or human_updates != 0:
                raise StrictPromotionArtifactError("Watch serving cannot claim human dispatch")
        else:
            if len(human_events) != 1 or human_updates != completed:
                raise StrictPromotionArtifactError(
                    "Play serving lacks one observed human input/update per frame"
                )
            human_event = _exact(
                human_events[0],
                {
                    "direction_name",
                    "direction_before",
                    "direction_after",
                    "run_started_before",
                    "run_started_after",
                    "accepted_observed",
                },
                "serving human control event",
            )
            before_direction = human_event["direction_before"]
            after_direction = human_event["direction_after"]
            if (
                not isinstance(human_event["direction_name"], str)
                or human_event["direction_name"] not in {"up", "down", "left", "right"}
                or human_event["run_started_before"] is not False
                or human_event["run_started_after"] is not True
                or human_event["accepted_observed"] is not True
                or not isinstance(before_direction, list)
                or not isinstance(after_direction, list)
                or len(before_direction) != 2
                or len(after_direction) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in [*before_direction, *after_direction]
                )
                or sum(abs(item) for item in before_direction) != 1
                or sum(abs(item) for item in after_direction) != 1
                or sum(a * b for a, b in zip(before_direction, after_direction)) != 0
            ):
                raise StrictPromotionArtifactError(
                    "Play serving human input is not an applied perpendicular direction"
                )
        decisions = dispatch_observation["candidate_action_context_calls"]
        if not isinstance(decisions, list) or len(decisions) != len(expected_candidate_slots):
            raise StrictPromotionArtifactError("candidate serving decisions must be a list")
        parsed_decisions: dict[int, int] = {}
        for expected_slot, value in zip(expected_candidate_slots, decisions):
            decision = _exact(
                value,
                {"slot", "snake_id", "calls"},
                "candidate serving decision",
            )
            slot = decision["slot"]
            calls = decision["calls"]
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot != expected_slot
                or slot in parsed_decisions
                or slot not in expected_candidate_slots
                or decision["snake_id"] != parsed_participants[slot]["snake_id"]
                or isinstance(calls, bool)
                or not isinstance(calls, int)
                or calls <= 0
                or calls > completed
            ):
                raise StrictPromotionArtifactError(
                    "candidate serving decisions do not cover the frozen AI slots"
                )
            parsed_decisions[slot] = calls
        if set(parsed_decisions) != set(expected_candidate_slots):
            raise StrictPromotionArtifactError(
                "candidate serving decisions do not cover the frozen AI slots"
            )
        _serving_runtime_contract(evidence["serving_contract"], request, runtime_mode)
        seen_ids.add(episode["episode_id"])
        seen_seeds.add(seed)
        receipt_hashes.append(receipt.byte_sha256)
    if seen_seeds != expected_seeds:
        raise StrictPromotionArtifactError("serving bundle does not cover the serving namespace")
    return receipt_hashes


def _validate_bound_inputs(
    request_opened: _OpenedJSON,
    e0_opened: _OpenedJSON,
    pilot_opened: _OpenedJSON,
    calibration_opened: _OpenedJSON,
    serving_opened: _OpenedJSON,
) -> tuple[Mapping[str, Any], EvaluationProfile, Mapping[str, Any], list[str]]:
    request, profile = _request(request_opened.value)
    bindings = request["artifact_bindings"]
    actual = {
        "e0": e0_opened.byte_sha256,
        "pilot": pilot_opened.byte_sha256,
        "calibration": calibration_opened.byte_sha256,
        "serving": serving_opened.byte_sha256,
    }
    if dict(bindings) != actual:
        raise StrictPromotionArtifactError("pre-world artifact bytes differ from request bindings")
    candidate_path = _validate_e0(e0_opened, request, profile)
    _candidate_checkpoint_contracts(candidate_path, request)
    pilot_sizing = _pilot(pilot_opened, request)
    _calibration(calibration_opened, request)
    serving_receipts = _serving(serving_opened, request)
    if len(request["seed_namespaces"]["final"]) < pilot_sizing["required_final_worlds"]:
        raise StrictPromotionArtifactError(
            "final worlds are below the recomputed pilot requirement"
        )
    return request, profile, pilot_sizing, serving_receipts


def freeze_strict_request(
    request_path: str | Path,
    *,
    e0_receipt_path: str | Path,
    pilot_artifact_path: str | Path,
    calibration_artifact_path: str | Path,
    serving_bundle_path: str | Path,
) -> StrictRequestToken:
    """Validate and freeze the actual pre-final-world artifact bytes."""
    opened = {
        "request": _open_json(request_path, "strict request"),
        "e0": _open_json(e0_receipt_path, "E0 receipt"),
        "pilot": _open_json(pilot_artifact_path, "pilot artifact"),
        "calibration": _open_json(calibration_artifact_path, "calibration artifact"),
        "serving": _open_json(serving_bundle_path, "serving bundle"),
    }
    request, _, pilot_sizing, _ = _validate_bound_inputs(
        opened["request"], opened["e0"], opened["pilot"], opened["calibration"], opened["serving"]
    )
    semantic = canonical_digest(dict(request))
    return StrictRequestToken(
        request_path=opened["request"].path,
        request_byte_sha256=opened["request"].byte_sha256,
        request_semantic_digest=semantic,
        request=_freeze(dict(request)),
        required_final_worlds=pilot_sizing["required_final_worlds"],
        artifact_byte_sha256=_freeze(dict(request["artifact_bindings"])),
        checkpoint_role_sha256=_freeze(
            {
                "candidate": request["candidate"]["sha256"],
                "incumbent": request["incumbent"]["sha256"],
                "checkpoint_pool": [agent["sha256"] for agent in request["checkpoint_pool"]],
            }
        ),
    )


def _expected_world_identity(roster: Mapping[str, Any]) -> dict[str, Any]:
    hashes = [slot["member_sha256"] for slot in roster["slots"]]
    return {
        "seed_namespace": "evaluation-world/v1",
        "seed": roster["world_seed"],
        "mix_id": roster["mix"],
        "ordered_slot_content_hashes": hashes,
        "roster_id": canonical_digest({"slots": hashes}),
    }


def _validate_e1_record(
    record_value: Any, profile: Mapping[str, Any], roster: Mapping[str, Any]
) -> Mapping[str, Any]:
    record = _exact(
        record_value,
        {
            "seed",
            "evaluation_profile",
            "evaluation_profile_digest",
            "world_identity",
            "mass_integral",
            "max_mass",
            "mean_mass_alive",
            "survival_fraction",
            "kills",
            "deaths",
            "probes",
            "probe_unavailable_reasons",
            "denominators",
        },
        "E1 actual world record",
    )
    if (
        record["seed"] != roster["world_seed"]
        or record["evaluation_profile"] != profile["descriptor"]
        or record["evaluation_profile_digest"] != profile["digest"]
        or record["world_identity"] != _expected_world_identity(roster)
    ):
        raise StrictPromotionArtifactError("actual world record identity differs from its roster")
    if (
        _finite(record["mass_integral"], "mass_integral") < 0
        or _finite(record["max_mass"], "max_mass") < 0
    ):
        raise StrictPromotionArtifactError("mass metrics must be non-negative")
    if (
        record["mean_mass_alive"] is not None
        and _finite(record["mean_mass_alive"], "mean_mass_alive") < 0
    ):
        raise StrictPromotionArtifactError("mean_mass_alive must be non-negative/null")
    survival = _finite(record["survival_fraction"], "survival_fraction")
    if not 0 <= survival <= 1:
        raise StrictPromotionArtifactError("survival_fraction must be within [0, 1]")
    for name in ("kills", "deaths"):
        if isinstance(record[name], bool) or not isinstance(record[name], int) or record[name] < 0:
            raise StrictPromotionArtifactError(f"{name} must be a non-negative integer")
    probes = _exact(
        record["probes"],
        {
            "food_eaten",
            "boost_frame_fraction",
            "death_cause",
            "peak_length",
            "kill_opportunity_count",
            "entrapment_event",
        },
        "E1 probes",
    )
    if not isinstance(record["probe_unavailable_reasons"], Mapping):
        raise StrictPromotionArtifactError("probe_unavailable_reasons must be a mapping")
    reasons = record["probe_unavailable_reasons"]
    if any(
        key not in probes or not isinstance(reason, str) or not reason
        for key, reason in reasons.items()
    ):
        raise StrictPromotionArtifactError("probe unavailable reasons must name actual probes")
    for name in ("food_eaten", "kill_opportunity_count"):
        value = probes[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise StrictPromotionArtifactError(f"probe {name} must be non-negative integer/null")
    for name in ("boost_frame_fraction", "peak_length"):
        value = probes[name]
        if value is not None:
            parsed = _finite(value, f"probe {name}")
            if parsed < 0 or (name == "boost_frame_fraction" and parsed > 1):
                raise StrictPromotionArtifactError(f"probe {name} is outside its valid range")
    if probes["death_cause"] is not None and (
        not isinstance(probes["death_cause"], str) or not probes["death_cause"]
    ):
        raise StrictPromotionArtifactError("death_cause must be a non-empty string/null")
    if probes["entrapment_event"] is not None and not isinstance(probes["entrapment_event"], bool):
        raise StrictPromotionArtifactError("entrapment_event must be boolean/null")
    fixed_unavailable = {
        "kill_opportunity_count": "not supplied by this transition adapter",
        "entrapment_event": "temporal probe is not integrated into this accumulator",
    }
    if any(
        probes[name] is not None or reasons.get(name) != reason
        for name, reason in fixed_unavailable.items()
    ):
        raise StrictPromotionArtifactError(
            "this evaluator version must preserve its fixed unavailable probes"
        )
    for name, value in probes.items():
        if name in reasons and value is not None:
            raise StrictPromotionArtifactError(
                "probe availability and unavailable reasons contradict each other"
            )
    for name in ("boost_frame_fraction", "kill_opportunity_count", "entrapment_event"):
        if probes[name] is None and name not in reasons:
            raise StrictPromotionArtifactError(
                "probe availability and unavailable reasons contradict each other"
            )
    denominators = _exact(
        record["denominators"],
        {
            "scored_frames",
            "decision_frames",
            "alive_frames",
            "food_event_frames",
            "boost_executed_frames",
        },
        "E1 denominators",
    )
    if denominators["scored_frames"] != 5000:
        raise StrictPromotionArtifactError("actual record does not cover the fixed horizon")
    for name, value in denominators.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StrictPromotionArtifactError(f"denominator {name} must be non-negative")
    if denominators["alive_frames"] > 5000 or denominators["decision_frames"] > 5000:
        raise StrictPromotionArtifactError("actual record denominators exceed the horizon")
    alive_frames = denominators["alive_frames"]
    decision_frames = denominators["decision_frames"]
    if not math.isclose(survival, alive_frames / 5000, rel_tol=0.0, abs_tol=1e-12):
        raise StrictPromotionArtifactError("survival_fraction contradicts alive_frames")
    if decision_frames != alive_frames + record["deaths"]:
        raise StrictPromotionArtifactError(
            "terminal decision/alive/death lifecycle is inconsistent"
        )
    if record["deaths"] == 0 and (alive_frames != 5000 or decision_frames != 5000):
        raise StrictPromotionArtifactError("a surviving terminal hero must cover the full horizon")
    if denominators["food_event_frames"] > decision_frames:
        raise StrictPromotionArtifactError("food event denominator exceeds decision frames")
    mass_sum = float(record["mass_integral"]) * 5000
    if alive_frames:
        if record["mean_mass_alive"] is None or not math.isclose(
            float(record["mean_mass_alive"]), mass_sum / alive_frames, rel_tol=0.0, abs_tol=1e-9
        ):
            raise StrictPromotionArtifactError("mean mass contradicts mass integral/alive frames")
        if float(record["max_mass"]) < float(record["mean_mass_alive"]):
            raise StrictPromotionArtifactError("max_mass cannot be below mean_mass_alive")
    elif record["mean_mass_alive"] is not None or mass_sum != 0:
        raise StrictPromotionArtifactError("dead-for-horizon record has inconsistent mass")
    if probes["peak_length"] != record["max_mass"]:
        raise StrictPromotionArtifactError("peak_length contradicts max_mass")
    if probes["food_eaten"] != denominators["food_event_frames"]:
        raise StrictPromotionArtifactError("food probe contradicts its denominator")
    if decision_frames:
        if probes["boost_frame_fraction"] is None or not math.isclose(
            float(probes["boost_frame_fraction"]),
            denominators["boost_executed_frames"] / decision_frames,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StrictPromotionArtifactError("boost probe contradicts its denominators")
    elif probes["boost_frame_fraction"] is not None or denominators["boost_executed_frames"]:
        raise StrictPromotionArtifactError("zero-decision record has inconsistent boost evidence")
    if record["deaths"] > 1 or (record["deaths"] == 0) != (probes["death_cause"] is None):
        raise StrictPromotionArtifactError("terminal hero death fields are inconsistent")
    return record


def _raw_worlds(
    opened: _OpenedJSON, request: Mapping[str, Any], request_digest: str
) -> tuple[dict[str, list[float]], list[Mapping[str, Any]]]:
    raw = _exact(
        opened.value,
        {
            "schema_version",
            "engine",
            "strict_request_semantic_digest",
            "e0_receipt_sha256",
            "profile_digest",
            "source_closure_sha256",
            "materialized_rosters_digest",
            "records",
        },
        "raw world artifact",
    )
    roster_digest = canonical_digest({"rows": request["materialized_rosters"]})
    if (
        raw["schema_version"] != STRICT_RAW_WORLD_VERSION
        or raw["engine"] != "live"
        or raw["strict_request_semantic_digest"] != request_digest
        or raw["e0_receipt_sha256"] != request["artifact_bindings"]["e0"]
        or raw["profile_digest"] != request["profile"]["digest"]
        or raw["source_closure_sha256"] != request["evaluator_source"]["closure_sha256"]
        or raw["materialized_rosters_digest"] != roster_digest
        or not isinstance(raw["records"], list)
    ):
        raise StrictPromotionArtifactError("raw world artifact identity/schema is invalid")
    rosters = {(row["mix"], row["world_seed"]): row for row in request["materialized_rosters"]}
    expected = {
        (role, mix, seed)
        for role in ("candidate", "incumbent")
        for mix in STRICT_MIXES
        for seed in request["seed_namespaces"]["final"]
    }
    records: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for value in raw["records"]:
        wrapper = _exact(
            value, {"role", "mix", "world_seed", "checkpoint_sha256", "record"}, "raw world row"
        )
        key = (wrapper["role"], wrapper["mix"], wrapper["world_seed"])
        if key not in expected or key in records:
            raise StrictPromotionArtifactError(
                "raw worlds contain duplicate/undeclared compound keys"
            )
        expected_checkpoint = request[wrapper["role"]]["sha256"]
        if wrapper["checkpoint_sha256"] != expected_checkpoint:
            raise StrictPromotionArtifactError("raw world role/checkpoint identity is incorrect")
        roster = rosters[(wrapper["mix"], wrapper["world_seed"])]
        records[key] = _validate_e1_record(wrapper["record"], request["profile"], roster)
    if set(records) != expected:
        raise StrictPromotionArtifactError("raw world artifact lacks full 2 x 3 x N cardinality")
    deltas = {
        mix: [
            records[("candidate", mix, seed)]["mass_integral"]
            - records[("incumbent", mix, seed)]["mass_integral"]
            for seed in request["seed_namespaces"]["final"]
        ]
        for mix in STRICT_MIXES
    }
    candidate_records = [
        {"mix": mix, "record": records[("candidate", mix, seed)]}
        for mix in STRICT_MIXES
        for seed in request["seed_namespaces"]["final"]
    ]
    return deltas, candidate_records


def bind_strict_raw_world_artifact(
    request_token: StrictRequestToken, raw_world_artifact_path: str | Path
) -> StrictRawArtifactToken:
    """Validate and bind the actual post-world bytes to a frozen request."""
    if not isinstance(request_token, StrictRequestToken):
        raise StrictPromotionArtifactError("a validated StrictRequestToken is required")
    request_opened = _open_json(request_token.request_path, "strict request")
    if (
        request_opened.byte_sha256 != request_token.request_byte_sha256
        or canonical_digest(dict(request_opened.value)) != request_token.request_semantic_digest
    ):
        raise StrictPromotionArtifactError("strict request bytes changed after freezing")
    request, _ = _request(request_opened.value)
    raw = _open_json(raw_world_artifact_path, "raw world artifact")
    _raw_worlds(raw, request, request_token.request_semantic_digest)
    return StrictRawArtifactToken(
        raw_world_artifact_path=raw.path,
        raw_world_artifact_byte_sha256=raw.byte_sha256,
        request_semantic_digest=request_token.request_semantic_digest,
    )


def _metric_value(record: Mapping[str, Any], path: str) -> float | None:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    if isinstance(current, bool):
        return float(current)
    if not isinstance(current, (int, float)) or not math.isfinite(float(current)):
        return None
    return float(current)


def _behavioral_measurements(
    records: Sequence[Mapping[str, Any]], bands: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    for band in bands:
        scoped_records = [
            record["record"] for record in records if record["mix"] in band["mix_scope"]
        ]
        values = [_metric_value(record, band["metric"]) for record in scoped_records]
        unavailable = any(value is None for value in values)
        measurement = (
            None
            if unavailable
            else sum(value for value in values if value is not None) / len(values)
        )
        passes = bool(measurement is not None and band["lower"] <= measurement <= band["upper"])
        output.append(
            {
                "name": band["name"],
                "metric": band["metric"],
                "aggregation": "mean_over_candidate_final_worlds",
                "mix_scope": list(band["mix_scope"]),
                "denominator": band["denominator"],
                "n": len(values),
                "value": measurement,
                "lower": band["lower"],
                "upper": band["upper"],
                "available": not unavailable,
                "passes": passes,
            }
        )
    return output, all(entry["passes"] for entry in output)


def build_strict_final_receipt(
    request_token: StrictRequestToken,
    raw_token: StrictRawArtifactToken,
    *,
    request_path: str | Path,
    e0_receipt_path: str | Path,
    raw_world_artifact_path: str | Path,
    pilot_artifact_path: str | Path,
    calibration_artifact_path: str | Path,
    serving_bundle_path: str | Path,
) -> dict[str, Any]:
    """Reopen all evidence and derive the only strict promotion authority bit."""
    if not isinstance(request_token, StrictRequestToken) or not isinstance(
        raw_token, StrictRawArtifactToken
    ):
        raise StrictPromotionArtifactError("validated request and raw artifact tokens are required")
    opened = {
        "request": _open_json(request_path, "strict request"),
        "e0": _open_json(e0_receipt_path, "E0 receipt"),
        "raw": _open_json(raw_world_artifact_path, "raw world artifact"),
        "pilot": _open_json(pilot_artifact_path, "pilot artifact"),
        "calibration": _open_json(calibration_artifact_path, "calibration artifact"),
        "serving": _open_json(serving_bundle_path, "serving bundle"),
    }
    request_semantic = canonical_digest(dict(opened["request"].value))
    if (
        opened["request"].path != request_token.request_path
        or opened["request"].byte_sha256 != request_token.request_byte_sha256
        or request_semantic != request_token.request_semantic_digest
        or opened["raw"].path != raw_token.raw_world_artifact_path
        or opened["raw"].byte_sha256 != raw_token.raw_world_artifact_byte_sha256
        or raw_token.request_semantic_digest != request_semantic
    ):
        raise StrictPromotionArtifactError("request/raw artifact bytes changed after binding")
    request, _, pilot_sizing, serving_receipts = _validate_bound_inputs(
        opened["request"], opened["e0"], opened["pilot"], opened["calibration"], opened["serving"]
    )
    if pilot_sizing["required_final_worlds"] != request_token.required_final_worlds:
        raise StrictPromotionArtifactError("pilot requirement changed after request freezing")
    deltas, candidate_records = _raw_worlds(opened["raw"], request, request_semantic)
    decision = strict_promotion_decision(
        deltas,
        scripted_mix="scripted",
        absolute_delta_ni=request["calibration"]["absolute_delta_ni"],
    )
    measurements, bands_pass = _behavioral_measurements(
        candidate_records, request["calibration"]["behavioral_bands"]
    )
    authority = bool(decision["valid"] and decision["passes"] and bands_pass)
    return {
        "schema_version": STRICT_FINAL_RECEIPT_VERSION,
        "strict_authority": authority,
        "release_action_authorized": False,
        "strict_request": {
            "byte_sha256": opened["request"].byte_sha256,
            "semantic_digest": request_semantic,
        },
        "artifact_byte_sha256": {
            "e0": opened["e0"].byte_sha256,
            "raw_worlds": opened["raw"].byte_sha256,
            "pilot": opened["pilot"].byte_sha256,
            "calibration": opened["calibration"].byte_sha256,
            "serving": opened["serving"].byte_sha256,
        },
        "serving_episode_receipt_sha256s": serving_receipts,
        "candidate": _thaw(request["candidate"]),
        "incumbent": _thaw(request["incumbent"]),
        "profile": _thaw(request["profile"]),
        "evaluator_source": _thaw(request["evaluator_source"]),
        "seed_namespaces": _thaw(request["seed_namespaces"]),
        "roster_design_digest": canonical_digest(dict(request["roster_design"])),
        "materialized_rosters_digest": canonical_digest({"rows": request["materialized_rosters"]}),
        "pilot_sizing": pilot_sizing,
        "paired_deltas_by_mix": deltas,
        "statistics": decision,
        "behavioral_measurements": measurements,
        "readiness": {
            "artifact_validation": "passed",
            "serving_episodes": 50,
            "behavioral_bands_passed": bands_pass,
        },
    }


def _atomic_create_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return destination


def write_strict_final_receipt(
    path: str | Path,
    request_token: StrictRequestToken,
    raw_token: StrictRawArtifactToken,
    **artifact_paths: str | Path,
) -> Path:
    """Create, fsync, and never overwrite a derived strict final receipt."""
    receipt = build_strict_final_receipt(request_token, raw_token, **artifact_paths)
    return _atomic_create_json(path, receipt)


def validate_strict_final_receipt(
    receipt_path: str | Path,
    request_token: StrictRequestToken,
    raw_token: StrictRawArtifactToken,
    **artifact_paths: str | Path,
) -> Mapping[str, Any]:
    """Recompute an archived receipt from its exact evidence instead of trusting booleans."""
    archived = _open_json(receipt_path, "strict final receipt")
    expected = build_strict_final_receipt(request_token, raw_token, **artifact_paths)
    if archived.value != expected:
        raise StrictPromotionArtifactError(
            "strict final receipt differs from recomputed artifact evidence"
        )
    return archived.value
