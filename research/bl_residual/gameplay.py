"""F3's sealed gameplay-development screen hook.

The hook deliberately has a narrow job: it evaluates already-frozen controllers
on a fresh, paired H5000 bank using the existing terminal-hero evaluator.  It
does not create a bank, inspect factory labels, or turn a screen into a release
decision.  In particular, production execution requires a hash-bound ratified
F4 successor; environment variables are never an approval mechanism.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.bl_served_search import ceiling

B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
HORIZON = 5_000
MIXES = ("scripted", "frozen", "mixed")
COUNT_PER_MIX = 128
DOSES = ("D1024", "D4096")
ARMS = ("B8", *DOSES)
PROBE_COUNT_PER_MIX = {"scripted": 11, "frozen": 11, "mixed": 10}
WATCH6_HORIZON = 14_400
SCHEMA = "bl-residual-gameplay-screen/v1"
PROBE_SCHEMA = "bl-residual-gameplay-probe/v1"
WATCH6_SCHEMA = "bl-residual-watch6/v1"
F4_REGISTRATION_SHA256 = "66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65"


class ControllerFault(RuntimeError):
    """A registered, deterministic controller fault that closes the affected arm."""


def sha256_file(path: Path) -> str:
    """Return the checkpoint digest without accepting a caller-provided digest."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_mac() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("F4 gameplay development screen is Mac-only")


def _approved_successor(evidence: Mapping[str, Any]) -> dict[str, str]:
    """Bind production only to the exact F4 block and its approval JSON."""
    if not isinstance(evidence, Mapping):
        raise ValueError("production requires explicit hash-bound F4 review evidence")
    registration = evidence.get("registration_sha256")
    recipe = evidence.get("recipe_sha256")
    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import ROUND_26_JUDGMENT_SHA, canonical_sha

    expected = {
        "schema": "bl-residual-f4-judgment/v1",
        "decision": "APPROVE",
        "judgment_sha256": ROUND_26_JUDGMENT_SHA,
        "registration_sha256": F4_REGISTRATION_SHA256,
        "recipe_sha256": canonical_sha(asdict(FitRecipe())),
    }
    if (
        set(evidence) != set(expected)
        or dict(evidence) != expected
        or not _require_sha256(recipe, "recipe digest")
    ):
        raise RuntimeError("F4 is unratified: production gameplay is refused")
    registration_path = Path(__file__).with_name("F4_REGISTRATION.md")
    if sha256_file(registration_path) != registration:
        raise RuntimeError("F4 registration bytes do not match the approval JSON")
    return {"registration_sha256": registration, "recipe_sha256": recipe}


def allocation_sha256(rows: list[Mapping[str, Any]]) -> str:
    """Digest precisely the paired-world identities controlled by a freshness audit."""
    identity = [
        {
            "world_id": row["world_id"],
            "parent_world_id": row["parent_world_id"],
            "world_seed": row["world_seed"],
            "mix": row["mix"],
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate_collision_receipt(
    bank: Mapping[str, Any], rows: list[dict[str, Any]], *, production: bool
) -> None:
    """Bind the live allocation to an independently written collision receipt."""
    receipt = bank.get("collision_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("status") != "CLEAR":
        raise ValueError("bank requires a CLEAR collision receipt")
    if receipt.get("namespace") not in (None, bank["namespace"]):
        raise ValueError("collision receipt namespace mismatch")
    if not production:
        return
    path, digest = receipt.get("path"), receipt.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(digest, str):
        raise ValueError("production requires a file-bound collision receipt")
    materialized = Path(path)
    if not materialized.is_file() or sha256_file(materialized) != _require_sha256(
        digest, "receipt sha"
    ):
        raise ValueError("collision receipt file is absent or digest mismatch")
    try:
        recorded = json.loads(materialized.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("collision receipt is not readable JSON") from exc
    exact = allocation_sha256(rows)
    if (
        not isinstance(recorded, Mapping)
        or recorded.get("status") != "CLEAR"
        or recorded.get("namespace") != bank["namespace"]
        or recorded.get("allocation_sha256") != exact
        or receipt.get("allocation_sha256") != exact
        or bank.get("allocation_sha256") != exact
    ):
        raise ValueError("collision receipt is not bound to this exact allocation")


def _validate_materialized_rosters(rows: list[dict[str, Any]]) -> None:
    """Require the evaluator's frozen H5000 roster constructor, not a synthetic map."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    expected = planned_rosters(
        gid._context(1), [row["world_seed"] for row in rows], [row["mix"] for row in rows]
    )
    for row, plan in zip(rows, expected):
        if row["plan"] != plan:
            raise ValueError("bank plan differs from the frozen terminal-hero roster constructor")


def validate_bank(
    bank: Mapping[str, Any], *, training_parent_ids: set[str] | None = None, production: bool
) -> list[dict[str, Any]]:
    """Fail closed on pairing, namespace, collision, and fixed-N violations."""
    if (
        not isinstance(bank, Mapping)
        or not isinstance(bank.get("namespace"), str)
        or not bank["namespace"]
    ):
        raise ValueError("bank requires a nonempty fresh namespace")
    rows = bank.get("rows")
    if not isinstance(rows, list):
        raise ValueError("bank rows must be a list")
    expected_counts = (
        {mix: COUNT_PER_MIX for mix in MIXES}
        if production
        else {
            "scripted": 1,
            "frozen": 0,
            "mixed": 0,
        }
    )
    counts = {mix: 0 for mix in MIXES}
    ids: set[str] = set()
    seeds: set[int] = set()
    parent_ids: set[str] = set()
    normalised: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("bank row must be an object")
        world_id, parent_id, mix, seed, plan = (
            row.get("world_id"),
            row.get("parent_world_id"),
            row.get("mix"),
            row.get("world_seed"),
            row.get("plan"),
        )
        if (
            not isinstance(world_id, str)
            or not world_id
            or not isinstance(parent_id, str)
            or not parent_id
        ):
            raise ValueError("bank row lacks world/parent identity")
        if mix not in counts or type(seed) is not int or not isinstance(plan, dict):
            raise ValueError("bank row has malformed mix, seed, or materialized plan")
        if plan.get("mix") != mix or int(plan.get("seed", -1)) != seed:
            raise ValueError("bank row plan identity mismatch")
        if world_id in ids or seed in seeds or parent_id in parent_ids:
            raise ValueError("bank has duplicate world, seed, or parent identity")
        if training_parent_ids is not None and parent_id in training_parent_ids:
            raise ValueError("gameplay parent collides with training/held-out parent")
        ids.add(world_id)
        seeds.add(seed)
        parent_ids.add(parent_id)
        counts[mix] += 1
        normalised.append(dict(row))
    if counts != expected_counts:
        raise ValueError(f"bank has wrong fixed-N mix counts: {counts}")
    if production and training_parent_ids is None:
        raise ValueError("production requires the training/held-out parent allocation")
    _validate_collision_receipt(bank, normalised, production=production)
    _validate_materialized_rosters(normalised)
    return normalised


def _root_digest(plan: Mapping[str, Any], *, b8_path: Path) -> str:
    """Exercise the existing exact clone/digest seam before the evaluator runs."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t34_shadow.shadow import clone_world_exact, full_digest

    root = build_world("b8_s3", dict(plan), hero_checkpoint=b8_path)
    digest = full_digest(root)
    # ``CapturedWorld.clone`` establishes the selector-prepared branch seam.
    # clone_world_exact deliberately normalizes that lifecycle state, so the
    # exact digest comparison belongs on that prepared snapshot rather than on
    # a never-dispatched capture.
    prepared = root.clone()
    if full_digest(clone_world_exact(prepared)) != full_digest(prepared):
        raise RuntimeError("exact root clone digest mismatch")
    return digest


def _step_with_trace(
    world: Any, decision: int, *, reference_hero: Any | None = None
) -> dict[str, int]:
    """Take one exact terminal-world step and retain the delivered action.

    This is intentionally a thin composition of the frozen T3 dispatch path.  It
    does not approximate a simulator step or reimplement action masking.  When
    ``reference_hero`` is supplied, it is scored at the *candidate's prepared
    state*, which is the R4 disagreement estimand.
    """
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, int] = {}

    def choose(prepared: Any) -> np.ndarray:
        masks = prepared.get_resolved_action_mask()
        domain_legal = prepared.get_legal_action_mask()[0, 0]
        world.runtime.prepare(prepared)
        alive = bool(prepared.alive[0, 0])
        length = int(prepared.length[0, 0]) if alive else 0
        advisory = prepared.get_advisory_action_mask()[0, 0]
        obs = build_ego_raster(
            prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
        )
        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ControllerFault("nonfinite_or_malformed_q")
        legal, action = ceiling._legal(q, advisory)
        if not bool(domain_legal[action]):
            raise ControllerFault("illegal_served_action")
        selected.update(alive=int(alive), hero_length=length, served_action=int(action))
        if reference_hero is not None:
            candidate_hero = world.hero
            world.hero = reference_hero
            try:
                reference_q = np.asarray(prepared_q(world, obs), dtype=np.float32)
            finally:
                world.hero = candidate_hero
            _, reference_action = ceiling._legal(reference_q, advisory)
            selected["b8_served_action"] = int(reference_action)
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_actions(prepared, masks, actions, [world.hero], [world.opponents])
        if int(actions[0, 0]) != action:
            raise RuntimeError(f"dispatch/Q mismatch at decision {decision}")
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return selected


def _run_exact_arm_from_root(
    root: Any, hero: Any, *, horizon: int, reference_hero: Any | None = None
) -> dict[str, Any]:
    """Run one arm from a clone of the same prepared root with an action trace."""
    from research.t34_shadow.shadow import clone_world_exact

    world = clone_world_exact(root)
    world.hero = copy.copy(hero)
    mass = alive_frames = 0
    trace: list[dict[str, int]] = []
    for decision in range(horizon):
        selected = _step_with_trace(world, decision, reference_hero=reference_hero)
        if not selected["alive"]:
            break
        trace.append({"decision": decision, **selected})
        alive = bool(world.sim.alive[0, 0])
        alive_frames += int(alive)
        mass += int(world.sim.length[0, 0]) if alive else 0
        if not alive:
            break
    decisions = len(trace)
    if not decisions:
        raise RuntimeError("terminal arm produced no hero decisions")
    boost_share = sum(row["served_action"] >= 3 for row in trace) / decisions
    return {
        "mass_integral": mass / horizon,
        "alive_frame_fraction": alive_frames / horizon,
        "no_death_episode": bool(world.sim.alive[0, 0]),
        "decisions": decisions,
        "action_trace": trace,
        "boost_share": boost_share,
    }


def _build_exact_roots(
    plan: Mapping[str, Any], paths: Mapping[str, Path]
) -> tuple[Any, dict[str, Any]]:
    """Build controllers once, then run every arm from exact cloned state bytes."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t34_shadow.shadow import clone_world_exact, full_digest

    templates = {
        name: build_world("b8_s3", dict(plan), hero_checkpoint=path) for name, path in paths.items()
    }
    root = templates["B8"].clone()
    before = full_digest(root, checkpoint="bl-residual-f4-root-v1")
    if full_digest(clone_world_exact(root), checkpoint="bl-residual-f4-root-v1") != before:
        raise RuntimeError("exact root clone digest mismatch")
    return root, templates


def _validate_candidates(paths: Mapping[str, Path], metadata: Mapping[str, Any] | None) -> None:
    """Require immutable final-export identity before production gameplay."""
    if not isinstance(metadata, Mapping) or set(metadata) != set(paths):
        raise RuntimeError("production requires final candidate metadata for every arm")
    import torch

    from research.bl_ppo.exports import verify_export
    from research.bl_residual.protocol import canonical_sha

    expected_recipe = canonical_sha(
        asdict(__import__("research.bl_residual.fit", fromlist=["FitRecipe"]).FitRecipe())
    )
    for dose, path in paths.items():
        item = metadata[dose]
        stage_path = (
            Path(str(item.get("stage_receipt", ""))) if isinstance(item, Mapping) else Path()
        )
        if not isinstance(item, Mapping) or item.get("dose") != dose or not stage_path.is_file():
            raise RuntimeError("candidate metadata does not bind the frozen final export")
        try:
            fit_sha = verify_export(path)
            verify_export(stage_path)
            stage = json.loads(stage_path.read_text())
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            raise RuntimeError("candidate export or stage receipt is unsealed") from exc
        meta = checkpoint.get("meta", {}) if isinstance(checkpoint, Mapping) else {}
        expected_dose = int(dose[1:])
        if (
            item.get("sha256") != fit_sha
            or stage.get("schema") != "bl-residual-stage-fit/v1"
            or stage.get("dose") != expected_dose
            or stage.get("fit_sha256") != fit_sha
            or stage.get("recipe_sha256") != expected_recipe
            or stage.get("complete") is not True
            or meta.get("schema") != "bl-residual-fit/v1"
            or meta.get("synthetic") is not False
            or meta.get("dose") != expected_dose
            or meta.get("recipe_sha256") != expected_recipe
            or meta.get("updates")
            != __import__("research.bl_residual.fit", fromlist=["FitRecipe"]).FitRecipe().updates
        ):
            raise RuntimeError("candidate is not this dose's sealed final registered fit")


def _collapse_tripwire(observations: list[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """Evaluate R4's full sliding 100k-decision action-collapse window."""
    actions = [
        int(row["served_action"])
        for observation in observations
        for row in observation["arms"][arm]["action_trace"]
    ]
    window = 100_000
    counts = [0] * 6
    max_share = 0.0
    for index, action in enumerate(actions):
        counts[action] += 1
        if index >= window:
            counts[actions[index - window]] -= 1
        if index >= window - 1:
            max_share = max(max_share, max(counts) / window)
    return {
        "decisions": len(actions),
        "window": window,
        "full_window_observed": len(actions) >= window,
        "max_full_window_action_share": max_share if len(actions) >= window else None,
        "pass": len(actions) < window or max_share <= 0.90,
    }


def run_screen(
    bank: Mapping[str, Any],
    *,
    b8_path: Path,
    candidates: Mapping[str, Path],
    review_evidence: Mapping[str, Any] | None = None,
    candidate_metadata: Mapping[str, Any] | None = None,
    horizon: int = HORIZON,
    training_parent_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Materialize paired observations and explicitly decline a scientific verdict.

    ``horizon <= 3`` is a plumbing fixture only.  It permits one scripted row
    while F4 remains unratified; it cannot emit a decision or count as data.
    """
    if horizon <= 0 or horizon > HORIZON:
        raise ValueError("horizon must be in [1, 5000]")
    fixture = horizon <= 3
    if not fixture:
        if horizon != HORIZON:
            raise ValueError("production gameplay horizon is exactly H5000")
        _require_mac()
        approval = _approved_successor(review_evidence or {})
    else:
        approval = None
    rows = validate_bank(bank, training_parent_ids=training_parent_ids, production=not fixture)
    if fixture and any(row["mix"] != "scripted" for row in rows):
        raise ValueError("fixture screen permits only scripted worlds")
    if set(candidates) != set(DOSES):
        raise ValueError("candidates must contain exactly D256, D1024, and D4096")
    b8_path = Path(b8_path)
    if sha256_file(b8_path) != B8_SHA256:
        raise ValueError("baseline is not pinned B8 s3")
    checked = {dose: Path(path) for dose, path in candidates.items()}
    if any(not path.is_file() for path in checked.values()):
        raise ValueError("candidate checkpoint missing")
    if not fixture:
        _validate_candidates(checked, candidate_metadata)

    observations = []
    for row in rows:
        plan = row["plan"]
        root_digest = _root_digest(plan, b8_path=b8_path)
        root, templates = _build_exact_roots(plan, {"B8": b8_path, **checked})
        arms, faults = {}, {}
        for arm in ARMS:
            try:
                arms[arm] = _run_exact_arm_from_root(root, templates[arm].hero, horizon=horizon)
            except ControllerFault as exc:
                faults[arm] = {"kind": "KILL_FIT", "fault": str(exc)}
        if faults:
            return {
                "schema": SCHEMA,
                "namespace": bank["namespace"],
                "horizon": horizon,
                "fixture": fixture,
                "approval": approval,
                "known_controller_fault": faults,
                "decision": {"label": "KILL_FIT", "reason": "registered_controller_fault"},
            }
        observations.append(
            {
                "world_id": row["world_id"],
                "parent_world_id": row["parent_world_id"],
                "world_seed": row["world_seed"],
                "mix": row["mix"],
                "root_full_digest": root_digest,
                "arms": arms,
                "boost_share": {name: result["boost_share"] for name, result in arms.items()},
            }
        )
    return {
        "schema": SCHEMA,
        "namespace": bank["namespace"],
        "horizon": horizon,
        "fixture": fixture,
        "approval": approval,
        "controller_sha256": {
            "B8": B8_SHA256,
            **{dose: sha256_file(path) for dose, path in checked.items()},
        },
        "collision_receipt": dict(bank["collision_receipt"]),
        "observations": observations,
        "trajectory_tripwires": {
            arm: not _collapse_tripwire(observations, arm)["pass"] for arm in DOSES
        },
        "trajectory_tripwire_detail": {arm: _collapse_tripwire(observations, arm) for arm in ARMS},
        "decision": {
            "label": "NOT_A_DECISION",
            "reason": "F4 decision analyzer awaits ratified review",
        },
    }


def validate_probe_bank(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate R4's separate, collision-audited 32-world probe allocation."""
    if not isinstance(bank, Mapping) or not isinstance(bank.get("namespace"), str):
        raise ValueError("probe bank requires a namespace")
    rows = bank.get("rows")
    if not isinstance(rows, list):
        raise ValueError("probe bank rows must be a list")
    normalised = []
    counts = {mix: 0 for mix in MIXES}
    world_ids: set[str] = set()
    seeds: set[int] = set()
    parents: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("probe row must be an object")
        copied = dict(row)
        world_id, seed, parent = (
            copied.get("world_id"),
            copied.get("world_seed"),
            copied.get("parent_world_id"),
        )
        if (
            not isinstance(world_id, str)
            or not world_id
            or type(seed) is not int
            or not isinstance(parent, str)
            or not parent
            or world_id in world_ids
            or seed in seeds
            or parent in parents
            or copied.get("mix") not in counts
        ):
            raise ValueError("probe bank has invalid or duplicate world identity")
        world_ids.add(world_id)
        seeds.add(seed)
        parents.add(parent)
        counts[copied["mix"]] += 1
        normalised.append(copied)
    if counts != PROBE_COUNT_PER_MIX:
        raise ValueError(f"probe bank requires R4's 11/11/10 mix allocation: {counts}")
    _validate_collision_receipt(bank, normalised, production=True)
    _validate_materialized_rosters(normalised)
    return normalised


def run_probe(
    bank: Mapping[str, Any],
    *,
    b8_path: Path,
    d4096_path: Path,
    heldout_receipt: Mapping[str, Any],
    review_evidence: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Run R4's D4096-only fresh probe and publish its exact tripwire inputs."""
    _require_mac()
    _approved_successor(review_evidence)
    _validate_candidates({"D4096": d4096_path}, candidate_metadata)
    from research.bl_ppo.exports import verify_export

    if not isinstance(heldout_receipt, Mapping):
        raise ValueError("probe requires a sealed held-out retention receipt")
    receipt_path = Path(str(heldout_receipt.get("receipt_path", "")))
    receipt_sha = verify_export(receipt_path)
    recorded = json.loads(receipt_path.read_text())
    if (
        heldout_receipt.get("receipt_sha256") != receipt_sha
        or recorded
        != {
            key: value
            for key, value in heldout_receipt.items()
            if key not in {"receipt_path", "receipt_sha256"}
        }
        or heldout_receipt.get("schema") != "bl-residual-heldout/v2"
        or heldout_receipt.get("dose") != 4096
        or heldout_receipt.get("checkpoint_sha256") != sha256_file(d4096_path)
        or heldout_receipt.get("registration_sha256") != F4_REGISTRATION_SHA256
        or not isinstance(heldout_receipt.get("metrics"), Mapping)
    ):
        raise ValueError("probe requires a sealed held-out retention receipt")
    heldout_retention = heldout_receipt["metrics"].get("life_macro_rescue_retention")
    if (
        isinstance(heldout_retention, bool)
        or not isinstance(heldout_retention, (int, float))
        or not np.isfinite(heldout_retention)
        or not 0 <= heldout_retention <= 1
    ):
        raise ValueError("held-out receipt lacks finite life-macro retention")
    rows = validate_probe_bank(bank)
    if sha256_file(b8_path) != B8_SHA256 or not Path(d4096_path).is_file():
        raise ValueError("probe requires pinned B8 and a materialized D4096 checkpoint")
    traces, b8_boosts, b8_decisions = [], 0, 0
    for row in rows:
        try:
            root, templates = _build_exact_roots(row["plan"], {"B8": b8_path, "D4096": d4096_path})
            candidate = _run_exact_arm_from_root(
                root, templates["D4096"].hero, horizon=HORIZON, reference_hero=templates["B8"].hero
            )
            baseline = _run_exact_arm_from_root(root, templates["B8"].hero, horizon=HORIZON)
        except ControllerFault as exc:
            return {
                "schema": PROBE_SCHEMA,
                "namespace": bank["namespace"],
                "horizon": HORIZON,
                "controller_sha256": {"B8": B8_SHA256, "D4096": sha256_file(d4096_path)},
                "known_controller_fault": {"D4096": {"kind": "KILL_FIT", "fault": str(exc)}},
                "pass": False,
                "tripwires": {"controller_fault": False},
            }
        traces.append({"world_id": row["world_id"], "mix": row["mix"], **candidate})
        b8_boosts += sum(x["served_action"] >= 3 for x in baseline["action_trace"])
        b8_decisions += baseline["decisions"]
    actions = [x for row in traces for x in row["action_trace"]]
    if not actions or any("b8_served_action" not in x for x in actions):
        raise RuntimeError("probe lacks same-state B8 counterfactual actions")
    lengths = np.asarray([x["hero_length"] for x in actions], dtype=np.float64)
    edges = np.quantile(lengths, np.arange(1, 10) / 10.0)
    bins = np.searchsorted(edges, lengths, side="right")
    disagree = np.asarray([x["served_action"] != x["b8_served_action"] for x in actions])
    deciles = [
        float(disagree[bins == index].mean()) if (bins == index).any() else 0.0
        for index in range(10)
    ]
    boost_share = float(np.mean([x["served_action"] >= 3 for x in actions]))
    b8_boost_share = b8_boosts / b8_decisions
    tripwires = {
        "max_decile_disagreement_le_0_05": max(deciles) <= 0.05,
        "boost_share_le_1_5x_b8": boost_share <= 1.5 * b8_boost_share,
        "heldout_retention_ge_0_70": heldout_retention >= 0.70,
    }
    collapse = _collapse_tripwire([{"arms": {"D4096": {"action_trace": actions}}}], "D4096")
    tripwires["action_collapse"] = bool(collapse["pass"])
    return {
        "schema": PROBE_SCHEMA,
        "namespace": bank["namespace"],
        "controller_sha256": {"B8": B8_SHA256, "D4096": sha256_file(d4096_path)},
        "horizon": HORIZON,
        "world_counts": dict(PROBE_COUNT_PER_MIX),
        "traces": traces,
        "decile_edges": edges.tolist(),
        "disagreement_by_length_decile": deciles,
        "d4096_boost_share": boost_share,
        "b8_boost_share": b8_boost_share,
        "heldout_retention": heldout_retention,
        "tripwires": tripwires,
        "action_collapse": {"D4096": collapse},
        "pass": all(tripwires.values()),
    }


def run_watch6_sessions(
    bank: Mapping[str, Any],
    *,
    b8_path: Path,
    d4096_path: Path,
    review_evidence: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    h5000_go_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Run R8's paired continuous-respawn Watch6 product sessions.

    Callers must provide a fresh, collision-audited 128-row allocation.  This
    runner deliberately has no analyzer or verdict path.
    """
    _require_mac()
    _approved_successor(review_evidence)
    _validate_candidates({"D4096": d4096_path}, candidate_metadata)
    from research.bl_ppo.exports import verify_export

    receipt_path = (
        Path(str(h5000_go_receipt.get("receipt_path", "")))
        if isinstance(h5000_go_receipt, Mapping)
        else Path()
    )
    if (
        not isinstance(h5000_go_receipt, Mapping)
        or h5000_go_receipt.get("schema") != "bl-residual-f4-h5000-go/v1"
        or h5000_go_receipt.get("status") != "PASS_WATCH6"
        or h5000_go_receipt.get("scientific_result") is not False
        or h5000_go_receipt.get("all_go_bars_pass") is not True
        or h5000_go_receipt.get("registration_sha256") != F4_REGISTRATION_SHA256
        or not _require_sha256(h5000_go_receipt.get("analyzer_sha256"), "analyzer source")
        or not _require_sha256(h5000_go_receipt.get("screen_input_sha256"), "screen input")
        or not _require_sha256(h5000_go_receipt.get("recipe_sha256"), "recipe")
        or not receipt_path.is_file()
    ):
        raise RuntimeError("Watch6 is refused until a source-bound H5000 GO receipt exists")
    try:
        receipt_sha = verify_export(receipt_path)
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("H5000 GO receipt is not sealed") from exc
    if (
        h5000_go_receipt.get("receipt_sha256") != receipt_sha
        or receipt
        != {
            key: value
            for key, value in h5000_go_receipt.items()
            if key not in {"receipt_path", "receipt_sha256"}
        }
        or receipt.get("controller_sha256", {}).get("B8") != B8_SHA256
        or receipt.get("controller_sha256", {}).get("D4096") != sha256_file(d4096_path)
        or not _require_sha256(receipt.get("controller_sha256", {}).get("D1024"), "D1024")
    ):
        raise RuntimeError("H5000 GO receipt does not bind these final controllers")
    if not isinstance(bank, Mapping) or not isinstance(bank.get("rows"), list):
        raise ValueError("Watch6 requires an audited bank object")
    rows = [dict(row) for row in bank["rows"] if isinstance(row, Mapping)]
    if len(rows) != 128 or any(row.get("mix") != "mixed" for row in rows):
        raise ValueError("Watch6 requires exactly 128 fresh mixed sessions")
    session_ids, seeds, parents = set(), set(), set()
    audit_rows = []
    for row in rows:
        session_id, seed, parent = (
            row.get("session_id"),
            row.get("seed"),
            row.get("parent_world_id"),
        )
        if (
            not isinstance(session_id, str)
            or not session_id
            or type(seed) is not int
            or not isinstance(parent, str)
            or not parent
            or session_id in session_ids
            or seed in seeds
            or parent in parents
        ):
            raise ValueError("Watch6 has duplicate or malformed session identity")
        session_ids.add(session_id)
        seeds.add(seed)
        parents.add(parent)
        audit_rows.append(
            {"world_id": session_id, "parent_world_id": parent, "world_seed": seed, "mix": "mixed"}
        )
    _validate_collision_receipt(bank, audit_rows, production=True)
    if sha256_file(b8_path) != B8_SHA256 or not Path(d4096_path).is_file():
        raise ValueError("Watch6 requires pinned B8 and D4096 checkpoints")
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t34_pathb.adapter_v2 import RunSession, open_watch
    from research.t34_pathb.engine_v2 import NonFiniteQ
    from research.t34_shadow.shadow import clone_world_exact, full_digest

    sessions = []
    for row in rows:
        with closing(open_watch(dict(row), 24, "bare")) as source:
            root = clone_world_exact(source.world)
            root_digest = full_digest(root, checkpoint="bl-residual-f4-watch-root-v1")
            candidate_template = build_world(
                "b8_s3",
                {"seed": int(row["seed"]), "mix": "mixed", "specs": source.specs},
                hero_checkpoint=d4096_path,
            )
            arms: dict[str, dict[str, Any]] = {}
            for name, hero, sha in (
                ("B8", source.world.hero, B8_SHA256),
                ("D4096", candidate_template.hero, sha256_file(d4096_path)),
            ):
                arm_root = clone_world_exact(root)
                arm_root.hero = copy.copy(hero)
                session = RunSession(arm_root, "bare", specs=source.specs, checkpoint_sha=sha)
                masses: list[int] = []
                lengths: list[int] = []
                trace: list[dict[str, int]] = []
                death_events: list[dict[str, int]] = []
                try:
                    for decision in range(WATCH6_HORIZON):
                        session.step()
                        if session.decisions_delta:
                            trace.append(
                                {
                                    "decision": decision,
                                    "served_action": session.executed_action,
                                    "hero_length": session.hero_length,
                                }
                            )
                        masses.append(session.hero_length)
                        lengths.append(session.hero_length)
                        if session.hero_death:
                            death_events.append(
                                {"frame": decision, "pre_death_length": session.death_length}
                            )
                except (ControllerFault, NonFiniteQ) as exc:
                    return {
                        "schema": WATCH6_SCHEMA,
                        "namespace": bank["namespace"],
                        "horizon": WATCH6_HORIZON,
                        "controller_sha256": {"B8": B8_SHA256, "D4096": sha256_file(d4096_path)},
                        "known_controller_fault": {name: {"kind": "KILL_FIT", "fault": str(exc)}},
                    }
                finally:
                    session.close()
                arms[name] = {
                    "deaths": len(death_events),
                    "hero_decisions": session.decisions,
                    "mass_integral": sum(masses) / WATCH6_HORIZON,
                    "action_trace": trace,
                    "boost_share": session.boost_decisions / session.decisions,
                    "median_length": float(np.median(lengths)),
                    "p95_length": float(np.percentile(lengths, 95)),
                    "death_events": death_events,
                }
            sessions.append(
                {
                    "session_id": str(row["session_id"]),
                    "root_full_digest": root_digest,
                    "arms": arms,
                }
            )
    watch_collapse = {
        arm: _collapse_tripwire(
            [{"arms": {arm: session["arms"][arm]}} for session in sessions], arm
        )
        for arm in ("B8", "D4096")
    }
    return {
        "schema": WATCH6_SCHEMA,
        "horizon": WATCH6_HORIZON,
        "controller_sha256": {"B8": B8_SHA256, "D4096": sha256_file(d4096_path)},
        "sessions": sessions,
        "action_collapse": watch_collapse,
    }


def canonical_receipt(receipt: Mapping[str, Any]) -> str:
    """Return stable receipt bytes for the package's create-only publisher."""
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fixture_bank(seed: int) -> dict[str, Any]:
    """Build a real one-world scripted fixture through the frozen roster constructor."""
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    plan = planned_rosters(gid._context(1), [seed], ["scripted"])[0]
    return {
        "namespace": "bl-residual-fixture/v1",
        "collision_receipt": {"status": "CLEAR", "namespace": "bl-residual-fixture/v1"},
        "rows": [
            {
                "world_id": f"fixture-{seed}",
                "parent_world_id": f"fixture-{seed}",
                "world_seed": seed,
                "mix": "scripted",
                "plan": plan,
            }
        ],
    }


def _sealed_receipt_wrapper(path: Path) -> dict[str, Any]:
    """Load a complete producer export and bind its actual bytes for a consumer."""
    from research.bl_ppo.exports import verify_export

    receipt_sha = verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or {"receipt_path", "receipt_sha256"} & value.keys():
        raise ValueError("expected a producer receipt, not a nested receipt wrapper")
    return {**value, "receipt_path": str(path.resolve()), "receipt_sha256": receipt_sha}


def main() -> None:
    """Run the real fixture or a prepared production bank; never create a decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--kind", choices=("screen", "probe", "watch6"), default="screen")
    parser.add_argument("--fixture-seed", type=int, default=3403)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="DOSE=PATH")
    parser.add_argument("--review-evidence", type=Path)
    parser.add_argument("--candidate-metadata", type=Path)
    parser.add_argument("--heldout-receipt", type=Path)
    parser.add_argument("--go-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--training-parents", type=Path)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    if args.fixture == (args.bank is not None):
        parser.error("choose exactly one of --fixture or --bank")
    if args.kind != "screen" and args.fixture:
        parser.error("probe and Watch6 refuse fixture mode")
    candidates: dict[str, Path] = {}
    for item in args.candidate:
        dose, separator, raw_path = item.partition("=")
        if not separator or dose in candidates:
            parser.error("--candidate must be unique DOSE=PATH")
        candidates[dose] = Path(raw_path)
    bank = _fixture_bank(args.fixture_seed) if args.fixture else json.loads(args.bank.read_text())
    review = json.loads(args.review_evidence.read_text()) if args.review_evidence else None
    candidate_metadata = (
        json.loads(args.candidate_metadata.read_text()) if args.candidate_metadata else None
    )
    parents = set(json.loads(args.training_parents.read_text())) if args.training_parents else None
    if args.kind == "screen":
        result = run_screen(
            bank,
            b8_path=args.b8,
            candidates=candidates,
            review_evidence=review,
            candidate_metadata=candidate_metadata,
            horizon=min(args.horizon, 3) if args.fixture else args.horizon,
            training_parent_ids=parents,
        )
    elif args.kind == "probe":
        if not args.heldout_receipt or "D4096" not in candidates:
            parser.error("probe requires --heldout-receipt and D4096 candidate")
        result = run_probe(
            bank,
            b8_path=args.b8,
            d4096_path=candidates["D4096"],
            heldout_receipt=_sealed_receipt_wrapper(args.heldout_receipt),
            review_evidence=review or {},
            candidate_metadata=candidate_metadata or {},
        )
    else:
        if not args.go_receipt or "D4096" not in candidates:
            parser.error("watch6 requires --go-receipt and D4096 candidate")
        result = run_watch6_sessions(
            bank,
            b8_path=args.b8,
            d4096_path=candidates["D4096"],
            review_evidence=review or {},
            candidate_metadata=candidate_metadata or {},
            h5000_go_receipt=_sealed_receipt_wrapper(args.go_receipt),
        )
    if args.out:
        from research.bl_ppo.exports import export_json

        export_json(args.out, result)
    else:
        print(canonical_receipt(result))


if __name__ == "__main__":
    main()
