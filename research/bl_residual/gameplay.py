"""F3's sealed gameplay-development screen hook.

The hook deliberately has a narrow job: it evaluates already-frozen controllers
on a fresh, paired H5000 bank using the existing terminal-hero evaluator.  It
does not create a bank, inspect factory labels, or turn a screen into a release
decision.  In particular, production execution requires a hash-bound ratified
F4 successor; environment variables are never an approval mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

from research.bl_served_search import ceiling

B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
HORIZON = 5_000
MIXES = ("scripted", "frozen", "mixed")
COUNT_PER_MIX = 128
DOSES = ("D256", "D1024", "D4096")
SCHEMA = "bl-residual-gameplay-screen/v1"


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
    """Validate explicit immutable review evidence; never read approval env vars."""
    if not isinstance(evidence, Mapping):
        raise ValueError("production requires explicit hash-bound F4 review evidence")
    status = evidence.get("status")
    registration = evidence.get("registration_sha256")
    review = evidence.get("independent_review_sha256")
    if status != "RATIFIED" or not all(
        _require_sha256(v, "approval digest") for v in (registration, review)
    ):
        raise RuntimeError("F4 is unratified: production gameplay is refused")
    # The draft is intentionally not acceptable evidence, even if its bytes are
    # supplied.  A ratified immutable successor must name a different artifact.
    draft = Path(__file__).with_name("F4_DRAFT.md").resolve()
    registration_path = evidence.get("registration_path")
    if registration_path == "research/bl_residual/F4_DRAFT.md" or (
        isinstance(registration_path, str) and Path(registration_path).resolve() == draft
    ):
        raise RuntimeError("F4 draft cannot authorize production gameplay")
    for path_key, digest_key in (
        ("registration_path", "registration_sha256"),
        ("independent_review_path", "independent_review_sha256"),
    ):
        path = evidence.get(path_key)
        if not isinstance(path, str) or not path:
            raise RuntimeError("production approval must name immutable review files")
        materialized = Path(path)
        if not materialized.is_file() or sha256_file(materialized) != evidence[digest_key]:
            raise RuntimeError("production approval file is absent or its digest does not bind")
    return {"registration_sha256": registration, "independent_review_sha256": review}


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


def run_screen(
    bank: Mapping[str, Any],
    *,
    b8_path: Path,
    candidates: Mapping[str, Path],
    review_evidence: Mapping[str, Any] | None = None,
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

    observations = []
    for row in rows:
        plan = row["plan"]
        root_digest = _root_digest(plan, b8_path=b8_path)
        baseline = ceiling.run_arm(plan, dense=False, b8_path=b8_path, horizon=horizon)
        arms = {"B8": baseline}
        for dose, path in checked.items():
            arms[dose] = ceiling.run_arm(plan, dense=False, b8_path=path, horizon=horizon)
        observations.append(
            {
                "world_id": row["world_id"],
                "parent_world_id": row["parent_world_id"],
                "world_seed": row["world_seed"],
                "mix": row["mix"],
                "root_full_digest": root_digest,
                "arms": {
                    name: {
                        "mass_integral": result["mass_integral"],
                        "alive_frame_fraction": result["alive_frame_fraction"],
                        "no_death_episode": result["no_death_episode"],
                    }
                    for name, result in arms.items()
                },
                "boost_share": {
                    "status": "UNAVAILABLE",
                    "reason": "ceiling.run_arm has no action trace",
                },
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
        "decision": {
            "label": "NOT_A_DECISION",
            "reason": "F4 decision analyzer awaits ratified review",
        },
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


def main() -> None:
    """Run the real fixture or a prepared production bank; never create a decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-seed", type=int, default=3403)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="DOSE=PATH")
    parser.add_argument("--review-evidence", type=Path)
    parser.add_argument("--training-parents", type=Path)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()
    if args.fixture == (args.bank is not None):
        parser.error("choose exactly one of --fixture or --bank")
    candidates: dict[str, Path] = {}
    for item in args.candidate:
        dose, separator, raw_path = item.partition("=")
        if not separator or dose in candidates:
            parser.error("--candidate must be unique DOSE=PATH")
        candidates[dose] = Path(raw_path)
    bank = _fixture_bank(args.fixture_seed) if args.fixture else json.loads(args.bank.read_text())
    review = json.loads(args.review_evidence.read_text()) if args.review_evidence else None
    parents = set(json.loads(args.training_parents.read_text())) if args.training_parents else None
    result = run_screen(
        bank,
        b8_path=args.b8,
        candidates=candidates,
        review_evidence=review,
        horizon=min(args.horizon, 3) if args.fixture else args.horizon,
        training_parent_ids=parents,
    )
    print(canonical_receipt(result))


if __name__ == "__main__":
    main()
