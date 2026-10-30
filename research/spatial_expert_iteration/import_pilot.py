"""Import the GO-nominated pilot final learner without resetting Adam or RNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.model import load_checkpoint
from research.spatial_student_pilot.prepare import B8_SHA, digest
from research.spatial_student_pilot.resume import UnitStore

from .config import RoundConfig
from .run import save_tree
from .search import tensor_identity


def convert(
    snapshot: dict[str, Any], actor_state: dict[str, Any], final_update: int
) -> dict[str, Any]:
    """Convert an already verified full final trajectory, never weights-only."""
    required = {"model", "optimizer", "generator", "torch_rng", "cuda_rng", "update", "final"}
    if (
        not required <= snapshot.keys()
        or snapshot["final"] is not True
        or snapshot["update"] != final_update
        or tensor_identity(snapshot["model"]) != tensor_identity(actor_state)
    ):
        raise ValueError("pilot final full-state/actor identity mismatch")
    return dict(
        schema="spatial-exit-learner/v1",
        model=snapshot["model"],
        optimizer=snapshot["optimizer"],
        generator=snapshot["generator"],
        torch_rng=snapshot["torch_rng"],
        cuda_rng=snapshot["cuda_rng"],
        round_update=0,
        global_update=final_update,
        round_index=0,
        final=True,
    )


def training_roots(store: UnitStore, cfg: PilotConfig, mature_length: int) -> list[dict[str, Any]]:
    """Import only natural training roots meeting the follow-on maturity threshold."""
    if type(mature_length) is not int or mature_length < 1:
        raise ValueError("positive mature_length required")
    roots = []
    for name in store.list_ids("collect-"):
        for root in store.get(name)["roots"]:
            if root["partition"] != "train" or root["hero_length"] < mature_length:
                continue
            roots.append(
                dict(
                    root,
                    parent_id=f"{cfg.namespace}/collection/w{root['world_id']}",
                    parent_partition="train",
                    parent_seed=cfg.world_seed(root["world_id"]),
                    parent_opp_seed=cfg.opp_seed(root["world_id"]),
                    collector_sha256=B8_SHA,
                    student_action=root["b8_action"],
                    start_distribution="fresh",
                    ancestor_uid=None,
                )
            )
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-go", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--verified-ledger", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir-output", type=Path, required=True)
    parser.add_argument("--mature-length", type=int, default=RoundConfig().mature_length)
    args = parser.parse_args()
    if args.mature_length < 1:
        parser.error("positive mature-length required; match the registered round config")
    go = json.loads(args.pilot_go.read_text())
    if (
        go.get("schema") != "spatial-student-readout/v1"
        or go.get("verdict") != "GO"
        or go.get("candidate_sha256") != digest(args.actor)
    ):
        raise ValueError("verified pilot GO nomination required")
    cfg = PilotConfig()
    # A separate fresh scratch directory is required by UnitStore. No new unit
    # is computed; its ledger verifies bytes and allocation accounting only.
    scratch = args.output.parent / (args.output.name + ".verification")
    store = UnitStore(scratch, cfg, json.loads(args.verified_ledger.read_text()), args.input_dir)
    final = cfg.bc_updates + cfg.improvement_updates
    snapshot = store.get(f"fit-{final:07d}")
    if snapshot is None:
        raise ValueError("verified final pilot learner snapshot is absent")
    actor, _ = load_checkpoint(args.actor)
    save_tree(args.output, convert(snapshot, actor.state_dict(), final))
    roots = training_roots(store, cfg, args.mature_length)
    save_tree(
        args.reservoir_output,
        dict(
            schema="spatial-exit-reservoir/v1",
            roots=roots,
            source=dict(
                kind="verified-pilot-training",
                pilot_go_sha256=digest(args.pilot_go),
                pilot_config_sha256=cfg.sha256,
                ledger_sha256=digest(args.verified_ledger),
                mature_length=args.mature_length,
            ),
        ),
    )


if __name__ == "__main__":
    main()
