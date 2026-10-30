"""Create-only round packet admission and automatic pre-data freeze; no allocator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from research.sandbox_a_resident.ct_run import B8_S3_SHA256
from research.spatial_student_pilot.prepare import (
    CHECKPOINT_NAMES,
    OPPONENT_SHAS,
    binding,
    digest,
    runtime_source_paths,
)
from research.spatial_student_pilot.resume import (
    _publish_bytes,
    empty_ledger,
    upload_inputs,
    validate_ledger,
)

from .config import RoundConfig, identity
from .data import validate_rows
from .reservoir import capacity_forecast
from .search import tensor_identity

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/spatial_expert_iteration"


def write_json(path: Path, value: Any) -> None:
    _publish_bytes(
        path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )


def sources(commit: str, *, tiny: bool) -> dict[str, str]:
    """Freeze full pilot runtime closure plus this package; exclude tests."""
    paths = set(runtime_source_paths(commit)) if not tiny else set()
    paths.update(
        str(p.relative_to(ROOT))
        for p in (ROOT / PACKAGE).glob("*.py")
        if not p.name.startswith("test_")
    )
    if tiny:
        # The smoke still binds its actual direct source dependencies.
        paths.update(
            str(p.relative_to(ROOT))
            for p in (ROOT / "research/spatial_student_pilot").glob("*.py")
            if not p.name.startswith("test_")
        )
    result = {}
    for name in sorted(paths):
        path = ROOT / name
        result[name] = digest(path)
        if not tiny:
            pinned = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            if hashlib.sha256(pinned).hexdigest() != result[name]:
                raise ValueError(f"source differs from committed pin: {name}")
    return result


def seed_keys(cfg: RoundConfig) -> list[int]:
    keys = []
    for phase, count in (
        ("collection", cfg.worlds),
        ("assessment", cfg.assessment_worlds),
        ("gameplay", cfg.gameplay_worlds),
    ):
        for i in range(count):
            keys.extend((cfg.world_seed(i, phase), cfg.opp_seed(i, phase)))
    return keys


def seed_audit(cfg: RoundConfig, inventory: dict[str, Any]) -> dict[str, Any]:
    """Reject the entire allocation on any collision, including prior rounds."""
    keys = seed_keys(cfg)
    excluded = inventory.get("excluded_seeds")
    if not isinstance(excluded, list) or any(
        type(key) is not int or not 0 <= key <= 0x7FFF_FFFF_FFFF_FFFF for key in excluded
    ):
        raise ValueError("integer effective seed inventory required")
    if len(keys) != len(set(keys)) or set(keys) & set(excluded):
        raise ValueError("effective seed collision; reject whole allocation")
    if not cfg.tiny and (
        inventory.get("scope") != "mac-historical-and-prior-rounds"
        or not inventory.get("prior_round_freezes")
    ):
        raise ValueError("bind Mac historical inventory and prior round/pilot freezes")
    return dict(
        schema="spatial-exit-seed-audit/v1",
        config_sha256=cfg.sha256,
        effective_keys=keys,
        inventory_sha256=identity(inventory),
        collision_count=0,
    )


def freeze(
    packet: Path,
    cfg: RoundConfig,
    *,
    inputs: dict[str, Path],
    registration: Path,
    inventory: Path,
    ledger: dict[str, Any] | None = None,
    resume_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """No round data can start until registration, provenance and budgets bind.

    Real round 1 additionally binds the Mac pilot GO and its nominated actor.
    Later rounds must bind the preceding final full learner state and reservoir.
    This is preparation only; the caller cannot ratify this draft template.
    """
    if not packet.is_absolute() or packet.exists():
        raise ValueError("fresh absolute packet required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    reg = json.loads(registration.read_text())
    if not cfg.tiny:
        if (
            reg.get("status") != "RATIFIED"
            or not reg.get("independent_judge")
            or reg.get("config_sha256") != cfg.sha256
            or not reg.get("decision_rule")
            or not reg.get("heldout_policy")
            or not reg.get("round_schedule")
            or reg.get("release_authority") != "owner-only"
        ):
            raise ValueError("prospective independent ratification and full registration required")
        rel = registration.resolve().relative_to(ROOT)
        if (
            subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
            != registration.read_bytes()
        ):
            raise ValueError("registration must be committed before data")
        if reg.get("seed_inventory_sha256") != digest(inventory):
            raise ValueError("registration must bind the Mac seed inventory")
        if reg.get("input_sha256") != {name: digest(path) for name, path in inputs.items()}:
            raise ValueError("registration must pin every round input before data")
        for dest, expected in {"b8-s3.pth": B8_S3_SHA256, **OPPONENT_SHAS}.items():
            if dest not in inputs or digest(inputs[dest]) != expected:
                raise ValueError(f"canonical frozen checkpoint mismatch: {dest}")
        go = json.loads(inputs["pilot-go.json"].read_text())
        if (
            go.get("schema") != "spatial-student-readout/v1"
            or go.get("verdict") != "GO"
            or not go.get("clauses")
            or not all(v is True for v in go["clauses"].values())
        ):
            raise ValueError("follow-on requires verified pilot GO")
        if cfg.round_index == 1 and go.get("candidate_sha256") != digest(
            inputs["initial-actor.pth"]
        ):
            raise ValueError("round-one actor differs from pilot GO nomination")
    warm = torch.load(inputs["learner.pt"], map_location="cpu", weights_only=False)
    if (
        warm.get("schema") != "spatial-exit-learner/v1"
        or warm.get("final") is not True
        or warm.get("round_index") != cfg.round_index - 1
    ):
        raise ValueError("only preceding final full learner state may warm-start")
    reservoir = torch.load(inputs["reservoir.pt"], map_location="cpu", weights_only=False)
    if not isinstance(reservoir, dict) or reservoir.get("schema") != "spatial-exit-reservoir/v1":
        raise ValueError("foreign reservoir schema")
    validate_rows(reservoir["roots"], "train")
    if not cfg.tiny:
        previous = (
            json.loads(inputs["parent-endpoint.json"].read_text()) if cfg.round_index > 1 else None
        )
        if previous:
            if (
                previous.get("status") != "ENDPOINT_EXPORTED_NO_DECISION"
                or previous.get("round_index") != cfg.round_index - 1
                or previous.get("program") != cfg.program
                or previous.get("learner_sha256") != digest(inputs["learner.pt"])
                or previous.get("reservoir_sha256") != digest(inputs["reservoir.pt"])
            ):
                raise ValueError("previous endpoint lineage mismatch")
        elif tensor_identity(warm["model"]) != tensor_identity(
            torch.load(inputs["initial-actor.pth"], map_location="cpu", weights_only=True)[
                "model_state_dict"
            ]
        ):
            raise ValueError("initial learner model differs from admitted actor")
        # All retained parents must come from the frozen predecessor reservoir;
        # importing biased historical C/T requires a separately registered path.
        if cfg.round_index == 1 and reservoir["roots"]:
            from research.spatial_student_pilot.config import PilotConfig

            source = reservoir.get("source", {})
            if (
                source.get("kind") != "verified-pilot-training"
                or source.get("pilot_go_sha256") != digest(inputs["pilot-go.json"])
                or source.get("pilot_config_sha256") != PilotConfig().sha256
            ):
                raise ValueError(
                    "round-one reservoir requires verified natural pilot training roots"
                )
    capacity = capacity_forecast(cfg, reservoir["roots"])
    if set(seed_keys(cfg)) & {
        int(r[k]) for r in reservoir["roots"] for k in ("parent_seed", "parent_opp_seed")
    }:
        raise ValueError("fresh round allocation collides with reservoir ancestry")
    audited = seed_audit(cfg, json.loads(inventory.read_text()))
    ledger = ledger if ledger is not None else empty_ledger(cfg)
    validate_ledger(ledger, cfg)
    resumed = upload_inputs(ledger, resume_roots or [], cfg)
    hashes = sources(commit, tiny=cfg.tiny)
    if not cfg.tiny and cfg.implementation_sha256 != identity(hashes):
        raise ValueError("round config must bind committed source closure before data")
    packet.mkdir(parents=True)
    write_json(packet / "config.json", cfg.to_dict())
    write_json(packet / "verified-ledger.json", ledger)
    write_json(packet / "seed-audit.json", audited)
    all_inputs = {**inputs, "REGISTRATION.json": registration, "seed-inventory.json": inventory}
    for name, path in all_inputs.items():
        if Path(name).name != name:
            raise ValueError("packet inputs must have simple relative names")
        _publish_bytes(packet / name, path.read_bytes())
    for entry in resumed:
        _publish_bytes(packet / entry["dest"], Path(entry["path"]).read_bytes())
    resumed = [binding(packet / entry["dest"], entry["dest"]) for entry in resumed]
    bindings = [binding(packet / name, name) for name in sorted(all_inputs)]
    frozen = dict(
        schema="spatial-exit-freeze/v1",
        config_sha256=cfg.sha256,
        namespace=cfg.namespace,
        allocation_index=cfg.allocation_index,
        round_index=cfg.round_index,
        source_commit=commit,
        source_sha256=hashes,
        inputs={r["dest"]: r["sha256"] for r in bindings},
        continuation_sha256=tensor_identity(warm["model"]),
        opponents_sha256=identity(
            {
                name: digest(path)
                for name, path in inputs.items()
                if name in {"b8-s3.pth", *OPPONENT_SHAS}
            }
        ),
        ledger_sha256=digest(packet / "verified-ledger.json"),
        seed_audit_sha256=digest(packet / "seed-audit.json"),
        reservoir_capacity=capacity,
        budgets=dict(
            base_timeout_seconds=cfg.base_timeout_seconds,
            effective_timeout_seconds=cfg.timeout_seconds,
            candidate_roots=cfg.candidate_roots,
            reservoir_max_roots=cfg.reservoir_max_roots,
            reservoir_max_bytes=cfg.reservoir_max_bytes,
            updates=cfg.updates,
            maximum_query_branch_steps=cfg.candidate_roots * 6 * 2 * cfg.branch_horizon,
            max_outcome_blind_retries=2,
        ),
        tiny=cfg.tiny,
        dispatchable=False,
    )
    write_json(packet / "FREEZE.json", frozen)
    # Reuse supervisor binding/ledger conventions, without pretending a qualified
    # Colab manifest exists. The operator must prepare canaries/transport later.
    write_json(
        packet / "PACKET.json",
        dict(
            status="PREPARED_NOT_DISPATCHABLE",
            commit=commit,
            source_paths=sorted(hashes),
            inputs=bindings + resumed,
            resume=dict(
                unit_kind="spatial-exit-round-unit",
                ledger_path="inputs/verified-ledger.json",
                skip_verified=True,
                max_outcome_blind_retries=2,
            ),
            freeze_sha256=digest(packet / "FREEZE.json"),
            allocations=0,
        ),
    )
    return frozen


def verify_packet(packet: Path) -> tuple[RoundConfig, dict[str, Any]]:
    frozen = json.loads((packet / "FREEZE.json").read_text())
    packet_info = json.loads((packet / "PACKET.json").read_text())
    if digest(packet / "FREEZE.json") != packet_info["freeze_sha256"]:
        raise ValueError("round freeze hash mismatch")
    cfg = RoundConfig.from_dict(json.loads((packet / "config.json").read_text()))
    if not cfg.tiny and cfg.implementation_sha256 != identity(frozen["source_sha256"]):
        raise ValueError("round ledger source closure identity differs")
    if cfg.sha256 != frozen["config_sha256"] or cfg.namespace != frozen["namespace"]:
        raise ValueError("round config drift")
    for name, expected in frozen["inputs"].items():
        if digest(packet / name) != expected:
            raise ValueError(f"round input drift: {name}")
    for name, expected in frozen["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen source drift: {name}")
    for name in ("ledger", "seed_audit"):
        file = "verified-ledger.json" if name == "ledger" else "seed-audit.json"
        if digest(packet / file) != frozen[name + "_sha256"]:
            raise ValueError(f"{name} drift")
    return cfg, frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--resume-roots", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    cfg = RoundConfig.from_dict(json.loads(args.config.read_text()))
    if cfg.tiny:
        parser.error("use run --smoke for development fixtures")
    names = ["learner.pt", "reservoir.pt", "pilot-go.json", *CHECKPOINT_NAMES]
    names += ["initial-actor.pth"] if cfg.round_index == 1 else ["parent-endpoint.json"]
    freeze(
        args.packet,
        cfg,
        inputs={name: args.input_dir / name for name in names},
        registration=args.registration,
        inventory=args.inventory,
        ledger=json.loads(args.ledger.read_text()) if args.ledger else None,
        resume_roots=args.resume_roots,
    )


if __name__ == "__main__":
    main()
