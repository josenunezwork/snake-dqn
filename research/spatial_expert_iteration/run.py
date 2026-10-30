"""One pre-frozen expert-iteration round, or a tiny real Mac entrypoint smoke."""

from __future__ import annotations

import argparse
import io
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from research.sandbox_a_resident.ct_run import B8_S3_SHA256, _load_vector61
from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.model import SpatialStudent, export_checkpoint
from research.spatial_student_pilot.observation import build_observation, compact_cat
from research.spatial_student_pilot.prepare import CHECKPOINT_NAMES, OPPONENT_SHAS, digest
from research.spatial_student_pilot.resume import UnitStore, _publish_bytes
from research.spatial_student_pilot.run import Control, seal_file
from research.spatial_student_pilot.search import paired_gameplay
from src.model.ego2s_network import load_ego2s_checkpoint

from .config import RoundConfig
from .data import check_support_possible, collect, dataset, query_gate, select_anchors
from .fit import fit, initial_state
from .prepare import freeze, verify_packet, write_json
from .reservoir import capacity_forecast, reservoir_value
from .search import assess, label_roots, tensor_identity
from .units import HardwareUnitStore


def save_tree(path: Path, value: Any, max_bytes: int | None = None) -> None:
    stream = io.BytesIO()
    torch.save(value, stream)
    if max_bytes is not None and stream.tell() > max_bytes:
        raise ValueError("frozen natural reservoir byte cap exceeded")
    _publish_bytes(path, stream.getvalue())
    seal_file(path)


def networks_from(packet: Path, device: str) -> tuple[Any, dict[str, Any]]:
    return load_ego2s_checkpoint(str(packet / "b8-s3.pth"), device=device), {
        sha: _load_vector61(packet / name, torch.device(device))
        for name, sha in OPPONENT_SHAS.items()
    }


def round_pipeline(
    cfg: RoundConfig,
    store: UnitStore,
    *,
    initial: dict[str, Any],
    reservoir: list[dict[str, Any]],
    frozen: dict[str, Any],
    b8: Any,
    networks: dict[str, Any],
    device: str,
    control: Control,
) -> dict[str, Any]:
    """Advance one registered round; leave outcomes sealed for the Mac look.

    A program is a chain of these immutable round packets. Later packets import
    final learner/reservoir bytes; no loop extends itself after seeing assessment.
    """
    current = SpatialStudent().to(device).eval()
    current.load_state_dict(initial["model"])
    if not isinstance(store, HardwareUnitStore):
        store = HardwareUnitStore(store, device)
    collector = tensor_identity(current.state_dict())
    if collector != frozen.get("collector_sha256", collector):
        raise ValueError("warm-start differs from frozen round")
    continuation = collector if cfg.continuation_mode == "student" else B8_S3_SHA256
    if continuation != frozen["continuation_sha256"]:
        raise ValueError("continuation differs from frozen round")
    if cfg.continuation_mode == "student" and not frozen.get("continuation_audit", {}).get(
        "passed"
    ):
        raise ValueError("student continuation lacks paired teacher-degradation audit")
    capacity_forecast(cfg, reservoir)
    anchors = select_anchors(cfg, reservoir)
    batches = []
    for index, start in enumerate(range(0, cfg.worlds, cfg.world_batch_size)):
        unit_id = f"collect-{index:04d}"
        unit = store.get(unit_id)
        if unit is None:
            groups = []
            end = min(cfg.worlds, start + cfg.world_batch_size)
            # Separate prepared anchor and fresh start populations, both driven
            # by exactly the frozen current student. No successful-world refill.
            for anchored in (True, False):
                ids = [i for i in range(start, end) if (i < cfg.anchor_worlds) == anchored]
                if not ids:
                    continue
                selected_anchors = [anchors[i] for i in ids] if anchored else []
                groups.append(
                    collect(
                        cfg,
                        ids,
                        student=current,
                        student_sha=collector,
                        b8=b8,
                        networks=networks,
                        device=device,
                        anchors=selected_anchors,
                        stop=control.stop,
                        progress=control.progress,
                    )
                )
            groups = [
                dict(
                    partition="train",
                    world_ids=[i for g in groups for i in g["world_ids"]],
                    roots=[r for g in groups for r in g["roots"]],
                    demos=[r for g in groups for r in g["demos"]],
                    decisions=sum(g["decisions"] for g in groups),
                )
            ]
            assessment_ids = list(
                range(start, min(cfg.assessment_worlds, start + cfg.world_batch_size))
            )
            if assessment_ids:
                groups.append(
                    collect(
                        cfg,
                        assessment_ids,
                        student=current,
                        student_sha=collector,
                        b8=b8,
                        networks=networks,
                        device=device,
                        partition="assessment",
                        stop=control.stop,
                        progress=control.progress,
                    )
                )
            if control.stop():
                raise InterruptedError("collection completed after frozen deadline")
            store.publish(unit_id, groups)
            unit = groups
        batches.extend(unit)
        check_support_possible(cfg, batches)
    data = dataset(cfg, batches, reservoir)
    endpoint_reservoir = reservoir_value(cfg, reservoir, data["reservoir"])
    # Reused snapshots retain collector provenance, but gate disagreement is
    # recomputed with this round's frozen student, not the historical collector.
    with torch.inference_mode():
        for start in range(0, len(data["train"]), cfg.microbatch):
            rows = data["train"][start : start + cfg.microbatch]
            inputs = build_observation(compact_cat([r["compact"] for r in rows]), device=device)
            masks = torch.stack([r["mask"] for r in rows]).to(device)
            logits = current(*inputs)
            masked = logits.masked_fill(~masks, -torch.inf)
            actions = masked.argmax(1).cpu().tolist()
            margins = masked.topk(2, dim=1).values.diff(dim=1).neg()[:, 0].cpu().tolist()
            for row, action, values, margin in zip(rows, actions, logits.cpu().tolist(), margins):
                row["gate_student_action"] = action
                row["gate_student_logits"] = values
                row["gate_top2_margin"] = margin if margin != float("inf") else None
    gate = query_gate(data["train"], cfg)
    by_uid = {r["uid"]: r for r in gate}
    labels = []
    # Pool across the whole search_chunk (registered >=512), then fill large
    # physics slabs. Unit and physical batch size are separate frozen settings.
    for index, start in enumerate(range(0, len(data["train"]), cfg.search_chunk)):
        unit_id = f"search-{index:04d}"
        unit = store.get(unit_id)
        if unit is None:
            chunk = data["train"][start : start + cfg.search_chunk]
            selected = [
                dict(r, **{k: v for k, v in by_uid[r["uid"]].items() if k != "uid"})
                for r in chunk
                if by_uid[r["uid"]]["queried"]
            ]
            unit = dict(
                gate=[by_uid[r["uid"]] for r in chunk],
                labels=label_roots(
                    selected,
                    cfg=cfg,
                    student=current,
                    continuation_sha=continuation,
                    opponents_sha=frozen["opponents_sha256"],
                    b8=b8,
                    networks=networks,
                    device=device,
                    stop=control.stop,
                    progress=control.progress,
                ),
            )
            if control.stop():
                raise InterruptedError("search completed after frozen deadline")
            store.publish(unit_id, unit)
        if unit["gate"] != gate[start : start + cfg.search_chunk]:
            raise ValueError("resume gate differs from frozen population")
        labels.extend(unit["labels"])
    candidate, final = fit(
        cfg,
        data["demos"],
        labels,
        initial=initial,
        population=len(data["train"]),
        device=device,
        store=store,
        stop=control.stop,
        progress=control.progress,
    )
    for index, start in enumerate(range(0, len(data["assessment"]), cfg.search_chunk)):
        unit_id = f"assess-{index:04d}"
        if not store.has(unit_id):
            scored = assess(
                data["assessment"][start : start + cfg.search_chunk],
                cfg=cfg,
                current=current,
                candidate=candidate,
                continuation_sha=continuation,
                opponents_sha=frozen["opponents_sha256"],
                b8=b8,
                networks=networks,
                device=device,
                stop=control.stop,
                progress=control.progress,
            )
            if control.stop():
                raise InterruptedError("assessment completed after frozen deadline")
            store.publish(unit_id, scored)
    worlds = [
        dict(
            world_id=i,
            roster_index=i,
            mix=r.mix,
            seed=cfg.world_seed(i, "gameplay"),
            opp_seed=cfg.opp_seed(i, "gameplay"),
        )
        for i, r in enumerate(training_rosters(cfg.gameplay_worlds))
    ]
    for index, start in enumerate(range(0, len(worlds), cfg.world_batch_size)):
        unit_id = f"gameplay-{index:04d}"
        if not store.has(unit_id):
            scores = paired_gameplay(
                worlds[start : start + cfg.world_batch_size],
                candidate,
                cfg=cfg,
                b8=b8,
                networks=networks,
                device=device,
                stop=control.stop,
                progress=control.progress,
            )
            if control.stop():
                raise InterruptedError("gameplay completed after frozen deadline")
            store.publish(unit_id, scores)
    if control.stop():
        raise InterruptedError("round endpoint exceeded frozen deadline")
    save_tree(
        store.output / "reservoir.pt",
        endpoint_reservoir,
        max_bytes=cfg.reservoir_max_bytes,
    )
    save_tree(store.output / "learner.pt", final)
    actor = store.output / "final-actor.pth"
    export_checkpoint(
        candidate,
        actor,
        metadata=dict(
            namespace=cfg.namespace,
            round_index=cfg.round_index,
            config_sha256=cfg.sha256,
            continuation_sha256=continuation,
            quarantined=True,
            product_training_eligible=False,
            target_dynamics="development-v1-not-product-replayed",
        ),
    )
    seal_file(actor)
    return dict(
        status=(
            "SMOKE_COMPLETE_NO_SCIENTIFIC_RESULT" if cfg.tiny else "ENDPOINT_EXPORTED_NO_DECISION"
        ),
        program=cfg.program,
        round_index=cfg.round_index,
        namespace=cfg.namespace,
        config_sha256=cfg.sha256,
        candidate_sha256=digest(actor),
        learner_sha256=digest(store.output / "learner.pt"),
        reservoir_sha256=digest(store.output / "reservoir.pt"),
        units=store.list_ids(),
        queried_roots=len(labels),
        candidate_roots=len(data["train"]),
        heldout_parent_count=len({r["parent_id"] for r in data["assessment"]}),
        scientific_result=False,
        allocations=0,
        quarantine_search_targets=True,
        product_training_eligible=False,
        hardware=store.runtime,
    )


def smoke_packet(root: Path, checkpoints: Path) -> Path:
    """Synthetic tiny curriculum with actual simulator, actor, B8 and opponents."""
    if platform.system() != "Darwin":
        raise ValueError("entrypoint smoke is Mac only")
    cfg = RoundConfig.smoke()
    inputs = root / "fixture-inputs"
    inputs.mkdir(parents=True)
    torch.manual_seed(cfg.seed)
    save_tree(inputs / "learner.pt", initial_state(SpatialStudent(), cfg.seed))
    save_tree(inputs / "reservoir.pt", dict(schema="spatial-exit-reservoir/v1", roots=[]))
    write_json(inputs / "REGISTRATION.json", dict(status="SMOKE_ONLY", config_sha256=cfg.sha256))
    write_json(inputs / "inventory.json", dict(excluded_seeds=[], scope="tiny-fixture-only"))
    paths = {dest: checkpoints / name for dest, name in CHECKPOINT_NAMES.items()}
    # The same immutable canonical teachers as a scientific round, never zeros.
    from research.sandbox_a_resident.ct_run import B8_S3_SHA256

    for name, expected in {"b8-s3.pth": B8_S3_SHA256, **OPPONENT_SHAS}.items():
        if digest(paths[name]) != expected:
            raise ValueError("smoke checkpoint mismatch")
    paths.update({name: inputs / name for name in ("learner.pt", "reservoir.pt")})
    freeze(
        root / "packet",
        cfg,
        inputs=paths,
        registration=inputs / "REGISTRATION.json",
        inventory=inputs / "inventory.json",
    )
    return root / "packet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("every attempt requires a fresh absolute output")
    if args.smoke and (args.packet or not args.checkpoint_dir or args.device != "cpu"):
        parser.error("Mac smoke requires CPU, canonical checkpoint-dir, and no packet")
    if not args.smoke and not args.packet:
        parser.error("a frozen packet is required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    packet = smoke_packet(args.output, args.checkpoint_dir) if args.smoke else args.packet
    cfg, frozen = verify_packet(packet)
    if cfg.tiny != args.smoke:
        parser.error("development packet cannot enter registered runtime")
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    remaining = cfg.timeout_seconds - ledger["consumed_seconds"]
    if remaining <= 0:
        raise ValueError("cumulative round timeout exhausted")
    store = UnitStore(args.output / "exports", cfg, ledger, packet)
    control = Control(store.output, remaining)
    control.store = store
    try:
        b8, networks = networks_from(packet, args.device)
        initial = torch.load(packet / "learner.pt", map_location="cpu", weights_only=False)
        reservoir = torch.load(packet / "reservoir.pt", map_location="cpu", weights_only=False)[
            "roots"
        ]
        result = round_pipeline(
            cfg,
            store,
            initial=initial,
            reservoir=reservoir,
            frozen=frozen,
            b8=b8,
            networks=networks,
            device=args.device,
            control=control,
        )
        result.update(
            elapsed_seconds=time.monotonic() - control.started,
            freeze_sha256=digest(packet / "FREEZE.json"),
        )
        write_json(store.output / "summary.json", result)
        seal_file(store.output / "summary.json")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        write_json(
            store.output / "summary.json",
            dict(
                status="INCOMPLETE",
                reason=str(error),
                units=store.list_ids(),
                scientific_result=False,
                allocations=0,
            ),
        )
        raise
    finally:
        control.close()


if __name__ == "__main__":
    raise SystemExit(main())
