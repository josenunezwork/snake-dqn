"""Late adoption regressions: exact science bytes, stage plans and sync controls."""

from __future__ import annotations

import io
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim
from research.spatial_student_pilot.batching import (
    capture_rows,
    grouped_q,
    observe_hero_bv8,
    restore_fanout,
)
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.fit import fit_student
from research.spatial_student_pilot.mechanics import PilotCTDeviceSim
from research.spatial_student_pilot.observation import build_observation, compact_from_sim
from research.spatial_student_pilot.resume import (
    UnitStore,
    build_ledger,
    empty_ledger,
    stage_unit_ids,
    upload_inputs,
)
from research.spatial_student_pilot.ruler_cells import cell_specs
from research.spatial_student_pilot.search import label_chunks, label_roots
from research.spatial_student_pilot.test_batching import FixedPolicy, exact, simulator
from research.spatial_student_pilot.test_search_readout import fixture_root


def test_pool_across_export_chunks_retains_chunk_freeze_and_results(monkeypatch: Any) -> None:
    root, cfg, b8 = fixture_root()
    roots = [dict(root, uid=root["uid"] + f"/replica-{i}") for i in range(4)]
    chunks = [roots[:2], roots[2:]]
    kwargs = dict(cfg=replace(cfg, branch_batch_size=48), b8=b8, networks={}, device="cpu")
    expected = [label_roots(chunk, **kwargs) for chunk in chunks]
    import research.spatial_student_pilot.search as search

    original, sizes = search._rollouts, []

    def observe(*args: Any, **kwargs: Any) -> Any:
        sizes.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "_rollouts", observe)
    actual = label_chunks(chunks, **kwargs)
    assert len(sizes) == 1 and sizes[0] == 2 * sum(sum(r["mask"]) for r in roots)
    for got_chunk, expected_chunk in zip(actual, expected):
        for got, want in zip(got_chunk, expected_chunk):
            for key in ("selection_mass", "selection_survival"):
                import numpy as np

                assert np.asarray(got[key]).tobytes() == np.asarray(want[key]).tobytes()
            exact(
                {
                    k: v
                    for k, v in got.items()
                    if not k.startswith("selection_m") and k != "selection_survival"
                },
                {
                    k: v
                    for k, v in want.items()
                    if not k.startswith("selection_m") and k != "selection_survival"
                },
            )


def test_bc_only_then_resume_exact_full_snapshot(tmp_path: Path) -> None:
    root, cfg, b8 = fixture_root()
    cfg = replace(cfg, bc_updates=2, improvement_updates=2, checkpoint_every=3)
    from research.sandbox_a_resident.ct_snapshots import (
        CompactWorldSnapshot,
        restore_world_snapshots_into,
    )

    sim, opp = simulator(cfg, [0])
    restore_world_snapshots_into([CompactWorldSnapshot.from_state_dict(root["snapshot"])], sim, opp)
    train = dict(
        root,
        partition="train",
        target_action=root["b8_action"],
        mask=torch.tensor(root["mask"]),
        compact=compact_from_sim(sim),
    )
    kwargs = dict(device="cpu", stop=lambda: False, progress=lambda _: None)
    full = UnitStore(tmp_path / "full", cfg, empty_ledger(cfg), tmp_path)
    fit_student(cfg, [train], [train], store=full, **kwargs)
    split = UnitStore(tmp_path / "split", cfg, empty_ledger(cfg), tmp_path)
    fit_student(cfg, [train], [], store=split, bc_only=True, **kwargs)
    assert split.has("fit-0000002") and not split.get("fit-0000002")["final"]
    fit_student(cfg, [train], [train], store=split, **kwargs)
    exact(split.get("fit-0000004"), full.get("fit-0000004"))


def test_padded_observation_and_async_bounds_exact() -> None:
    cfg = PilotConfig().development("preflight")
    sim, opp = simulator(cfg, list(range(6)))
    sim.prepare()
    # Ring/food holes and dead out-of-board heads are present in sparse padding.
    sim.food_present[:, 2:5] = False
    sim.alive[1, 0] = False
    sim.bodies[1, 0, sim.head_ptr[1, 0]] = -2
    r2_cache.bump(sim)
    expected = build_observation(compact_from_sim(sim))
    actual = build_observation(compact_from_sim(sim, padded=True), async_checks=True)
    exact(actual, expected)
    compact = compact_from_sim(sim, padded=True)
    compact["food_kind"][0, 0] = 1
    compact["food_cells"][0, 0] = -2
    with pytest.raises(RuntimeError, match="food lies outside"):
        build_observation(compact, async_checks=True)


def test_inference_batches_never_exceed_128() -> None:
    class ShapePolicy(FixedPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.sizes = []

        def forward(self, *args: Any) -> Any:
            self.sizes.append(len(args[0]))
            return super().forward(*args)

    net = ShapePolicy()
    q = grouped_q(net, (torch.zeros(1024, 3),), torch.ones(1024, 6, dtype=torch.bool))
    assert net.sizes == [128] * 8
    assert torch.equal(q, torch.arange(6).float().expand(1024, -1))


def test_controlled_vector_features_skip_scripted_slots(monkeypatch: Any) -> None:
    cfg = PilotConfig().development("preflight")
    sim, opp = simulator(cfg, list(range(6)))
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

    original, seen = DeviceVectorRuntime._features_rows, []

    def record(self: Any, sim: Any, rows: Any, **kwargs: Any) -> Any:
        seen.extend(rows.cpu().tolist())
        return original(self, sim, rows, **kwargs)

    monkeypatch.setattr(DeviceVectorRuntime, "_features_rows", record)
    sim.prepare()
    opp.actions(sim, observe_hero_bv8(sim, FixedPolicy())[-1])
    opp.after_step(sim)
    controlled = {tuple(row) for row in opp.runtime.controlled_rows.tolist()}
    assert set(map(tuple, seen)) == controlled
    assert len(controlled) < sim.E * sim.S


def test_fanout_restore_exact_full_buffers() -> None:
    from research.sandbox_a_resident.ct_snapshots import restore_world_snapshots_into

    cfg = PilotConfig().development("preflight")
    source, opp = simulator(cfg, [0, 1])
    source.prepare()
    _, snaps = capture_rows(source, opp, [0, 1], world_ids=[0, 1], tick=1, max_body_capacity=64)
    a, ao = simulator(cfg, [1, 0, 1, 1])
    b, bo = simulator(cfg, [1, 0, 1, 1])
    repeated = [snaps[i] for i in (1, 0, 1, 1)]
    restore_world_snapshots_into(repeated, a, ao)
    restore_fanout(repeated, b, bo)
    from research.spatial_student_pilot.test_batching import digest_rows

    assert digest_rows(a, ao) == digest_rows(b, bo)


@torch.inference_mode()
@pytest.mark.parametrize("v2", [False, True])
def test_ct_hoisting_exact_rng_food_deaths_respawn(v2: bool) -> None:
    cfg = PilotConfig().development("preflight")
    config = replace(cfg.sim_config(4), mechanics_version=2 if v2 else 1)
    old, new = [
        cls(config, device="cpu", train_mode=False, allow_respawn=True)
        for cls in (CTDeviceSim, PilotCTDeviceSim)
    ]
    for sim in (old, new):
        r2_cache.enable(sim, debug=True)
        sim.alive.fill_(False)
        sim.terminal_slots.fill_(False)
        sim.rng_key.copy_(torch.tensor([109, 211, 313, 419]))
        sim.rng_counter.zero_()
        sim.reset_fresh_worlds()
    for tick in range(45):
        # Include forced wall death and same-tick food encounters.
        for sim in (old, new):
            if tick % 7 == 0:
                sim.food_cells[:, 0] = sim.heads()[:, 0] + torch.tensor([1, 0])
                sim.food_present[:, 0] = True
                sim.corpse_food[:, 0] = tick % 2 == 0
                r2_cache.bump(sim)
            sim.prepare()
            sim.step(torch.full((4, 6), 1 if tick < 20 else 4))
        for key, value in old.__dict__.items():
            if isinstance(value, torch.Tensor):
                exact(getattr(new, key), value)
        assert old._prepared == new._prepared


def test_resume_packing_prunes_old_fit_and_completed_stages(tmp_path: Path) -> None:
    cfg = PilotConfig().development("preflight")
    first = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    names = [
        "collect-0000",
        "collect-0001",
        "fit-0000001",
        "fit-0000002",
        "search-0000",
        "aselect-0000",
        "ascore-0000",
        "assess-0000",
        "gameplay-0000",
    ]
    for name in names:
        first.publish(name, {"unit": name, "tensor": torch.arange(3)})
    ledger = build_ledger(cfg, [first.output], elapsed_seconds=0)
    assert stage_unit_ids(ledger, cfg) == {
        "collect-0000",
        "collect-0001",
        "fit-0000002",
    }
    with pytest.MonkeyPatch.context() as m:
        m.setattr(torch, "load", lambda *a, **k: pytest.fail("packing decoded an outcome"))
        a = upload_inputs(ledger, [first.output], cfg, pack_path=tmp_path / "a.tar")
        b = upload_inputs(ledger, [first.output], cfg, pack_path=tmp_path / "b.tar")
    assert len(a) == 1 and a[0]["sha256"] == b[0]["sha256"]
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    shutil.copyfile(a[0]["path"], uploaded / "resume-units.tar")
    resumed = UnitStore(tmp_path / "resume", cfg, ledger, uploaded)
    assert resumed.list_ids() == sorted(names)
    assert resumed.get("fit-0000002")["unit"] == "fit-0000002"
    assert resumed.has("assess-0000")
    with pytest.raises(ValueError, match="stage plan omitted"):
        resumed.get("fit-0000001")
    with pytest.raises(ValueError, match="recomputed"):
        resumed.publish("assess-0000", {})


@pytest.mark.parametrize("bad", ["traversal", "duplicate", "link", "plan", "payload"])
def test_resume_archive_refuses_corrupt_inventory(tmp_path: Path, bad: str) -> None:
    cfg = PilotConfig().development("preflight")
    first = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    first.publish("collect-0000", {"tensor": torch.ones(1)})
    ledger = build_ledger(cfg, [first.output], elapsed_seconds=0)
    upload_inputs(ledger, [first.output], cfg, pack_path=tmp_path / "good.tar")
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    with (
        tarfile.open(tmp_path / "good.tar") as source,
        tarfile.open(uploaded / "resume-units.tar", "w") as dest,
    ):
        for member in source.getmembers():
            raw = source.extractfile(member).read()
            if bad == "plan" and member.name == "resume-plan.json":
                raw = raw.replace(b"spatial-resume-pack/v1", b"spatial-resume-pack/v0")
            if bad == "payload" and member.name.endswith(".pt.gz"):
                raw = b"X" + raw[1:]
            member.size = len(raw)
            dest.addfile(member, io.BytesIO(raw))
        if bad in ("traversal", "duplicate", "link"):
            member = tarfile.TarInfo("../escape" if bad == "traversal" else "resume-plan.json")
            if bad == "link":
                member.type, member.linkname = tarfile.SYMTYPE, "/tmp/escape"
            dest.addfile(member)
    with pytest.raises(ValueError):
        UnitStore(tmp_path / "resume", cfg, ledger, uploaded)


def test_factorial_cells_cover_exact_cartesian_product() -> None:
    specs = cell_specs()
    assert len(specs) == 8
    assert {(r["regime"], r["worlds"], r["roster"]) for r in specs} == {
        (a, b, c)
        for a in ("early", "mature")
        for b in (128, 1024)
        for c in ("mixed", "homogeneous")
    }


@torch.inference_mode()
def test_corpse_span_hoist_wrapped_long_body_and_stale_padding() -> None:
    cfg = PilotConfig().development("preflight")
    old, new = [
        cls(cfg.sim_config(1), device="cpu", train_mode=False, allow_respawn=False)
        for cls in (CTDeviceSim, PilotCTDeviceSim)
    ]
    for sim in (old, new):
        sim.alive.zero_()
        sim.alive[:, 0] = True
        sim.length.fill_(1)
        sim.seg_count.fill_(1)
        sim.length[:, 0] = 33
        sim.seg_count[:, 0] = 33
        sim.head_ptr[:, 0] = 3
        sim.bodies.fill_(77)  # Irrelevant stale ring bytes must not become corpses.
        ring = (3 - torch.arange(33)) % sim.cap
        cells = torch.stack((torch.arange(33) % 12, torch.arange(33) // 12 + 3), 1)
        cells[0] = torch.tensor([sim.grid_w - 1, 3])
        sim.bodies[0, 0, ring] = cells
        sim.direction[:, 0] = 1
        sim._prepared = True
        sim.step(torch.ones(1, 6, dtype=torch.long))
    for key, value in old.__dict__.items():
        if isinstance(value, torch.Tensor):
            exact(getattr(new, key), value)
    assert old._last_done[0, 0]


def test_complete_pipeline_packed_resume_skips_sealed_work(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from research.spatial_student_pilot.run import Control, pipeline
    from research.spatial_student_pilot.test_batching import networks

    cfg = PilotConfig().development("preflight")
    first = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    control = Control(first.output, 60)
    kwargs = dict(b8=FixedPolicy(), networks=networks(), device="cpu", control=control)
    try:
        pipeline(cfg, first, **kwargs)
    finally:
        control.close()
    ledger = build_ledger(cfg, [first.output], elapsed_seconds=0)
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    upload_inputs(ledger, [first.output], cfg, pack_path=uploaded / "resume-units.tar")
    resumed = UnitStore(tmp_path / "resumed", cfg, ledger, uploaded)
    import research.spatial_student_pilot.search as search

    monkeypatch.setattr(search, "_rollouts", lambda *a, **k: pytest.fail("recomputed search"))
    control = Control(resumed.output, 60)
    try:
        pipeline(cfg, resumed, **dict(kwargs, control=control))
    finally:
        control.close()
    assert resumed.list_ids() == first.list_ids()
    assert list(resumed.output.rglob("*.pt.gz")) == []
    a = torch.load(first.output / "final-actor.pth", weights_only=False)
    b = torch.load(resumed.output / "final-actor.pth", weights_only=False)
    exact(a, b)


def test_factorial_runner_executes_all_diagnostic_paths_on_tiny_geometry(monkeypatch: Any) -> None:
    from types import SimpleNamespace

    from research.spatial_student_pilot import ruler_cells
    from research.spatial_student_pilot.test_batching import networks

    root, tiny, _ = fixture_root()
    cfg = replace(
        tiny, mode="ruler", namespace=PilotConfig().development("ruler").namespace, tiny=False
    )
    proxy = SimpleNamespace(
        **cfg.to_dict(),
        sim_config=tiny.sim_config,
        partitions=cfg.partitions,
        world_seed=cfg.world_seed,
        opp_seed=cfg.opp_seed,
    )
    fixture = dict(root, roster_index=0)
    # Exercise the eight paths cheaply; production cell_specs stays E128/E1024.
    monkeypatch.setattr(
        ruler_cells,
        "cell_specs",
        lambda: [dict(s, worlds=2 if s["worlds"] == 128 else 4) for s in cell_specs()],
    )
    rows = ruler_cells.factorial_ruler(
        proxy,
        [fixture],
        b8=FixedPolicy(),
        networks=networks(),
        device="cpu",
        stop=lambda: False,
        progress=lambda _: None,
    )
    assert len(rows) == 8
    for row in rows:
        assert row["ticks"] == 8 and len(row["measured_tick_seconds"]) == 6
        assert set(row["stage_diagnostics"]["seconds_by_stage"]) >= {"hero", "opponents", "prepare"}
        assert row["stage_diagnostics"]["sync_capable_ops_by_stage"]
        assert not row["scientific_result"]
