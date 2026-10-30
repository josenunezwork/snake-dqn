"""Actual producer integration and full optimizer/RNG resume equivalence."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256
from research.spatial_student_pilot.collect import collect_batch, select_dataset
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.fit import fit_student
from research.spatial_student_pilot.resume import UnitStore, build_ledger, empty_ledger
from research.spatial_student_pilot.run import Control, pipeline
from research.spatial_student_pilot.search import label_roots


class ZeroPolicy(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.zeros((args[0].shape[0], 6), device=args[0].device)


def test_real_pipeline_and_support_fail_closed(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    cfg = PilotConfig().development("preflight")
    store = UnitStore(tmp_path / "out", cfg, empty_ledger(cfg), tmp_path)
    control = Control(store.output, 60)
    zero = ZeroPolicy()
    networks = {FRP3_S12_SHA256: zero, CHAMPION_SHA256: zero}
    try:
        result = pipeline(cfg, store, b8=zero, networks=networks, device="cpu", control=control)
        assert result["status"] == "ENDPOINT_EXPORTED_NO_REMOTE_LOOK"
        assert (
            result["fit_timing"]["heldout_bc_roots"]
            == cfg.assessment_late + cfg.assessment_ordinary
        )
        assert len(store.list_ids()) == 19
        assert (store.output / "final-actor.pth.complete").is_file()
        batches = [store.get("collect-0000"), store.get("collect-0001")]
        data = select_dataset(cfg, batches)
        assert len(data["demos"]) == cfg.demos
        assert not (
            {r["world_id"] for r in data["demos"]}
            & {r["world_id"] for r in data["roots"] if r["partition"] == "assessment"}
        )
        with pytest.raises(ValueError, match="collection worlds"):
            select_dataset(cfg, batches[:1])
        control.store = store
        control.on_signal(0, None)
        assert (store.output / "sigusr1-marker.json.complete").is_file()
    finally:
        control.close()


def test_fit_restart_preserves_final_parameters_and_optimizer(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    cfg = replace(PilotConfig().development("preflight"), bc_updates=2, improvement_updates=2)
    zero = ZeroPolicy()
    networks = {FRP3_S12_SHA256: zero, CHAMPION_SHA256: zero}
    batches = [
        collect_batch(
            cfg,
            range(i, i + 3),
            b8=zero,
            networks=networks,
            device="cpu",
            stop=lambda: False,
            progress=lambda _: None,
        )
        for i in (0, 3)
    ]
    data = select_dataset(cfg, batches)
    train = [r for r in data["roots"] if r["partition"] == "train"]
    labels = label_roots(train, cfg=cfg, b8=zero, networks=networks, device="cpu")
    heldout = [r for r in data["roots"] if r["partition"] == "assessment"]
    full = UnitStore(tmp_path / "full", cfg, empty_ledger(cfg), tmp_path)
    expected, _ = fit_student(
        cfg,
        data["demos"],
        labels,
        device="cpu",
        store=full,
        stop=lambda: False,
        progress=lambda _: None,
        heldout=heldout,
    )
    first = UnitStore(tmp_path / "first", cfg, empty_ledger(cfg), tmp_path)
    progress = {"updates": 0}
    with pytest.raises(InterruptedError):
        fit_student(
            cfg,
            data["demos"],
            labels,
            device="cpu",
            store=first,
            stop=lambda: progress["updates"] >= 2,
            progress=lambda r: progress.update(r),
            heldout=heldout,
        )
    ledger = build_ledger(cfg, [first.output], elapsed_seconds=0)
    resumed = UnitStore(tmp_path / "resumed", cfg, ledger, first.output)
    actual, _ = fit_student(
        cfg,
        data["demos"],
        labels,
        device="cpu",
        store=resumed,
        stop=lambda: False,
        progress=lambda _: None,
        heldout=heldout,
    )
    assert all(
        torch.equal(value, actual.state_dict()[key]) for key, value in expected.state_dict().items()
    )
    assert full.get("fit-0000002")["heldout_bc"] == first.get("fit-0000002")["heldout_bc"]
    assert full.get("fit-0000002")["heldout_bc"]["total"] == len(heldout)
    baseline = UnitStore(tmp_path / "without-diagnostic", cfg, empty_ledger(cfg), tmp_path)
    unchanged, _ = fit_student(
        cfg,
        data["demos"],
        labels,
        device="cpu",
        store=baseline,
        stop=lambda: False,
        progress=lambda _: None,
    )
    assert all(
        torch.equal(value, unchanged.state_dict()[key])
        for key, value in expected.state_dict().items()
    )
    assert torch.equal(
        full.get("fit-0000004")["torch_rng"], baseline.get("fit-0000004")["torch_rng"]
    )
    a = full.get("fit-0000004")["optimizer"]
    b = resumed.get("fit-0000004")["optimizer"]
    assert a["param_groups"] == b["param_groups"]
    assert all(
        torch.equal(value, b["state"][key][field])
        for key, state in a["state"].items()
        for field, value in state.items()
    )


def test_scientific_configuration_and_partition_are_frozen() -> None:
    cfg = PilotConfig()
    assert cfg.partitions().count("assessment") == 256
    with pytest.raises(ValueError, match="differs from registration"):
        replace(cfg, horizon=4999)
    with pytest.raises(ValueError, match="allocation"):
        replace(cfg, allocation_index=True)
