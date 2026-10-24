"""No-look tests for the R5 archive census freeze and the Look R-A command packet.

Synthetic worlds only. The census is checked against the *real* held-out
evaluator's ancestry/receipt gates, stopped at its tensor-cache boundary (no
metric computed). The R-A lock is checked against the real
``heldout.preflight_look`` that ``funding.main`` uses, on synthetic checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.bl_ppo.exports import digest, export_bytes, export_json, export_torch
from research.bl_residual import f4_archive_census as census_tool
from research.bl_residual import f4_d1024_package as package
from research.bl_residual import f4_ra_prepare as ra
from research.bl_residual.data import world_fold
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import canonical_sha
from tests import f4_d1024_fixtures as fx


class MetricsNotComputed(Exception):
    """Raised at heldout.evaluate's tensor-cache boundary."""


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return fx.build_world(tmp_path_factory.mktemp("f4-census-world"))


@pytest.fixture()
def packs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world: dict) -> dict:
    calls: dict = {}
    fx.patch_package(monkeypatch, world, calls)
    d256_pack = fx.build_d256_pack(tmp_path, world)
    staging = fx.build_staging(tmp_path, d256_pack)
    ledgers = tmp_path / "ledgers"
    for name in ("d256", "d1024"):
        export_json(ledgers / f"{name}.json", {"synthetic_ledger": name})
    package._prepare(
        tmp_path / "d1024-pack",
        d256_ledger=ledgers / "d256.json",
        d1024_ledger=ledgers / "d1024.json",
        d1024_union_sha="d" * 64,
        staging=staging,
        commit="HEAD",
        preflight_receipt=tmp_path / "preflight.json",
    )
    packs = {"d256": d256_pack, "d1024": tmp_path / "d1024-pack", "ledgers": ledgers}
    packs["repo"] = _publish_fits(monkeypatch, tmp_path, packs)
    return packs


def _publish_fits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, packs: dict) -> Path:
    """Synthetic immutable fits and their published fit-phase ledgers in a fake repo."""
    from research.bl_residual import f4_d256_package as d256
    from research.bl_residual.protocol import build_ledger

    repo = tmp_path / "repo"
    (repo / "research/bl_residual").mkdir(parents=True)
    (repo / "research/bl_residual/F4_REGISTRATION.md").write_bytes(
        (fx.ROOT / "research/bl_residual/F4_REGISTRATION.md").read_bytes()
    )
    monkeypatch.setattr(package, "ROOT", repo)
    for pack, dose, run in (
        (packs["d256"], 256, "d256-output-l4-01"),
        (packs["d1024"], 1024, package.MAIN_OUTPUT),
    ):
        control = json.loads((pack / "inputs/control.json").read_text())
        fit = pack / run / "exports/fit"
        export_json(
            fit / "allocation.json",
            {"namespace": control["namespace"], "allocation": control["allocation"]},
        )
        _checkpoint(fit / f"fit-{(256, 1024).index(dose):04d}.pth", control, dose)
        publication = d256.FIT_PUBLICATION_PATH if dose == 256 else package.FIT_PUBLICATION_PATH
        build_ledger([fit], repo / publication, control["namespace"], control["allocation"])

    def origin(argv: list[str], **_kwargs) -> bytes:
        return (repo / argv[-1].split(":", 1)[1]).read_bytes()

    monkeypatch.setattr(package.subprocess, "check_output", origin)
    return repo


def _census(packs: dict, out: Path) -> dict:
    return census_tool._census(
        out,
        d256_ledger=packs["ledgers"] / "d256.json",
        d1024_ledger=packs["ledgers"] / "d1024.json",
        d1024_union_sha="d" * 64,
        d256_pack_inputs=packs["d256"] / "inputs",
        d1024_pack_inputs=packs["d1024"] / "inputs",
    )


def test_census_freezes_every_fold_zero_ordinary_row_and_whole_reserve(
    packs: dict, world: dict, tmp_path: Path
) -> None:
    result = _census(packs, tmp_path / "census")
    receipt = json.loads((tmp_path / "census" / census_tool.RECEIPT).read_text())
    allocation = world["prior"]
    reserve = set(allocation["doses"]["256"]["heldout_lives"])
    catalog = {row["life_id"]: row for row in allocation["catalog"]}
    reserve_max = max(catalog[life]["allocation_index"] for life in reserve)
    eligible = {
        life
        for life, row in catalog.items()
        if world_fold(row["parent_world"]) == 0 and row["allocation_index"] <= reserve_max
    }
    assert receipt["eligible_life_ids"] == sorted(eligible)
    assert receipt["reserve_max_allocation_index"] == reserve_max
    assert receipt["counts"]["eligible_non_rescue_lives"] > 0
    with np.load(tmp_path / "census" / census_tool.TABLE, allow_pickle=False) as values:
        table = {key: values[key] for key in values.files}
    # Non-reserve rescue-bearing fold-zero lives keep their ordinary rows only.
    present = set(table["life_id"].tolist())
    non_reserve = {
        life for life in eligible if life not in reserve and catalog[life]["rescue_bearing"]
    }
    assert reserve <= present
    assert set(table["life_id"][~table["ordinary"]].tolist()) <= reserve
    assert not (non_reserve & set(table["life_id"][~table["ordinary"]].tolist()))
    assert receipt["natural_row_ids"] == sorted(table["row_id"][table["natural_stream"]].tolist())
    assert (
        result["table_sha256"]
        == receipt["table_sha256"]
        == digest(tmp_path / "census" / census_tool.TABLE)
    )
    assert "metrics" not in receipt and receipt["outcome_reporting"] == "none"
    with pytest.raises(FileExistsError):
        _census(packs, tmp_path / "census")


@pytest.mark.parametrize("dose", [256, 1024])
def test_census_passes_the_real_heldout_ancestry_and_receipt_gates(
    monkeypatch: pytest.MonkeyPatch, packs: dict, tmp_path: Path, dose: int
) -> None:
    from research.bl_residual import heldout

    _census(packs, tmp_path / "census")
    inputs = packs["d256" if dose == 256 else "d1024"] / "inputs"
    checkpoint = tmp_path / "fit.pth"
    torch.save({"meta": {"dose": dose}}, checkpoint)
    preflight = {"dose": dose, "checkpoint_sha256": digest(checkpoint)}
    monkeypatch.setattr(heldout, "preflight_look", lambda *_: preflight)
    claim = {"schema": "bl-residual-r-a-lock/v1", "lock_root": str(tmp_path.resolve())}
    claim[f"d{dose}"] = preflight
    lock = tmp_path / f"F4_R_A_{canonical_sha(claim)}.lock.json"
    lock.write_text(json.dumps(claim))
    monkeypatch.setattr(heldout, "B8_SHA", digest(inputs / "b8.pth"))
    monkeypatch.setattr(heldout, "load_ego2s_checkpoint", lambda *_: object())

    def stop(*_args, **_kwargs):
        raise MetricsNotComputed

    monkeypatch.setattr(heldout, "tensor_cache", stop)
    monkeypatch.setattr(heldout, "heldout_metrics", lambda *_a, **_k: pytest.fail("metrics"))
    with pytest.raises(MetricsNotComputed):
        heldout.evaluate(
            checkpoint,
            inputs / "b8.pth",
            tmp_path / "census" / census_tool.TABLE,
            allocation_path=inputs / "life-allocation.json",
            control_path=inputs / "control.json",
            approval_path=inputs / "judgment.json",
            archive_receipt=tmp_path / "census" / census_tool.RECEIPT,
            joint_claim_path=lock,
        )


def test_census_refuses_unbound_allocation_before_label_io(
    monkeypatch: pytest.MonkeyPatch, packs: dict, tmp_path: Path
) -> None:
    from research.bl_residual import f4_d256_package as d256

    target = packs["d1024"] / "inputs/life-allocation.json"
    target.chmod(0o600)
    target.write_text(target.read_text().replace("S-gate", "S-gate", 1) + " ")
    monkeypatch.setattr(d256, "_life_catalog", lambda *_: pytest.fail("labels opened"))
    with pytest.raises(ValueError, match="not bound"):
        _census(packs, tmp_path / "census")
    assert not (tmp_path / "census").exists()


def test_census_holds_when_allocations_disagree_on_the_archive(
    monkeypatch: pytest.MonkeyPatch, packs: dict, tmp_path: Path
) -> None:
    real = census_tool.archive_identity
    seen = []

    def shifted(allocation: dict) -> dict:
        value = real(allocation)
        seen.append(value)
        if len(seen) == 2:
            value = {**value, "eligible_life_ids": value["eligible_life_ids"][1:]}
        return value

    monkeypatch.setattr(census_tool, "archive_identity", shifted)
    with pytest.raises(ValueError, match="disagree"):
        _census(packs, tmp_path / "census")


def test_census_refuses_before_label_io_until_d1024_fit_is_published(
    monkeypatch: pytest.MonkeyPatch, packs: dict, tmp_path: Path
) -> None:
    from research.bl_residual import f4_d256_package as d256

    publication = packs["repo"] / package.FIT_PUBLICATION_PATH
    for suffix in ("", ".sha256", ".complete"):
        Path(str(publication) + suffix).unlink()
    monkeypatch.setattr(d256, "_life_catalog", lambda *_: pytest.fail("labels opened"))
    with pytest.raises(FileNotFoundError):
        _census(packs, tmp_path / "census")
    assert not (tmp_path / "census").exists()


def _checkpoint(path: Path, control: dict, dose: int) -> str:
    return export_torch(
        path,
        {
            "meta": {
                "schema": "bl-residual-fit/v1",
                "synthetic": False,
                "dose": dose,
                "index": (256, 1024, 4096).index(dose),
                "updates": FitRecipe().updates,
                "allocation": control["allocation"],
                "namespace": control["namespace"],
                "recipe_sha256": control["recipe_sha256"],
            }
        },
    )["sha256"]


@pytest.fixture()
def ra_world(monkeypatch: pytest.MonkeyPatch, packs: dict, tmp_path: Path) -> dict:
    """Published fits (from ``packs``) plus a published census in the fake repo."""
    monkeypatch.setattr(ra, "lock_search_roots", lambda: [tmp_path / "f4-tree"])
    census_dir = tmp_path / "census"
    _census(packs, census_dir)
    export_bytes(
        packs["repo"] / census_tool.PUBLICATION_PATH,
        (census_dir / census_tool.RECEIPT).read_bytes(),
    )
    return {"census": census_dir, **packs}


def _prepare(ra_world: dict, tmp_path: Path) -> dict:
    return ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=tmp_path / "r-a/funding-look.json",
        packet_out=tmp_path / "r-a/packet.json",
        pin="e" * 40,
    )


def test_ra_packet_lock_equals_funding_main_preflight_identity(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    from research.bl_residual import heldout

    result = _prepare(ra_world, tmp_path)
    packet = json.loads((tmp_path / "r-a/packet.json").read_text())
    assert packet["status"] == "PREPARED_NOT_RUN" and packet["f2_receipt"] is None
    assert "--f2-receipt" not in packet["argv"]
    argv = packet["argv"]

    def arg(name: str) -> Path:
        return Path(argv[argv.index(name) + 1])

    # Reproduce funding.main's own lock derivation with the real preflight.
    left = heldout.preflight_look(
        arg("--d256-checkpoint"),
        arg("--d256-allocation"),
        arg("--d256-control"),
        arg("--approval"),
        arg("--archive-receipt"),
    )
    right = heldout.preflight_look(
        arg("--d1024-checkpoint"),
        arg("--d1024-allocation"),
        arg("--d1024-control"),
        arg("--approval"),
        arg("--archive-receipt"),
    )
    lock = {
        "schema": "bl-residual-r-a-lock/v1",
        "lock_root": str(arg("--d256-allocation").parent.resolve()),
        "d256": left,
        "d1024": right,
    }
    expected = arg("--d256-allocation").parent / f"F4_R_A_{canonical_sha(lock)}.lock.json"
    assert packet["lock"] == lock
    assert arg("--look-lock") == expected == Path(result["lock_path"])
    assert arg("--d256-table") == arg("--d1024-table")
    assert not expected.exists()


def test_ra_prepare_holds_on_existing_lock_or_unpublished_census(
    ra_world: dict, tmp_path: Path
) -> None:
    lock_root = ra_world["d256"] / "inputs"
    (lock_root / "F4_R_A_existing.lock.json").write_text("{}")
    with pytest.raises(ValueError, match="once-only"):
        _prepare(ra_world, tmp_path)
    (lock_root / "F4_R_A_existing.lock.json").unlink()
    published = ra_world["repo"] / census_tool.PUBLICATION_PATH
    published.chmod(0o600)
    published.write_text("{}\n")
    with pytest.raises(ValueError):
        _prepare(ra_world, tmp_path)
    assert not (tmp_path / "r-a/packet.json").exists()


def test_ra_prepare_holds_when_checkpoint_is_not_the_published_unit(
    ra_world: dict, tmp_path: Path
) -> None:
    fit = ra_world["d1024"] / package.MAIN_OUTPUT / "exports/fit"
    target = fit / "fit-0001.pth"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(target) + suffix).chmod(0o600)
        Path(str(target) + suffix).unlink()
    control = json.loads((ra_world["d1024"] / "inputs/control.json").read_text())
    _checkpoint(target, {**control, "recipe_sha256": "0" * 64}, 1024)
    with pytest.raises(ValueError, match="published fit unit"):
        _prepare(ra_world, tmp_path)


def test_ra_prepare_refuses_any_f4_lock_not_only_the_derived_path(
    ra_world: dict, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "f4-tree/other-pack/inputs" / ("F4_R_A_" + "0" * 64 + ".lock.json")
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("{}")
    with pytest.raises(ValueError, match="once-only"):
        _prepare(ra_world, tmp_path)
    assert not (tmp_path / "r-a/packet.json").exists()


def test_ra_command_is_cwd_independent_and_refuses_any_f4_lock(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _prepare(ra_world, tmp_path)
    source = tmp_path / "source"
    (source / "tools/ops").mkdir(parents=True)
    (source / "tools/ops/dispatch.sh").write_text("#!/bin/sh\n")
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    packet = Path("../r-a/packet.json")
    line = ra.command(packet, Path("../source"), "e" * 40)
    assert str(source.resolve() / "tools/ops/dispatch.sh") in line
    assert "research.bl_residual.funding" in line and "--f2-receipt" not in line
    with pytest.raises(ValueError, match="RA_PIN"):
        ra.command(packet, Path("../source"), "f" * 40)
    stray = tmp_path / "f4-tree/x/F4_R_A_stray.lock.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}")
    with pytest.raises(ValueError, match="once-only"):
        ra.command(packet, Path("../source"), "e" * 40)
    stray.unlink()
    (tmp_path / "r-a/funding-look.json").write_text("{}")
    with pytest.raises(ValueError, match="once-only"):
        ra.command(packet, Path("../source"), "e" * 40)
