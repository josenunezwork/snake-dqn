"""No-look tests for the R5 archive census freeze and the Look R-A command packet.

Synthetic worlds only. The census is checked against the *real* held-out
evaluator's ancestry/receipt gates, stopped at its tensor-cache boundary (no
metric computed). The R-A lock is checked against the real
``heldout.preflight_look`` that ``funding.main`` uses, on synthetic checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from research.bl_ppo.exports import digest, export_bytes, export_json, export_torch
from research.bl_residual import f4_archive_census as census_tool
from research.bl_residual import f4_d1024_package as package
from research.bl_residual import f4_ra_prepare as ra
from research.bl_residual import funding, ra_guard
from research.bl_residual.data import world_fold
from research.bl_residual.fit import FitRecipe
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
        tmp_path / "d1024-pack-04",
        d256_ledger=ledgers / "d256.json",
        d1024_ledger=ledgers / "d1024.json",
        d1024_union_sha="d" * 64,
        staging=staging,
        commit="HEAD",
        preflight_receipt=tmp_path / "preflight.json",
        attempt="04",
    )
    packs = {"d256": d256_pack, "d1024": tmp_path / "d1024-pack-04", "ledgers": ledgers}
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
        (packs["d1024"], 1024, package.output_names("04")[0]),
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
        if dose == 1024:
            export_json(pack / "main.json", {"output_dir": str(pack / run), "commit": "c" * 40})
            export_json(pack / run / "receipt.json", {"synthetic": "supervisor"})
            export_json(fit / "receipt.json", {"synthetic": "producer"})
            paths = {
                "manifest": pack / "main.json",
                "supervisor_receipt": pack / run / "receipt.json",
                "producer_receipt": fit / "receipt.json",
            }
            export_json(
                repo / package.FIT_BINDING_PATH,
                {
                    "schema": "bl-residual-metadata-fit-binding/v1",
                    "fit_pin": "c" * 40,
                    "pack": str(pack),
                    "phase": publication,
                    "phase_sha256": digest(repo / publication),
                    "files": {
                        name: {
                            "path": str(path),
                            "size": path.stat().st_size,
                            "sha256": digest(path),
                        }
                        for name, path in paths.items()
                    },
                },
            )

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
    monkeypatch.setattr(ra_guard, "CLAIM_ROOT", tmp_path / "claims")
    reserve = {"series_id": "synthetic", "reserve_allocation_sha256": "a" * 64}
    claim = ra_guard.lock_identity(reserve, preflight, preflight)
    lock = ra_guard.claim_path(reserve)
    export_json(lock, claim)
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
    monkeypatch.setattr(ra_guard, "CLAIM_ROOT", tmp_path / "authoritative-claims")
    monkeypatch.setattr(ra_guard, "D4096_CONTROL", tmp_path / "canonical-d4096/control.json")
    monkeypatch.setattr(ra_guard, "source_identity", lambda pin: {"synthetic-source": pin})
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    monkeypatch.setattr(
        ra_guard,
        "trusted_time",
        lambda: {"epoch": ra_guard.now_epoch(), "date": "synthetic server Date"},
    )
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
    assert packet["invocation"]["--d1024-checkpoint"] == str(
        ra_world["d1024"] / "d1024-output-l4-04/exports/fit/fit-0001.pth"
    )
    assert not (ra_world["d1024"] / "d1024-output-l4-01").exists()
    assert Path(packet["argv"][packet["argv"].index("--command-packet") + 1]) == (
        tmp_path / "r-a/packet.json"
    )
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
    reserve = ra_guard.reserve_identity(
        arg("--d256-control"),
        arg("--d1024-control"),
        arg("--d256-allocation"),
        arg("--d1024-allocation"),
    )
    lock = ra_guard.lock_identity(reserve, left, right)
    expected = ra_guard.claim_path(reserve)
    assert packet["lock"] == lock
    assert arg("--look-lock") == expected == Path(result["lock_path"])
    assert arg("--d256-table") == arg("--d1024-table")
    assert not expected.exists()


def test_ra_prepare_binds_only_the_verified_f2_receipt(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    receipt = tmp_path / "f2-disposition.json"
    binding = {
        "path": str(receipt),
        "sha256": "a" * 64,
        "published_ref": "f2-input-ledger-commit",
        "source_sha256": "b" * 64,
        "publication_ref": "research/bl_residual/ledger/f2-result-20261024.json",
    }
    seen: list[Path] = []

    def verified(path: Path) -> dict:
        seen.append(path)
        return binding

    monkeypatch.setattr(ra_guard, "receipt_binding", verified)
    export_json(receipt, {"synthetic": True})
    monkeypatch.setattr(
        ra_guard,
        "publication_evidence",
        lambda _: {
            "commit": "a" * 40,
            "published_epoch": ra_guard.CUTOFF - 1,
        },
    )
    ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=tmp_path / "r-a/funding-look.json",
        packet_out=tmp_path / "r-a/packet.json",
        pin="e" * 40,
        f2_receipt=receipt,
    )
    packet = json.loads((tmp_path / "r-a/packet.json").read_text())
    assert seen == [receipt]
    assert packet["f2_branch"] == "VERIFIED_PUBLISHED_F2_RECEIPT"
    assert packet["f2_receipt"] == binding
    argv = packet["argv"]
    assert Path(argv[argv.index("--f2-receipt") + 1]) == receipt


def _packet(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "r-a/packet.json").read_text())


def _invoke(monkeypatch: pytest.MonkeyPatch, packet: dict, **changes: str) -> None:
    argv = packet["argv"][3:].copy()
    for name, value in changes.items():
        option = "--" + name.replace("_", "-")
        if option in argv:
            argv[argv.index(option) + 1] = value
        else:
            argv.extend((option, value))
    monkeypatch.setattr(sys, "argv", ["funding.py", *argv])
    funding.main()


def test_funding_refuses_a_receipt_different_from_the_sealed_packet_before_preflight(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _prepare(ra_world, tmp_path)
    receipt = tmp_path / "other-f2.json"
    actual = export_json(receipt, {"synthetic": "other"})
    monkeypatch.setattr(ra_guard, "receipt_binding", lambda _: actual)
    monkeypatch.setattr(funding, "preflight_look", lambda *_: pytest.fail("preflight opened"))
    with pytest.raises(ValueError, match="F2 binding differs"):
        _invoke(monkeypatch, _packet(tmp_path), f2_receipt=str(receipt))


def _f2_packet(
    monkeypatch: pytest.MonkeyPatch,
    ra_world: dict,
    tmp_path: Path,
    *,
    publication_epoch: float | None = None,
    started_epoch: float | None = None,
) -> dict:
    receipt = tmp_path / "f2-disposition.json"
    value = {
        "schema": "bl-residual-f2-disposition/v1",
        "volume_struck": False,
        "published": True,
        "source_sha256": "a" * 64,
        "published_ref": "a" * 40,
    }
    exported = export_json(receipt, value)
    binding = {"path": str(receipt), "sha256": exported["sha256"], "publication_ref": "b" * 40}
    monkeypatch.setattr(ra_guard, "receipt_binding", lambda _: binding)
    monkeypatch.setattr(funding.f2, "verify_published_receipt", lambda _: binding)
    monkeypatch.setattr(
        ra_guard,
        "publication_evidence",
        lambda _: {
            "commit": "b" * 40,
            "published_epoch": (
                ra_guard.CUTOFF - 1 if publication_epoch is None else publication_epoch
            ),
            "ref": "origin/main",
            "receipt_sha256": binding["sha256"],
        },
    )
    start = None
    if started_epoch is not None:
        from tests.test_bl_residual_ra_timing import _authoritative

        start = _authoritative(ra_guard.D4096_CONTROL.parent, monkeypatch, started_epoch)
    ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=tmp_path / "r-a/funding-look.json",
        packet_out=tmp_path / "r-a/packet.json",
        pin="e" * 40,
        f2_receipt=receipt,
        d4096_start_receipt=start,
    )
    return value


def test_funding_uses_f2_bytes_loaded_before_heldout_look(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    original = _f2_packet(monkeypatch, ra_world, tmp_path)
    packet = _packet(tmp_path)
    seen = []
    calls = []

    def heldout(*_args: object, **_kwargs: object) -> dict:
        calls.append(_kwargs)
        Path(packet["f2_receipt"]["path"]).write_text('{"substituted":true}')
        return {"synthetic": "look"}

    monkeypatch.setattr(funding, "evaluate_heldout", heldout)
    monkeypatch.setattr(
        funding, "evaluate", lambda _l, _r, value, **kw: seen.append((value, kw)) or {}
    )
    _invoke(monkeypatch, packet)
    assert len(calls) == 2 and seen[0][0] == original
    assert seen[0][1]["f2_timing_eligible"] is True
    assert Path(packet["lock_path"]).exists()


@pytest.mark.parametrize("change", ["relocation", "output", "lock", "source", "input"])
def test_funding_refuses_stale_or_relocated_command_before_any_look(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path, change: str
) -> None:
    import shutil

    _prepare(ra_world, tmp_path)
    packet = _packet(tmp_path)
    monkeypatch.setattr(
        funding, "evaluate_heldout", lambda *_a, **_kw: pytest.fail("held-out opened")
    )
    changes = {}
    if change == "relocation":
        source = ra_world["d256"] / "inputs/life-allocation.json"
        target = tmp_path / "relocated/life-allocation.json"
        target.parent.mkdir()
        for suffix in ("", ".sha256", ".complete"):
            shutil.copyfile(str(source) + suffix, str(target) + suffix)
        changes["d256_allocation"] = str(target)
    elif change == "output":
        changes["out"] = str(tmp_path / "other.json")
    elif change == "lock":
        changes["look_lock"] = str(tmp_path / "other.lock.json")
    elif change == "source":
        changes["source_pin"] = "f" * 40
    else:
        table = ra_world["census"] / census_tool.TABLE
        table.chmod(0o600)
        table.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="invocation|identities"):
        _invoke(monkeypatch, packet, **changes)
    assert not Path(packet["lock_path"]).exists()


def test_funding_competing_invocation_gets_one_claim_and_two_callbacks(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _prepare(ra_world, tmp_path)
    packet = _packet(tmp_path)
    calls = []

    def heldout(*_a, **_kw):
        calls.append(_kw)
        if len(calls) == 1:
            # Invoke again while the first command owns its atomic claim and has
            # no output yet: the competitor must not enter either held-out look.
            with pytest.raises(FileExistsError):
                _invoke(monkeypatch, packet)
        return {}

    monkeypatch.setattr(funding, "evaluate_heldout", heldout)
    monkeypatch.setattr(funding, "evaluate", lambda *_a, **_kw: {})
    _invoke(monkeypatch, packet)
    assert len(calls) == 2


def test_post_ra_f2_replacement_cannot_open_another_look(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _prepare(ra_world, tmp_path)
    packet = _packet(tmp_path)
    calls = []
    monkeypatch.setattr(funding, "evaluate_heldout", lambda *_a, **_kw: calls.append(1) or {})
    monkeypatch.setattr(funding, "evaluate", lambda *_a, **_kw: {})
    _invoke(monkeypatch, packet)
    # A fresh packet/output, even with a subsequently published F2, cannot
    # replace the already spent series/reserve claim.
    with pytest.raises(ValueError, match="once-only"):
        ra.prepare(
            d256_pack=ra_world["d256"],
            d1024_pack=ra_world["d1024"],
            census_dir=ra_world["census"],
            funding_out=tmp_path / "replacement/out.json",
            packet_out=tmp_path / "replacement/packet.json",
            pin="e" * 40,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("publication_offset,eligible", [(-1, True), (0, False), (1, False)])
def test_funding_rechecks_cutoff_before_claim(
    monkeypatch: pytest.MonkeyPatch,
    ra_world: dict,
    tmp_path: Path,
    publication_offset: int,
    eligible: bool,
) -> None:
    _f2_packet(
        monkeypatch,
        ra_world,
        tmp_path,
        publication_epoch=ra_guard.CUTOFF + publication_offset,
        started_epoch=ra_guard.START_DEADLINE,
    )
    from tests.test_bl_residual_funding_consumer import _funding_look

    packet = _packet(tmp_path)

    def heldout(checkpoint: Path, *_a, **kw):
        dose = 256 if "d256" in str(checkpoint) else 1024
        return _funding_look(
            dose=dose,
            checkpoint_sha256=digest(checkpoint),
            identity=packet["lock"]["reserve"],
            allocation=kw["allocation_path"],
            retained_rows=1,
        )

    monkeypatch.setattr(funding, "evaluate_heldout", heldout)
    _invoke(monkeypatch, packet)
    result = json.loads(Path(packet["funding_output"]).read_text())
    # Both checkpoints retain equally: only a timely F2 non-strike can fund.
    assert result["verdict"] == ("PASS" if eligible else "STOP_FUNDING")

    claim = json.loads(Path(packet["lock_path"]).read_text())
    assert claim["f2_timing_eligible"] is eligible
    assert claim["timing"]["branch"] == "WAIT_FOR_F2_OR_CUTOFF"


def test_funding_waits_without_claim_and_refuses_changed_timing_evidence(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _f2_packet(
        monkeypatch,
        ra_world,
        tmp_path,
        publication_epoch=ra_guard.CUTOFF - 1,
        started_epoch=ra_guard.START_DEADLINE,
    )
    packet = _packet(tmp_path)
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF - 2)
    monkeypatch.setattr(
        funding, "evaluate_heldout", lambda *_a, **_kw: pytest.fail("heldout opened")
    )
    with pytest.raises(ValueError, match="waits"):
        _invoke(monkeypatch, packet)
    assert not Path(packet["lock_path"]).exists()
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    monkeypatch.setattr(
        ra_guard,
        "publication_evidence",
        lambda _: {"commit": "c" * 40, "published_epoch": ra_guard.CUTOFF + 1},
    )
    with pytest.raises(ValueError, match="timing evidence"):
        _invoke(monkeypatch, packet)
    assert not Path(packet["lock_path"]).exists()


def test_two_prepared_packets_share_one_authoritative_reserve_claim(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _prepare(ra_world, tmp_path)
    second = tmp_path / "second"
    ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=second / "funding.json",
        packet_out=second / "packet.json",
        pin="e" * 40,
    )
    packet2 = json.loads((second / "packet.json").read_text())
    assert packet2["lock_path"] == _packet(tmp_path)["lock_path"]
    calls = []
    monkeypatch.setattr(funding, "evaluate_heldout", lambda *_a, **_kw: calls.append(1) or {})
    monkeypatch.setattr(funding, "evaluate", lambda *_a, **_kw: {})
    _invoke(monkeypatch, _packet(tmp_path))
    with pytest.raises(FileExistsError):
        _invoke(monkeypatch, packet2)
    assert len(calls) == 2


def test_full_funding_entrypoint_passes_unchanged_i3_consumer(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    from research.bl_residual import job
    from research.bl_residual.protocol import B8_SHA
    from tests import test_bl_residual_collection_receipts as i3
    from tests.test_bl_residual_funding_consumer import _funding_look

    _prepare(ra_world, tmp_path)
    packet = _packet(tmp_path)
    identity = packet["lock"]["reserve"]
    before = {
        path: path.read_bytes()
        for pack in (ra_world["d256"], ra_world["d1024"])
        for path in (pack / "inputs/control.json", pack / "inputs/life-allocation.json")
    }

    def heldout(checkpoint: Path, *_a, **kw):
        dose = 256 if "d256" in str(checkpoint) else 1024
        look = _funding_look(
            dose=dose,
            checkpoint_sha256=digest(checkpoint),
            identity=identity,
            allocation=kw["allocation_path"],
            retained_rows=1 if dose == 1024 else 0,
        )
        if dose == 256:
            for row in look["metrics"]["life_retention"]:
                row.update(retained_rows=0, retention=0.0)
        return look

    monkeypatch.setattr(funding, "evaluate_heldout", heldout)
    _invoke(monkeypatch, packet)
    result = json.loads(Path(packet["funding_output"]).read_text())
    assert result["verdict"] == "PASS"
    monkeypatch.setattr(i3, "SERIES", identity["series_id"])
    monkeypatch.setattr(i3, "RESERVE", identity["reserve_allocation_sha256"])
    monkeypatch.setattr(job, "B8_SHA", B8_SHA)
    fit_hashes = {dose: packet["lock"][f"d{dose}"]["checkpoint_sha256"] for dose in (256, 1024)}
    original_stage_fit = i3.stage_fit
    monkeypatch.setattr(
        i3, "stage_fit", lambda dose: {**original_stage_fit(dose), "fit_sha256": fit_hashes[dose]}
    )
    staging = tmp_path / "i3-staging"
    control = i3.staging(staging)
    i3.replace_bound(staging, control, "D4096_FUNDING", result)
    i3.assert_dependent_seals(staging, control)
    job._validate_staging(staging, control, [4096], i3.RECIPE, set())
    assert all(path.read_bytes() == payload for path, payload in before.items())


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
    fit = ra_world["d1024"] / package.output_names("04")[0] / "exports/fit"
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


@pytest.mark.parametrize("failure", ["unavailable", "advanced", "rollback"])
def test_trusted_time_failure_never_claims_or_opens_heldout(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path, failure: str
) -> None:
    from research.bl_residual.github_authority import AuthorityUnavailable

    _prepare(ra_world, tmp_path)
    packet = _packet(tmp_path)
    monkeypatch.setattr(
        funding, "evaluate_heldout", lambda *_a, **_k: pytest.fail("heldout opened")
    )

    def refused():
        raise AuthorityUnavailable(f"synthetic {failure} clock/API observation")

    monkeypatch.setattr(ra_guard, "trusted_time", refused)
    with pytest.raises(AuthorityUnavailable):
        _invoke(monkeypatch, packet)
    assert not Path(packet["lock_path"]).exists()
    assert not Path(packet["funding_output"]).exists()


def test_wait_boundary_uses_server_epoch_and_seals_fresh_observation(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    from tests.test_bl_residual_ra_timing import _authoritative

    _authoritative(ra_guard.D4096_CONTROL.parent, monkeypatch, ra_guard.START_DEADLINE)
    _prepare(ra_world, tmp_path)
    packet = _packet(tmp_path)
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    monkeypatch.setattr(ra_guard, "trusted_time", lambda: {"epoch": ra_guard.CUTOFF - 1})
    monkeypatch.setattr(
        funding, "evaluate_heldout", lambda *_a, **_k: pytest.fail("heldout opened")
    )
    with pytest.raises(ValueError, match="waits"):
        _invoke(monkeypatch, packet)
    assert not Path(packet["lock_path"]).exists()
    server = {"epoch": ra_guard.CUTOFF, "date": "synthetic GitHub cutoff Date"}
    monkeypatch.setattr(ra_guard, "trusted_time", lambda: server)
    monkeypatch.setattr(funding, "evaluate_heldout", lambda *_a, **_k: {})
    monkeypatch.setattr(funding, "evaluate", lambda *_a, **_k: {})
    _invoke(monkeypatch, packet)
    claim = json.loads(Path(packet["lock_path"]).read_text())
    assert claim["claimed_epoch"] == ra_guard.CUTOFF
    assert claim["claim_time_observation"] == server
    assert claim["f2_timing_eligible"] is False


@pytest.mark.parametrize("failure_kind", ["raw-git", "metadata"])
def test_unavailable_local_binding_prepares_strikes_before_semantic_read(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path, failure_kind: str
) -> None:
    import subprocess

    from research.bl_residual import github_authority

    def unavailable(_receipt):
        if failure_kind == "raw-git":
            raise subprocess.CalledProcessError(128, ["git", "show", "missing-ledger"])
        raise ValueError("unverifiable publication metadata")

    monkeypatch.setattr(github_authority, "published_receipt_binding", unavailable)
    monkeypatch.setattr(
        funding.f2, "verify_published_receipt", lambda _: pytest.fail("sealed result parsed")
    )
    receipt = tmp_path / "unverifiable-receipt.json"
    export_json(receipt, {"synthetic": "opaque receipt; publication unverifiable"})
    result = ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=tmp_path / "strikes/funding.json",
        packet_out=tmp_path / "strikes/packet.json",
        pin="e" * 40,
        f2_receipt=receipt,
    )
    packet = json.loads(Path(result["packet"]).read_text())
    assert packet["f2_branch"] == "UNPUBLISHED_APPLY_STRIKES_BRANCH"
    assert packet["f2_receipt"] is None
    assert packet["timing"]["f2_publication"] is None
    assert not Path(packet["lock_path"]).exists()


def test_missing_publication_authority_takes_strikes_without_semantic_f2_read(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path
) -> None:
    _f2_packet(monkeypatch, ra_world, tmp_path)
    receipt = Path(_packet(tmp_path)["f2_receipt"]["path"])
    monkeypatch.setattr(ra_guard, "publication_evidence", lambda _: None)
    monkeypatch.setattr(
        funding.f2, "verify_published_receipt", lambda _: pytest.fail("sealed result parsed")
    )
    result = ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=tmp_path / "strikes/funding.json",
        packet_out=tmp_path / "strikes/packet.json",
        pin="e" * 40,
        f2_receipt=receipt,
    )
    packet = json.loads(Path(result["packet"]).read_text())
    assert packet["f2_branch"] == "UNPUBLISHED_APPLY_STRIKES_BRANCH"
    assert packet["timing"]["f2_publication"] is None
    seen = []
    monkeypatch.setattr(funding, "evaluate_heldout", lambda *_a, **_k: {})
    monkeypatch.setattr(funding, "evaluate", lambda _l, _r, f2, **kw: seen.append((f2, kw)) or {})
    _invoke(monkeypatch, packet)
    assert seen[0][0] == {} and seen[0][1]["f2_timing_eligible"] is False
    claim = json.loads(Path(packet["lock_path"]).read_text())
    assert claim["timing"]["f2_publication"] is None


@pytest.mark.parametrize("alter_evidence", [False, True])
def test_real_publication_receipt_is_bound_and_revalidated_in_packet_and_claim(
    monkeypatch: pytest.MonkeyPatch, ra_world: dict, tmp_path: Path, alter_evidence: bool
) -> None:
    import subprocess

    from tests.test_bl_residual_ra_timing import _published_binding

    # Fit publication intentionally stubs subprocess.check_output process-wide.
    # Keep F2's independent synthetic Git repository on the actual Git boundary.
    def real_git(root: Path, *args: str) -> bytes:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=30
        ).stdout

    monkeypatch.setattr(funding.f2, "_git", real_git)
    proof_root = tmp_path / "actual-f2"
    proof_root.mkdir()
    _, receipt, binding = _published_binding(proof_root, monkeypatch, int(ra_guard.CUTOFF - 1))
    ra.prepare(
        d256_pack=ra_world["d256"],
        d1024_pack=ra_world["d1024"],
        census_dir=ra_world["census"],
        funding_out=tmp_path / "r-a/funding-look.json",
        packet_out=tmp_path / "r-a/packet.json",
        pin="e" * 40,
        f2_receipt=receipt,
    )
    packet = _packet(tmp_path)
    publication = packet["timing"]["f2_publication"]
    evidence = Path(publication["evidence_path"])
    assert publication["receipt_sha256"] == binding["sha256"]
    assert publication["evidence_sha256"] == digest(evidence)
    assert publication["ledger_sha256"] == digest(funding.f2.ROOT / funding.f2.RESULT_LEDGER)
    if alter_evidence:
        evidence.write_bytes(evidence.read_bytes() + b" ")
        monkeypatch.setattr(
            funding.f2, "verify_published_receipt", lambda _: pytest.fail("sealed result parsed")
        )
        monkeypatch.setattr(
            funding, "evaluate_heldout", lambda *_a, **_k: pytest.fail("heldout opened")
        )
        with pytest.raises(ValueError, match="timing evidence"):
            _invoke(monkeypatch, packet)
        assert not Path(packet["lock_path"]).exists()
    else:
        monkeypatch.setattr(funding, "evaluate_heldout", lambda *_a, **_k: {})
        monkeypatch.setattr(funding, "evaluate", lambda *_a, **_k: {})
        _invoke(monkeypatch, packet)
        claim = json.loads(Path(packet["lock_path"]).read_text())
        assert claim["timing"]["f2_publication"] == publication
        assert claim["command_packet_sha256"] == digest(
            Path(packet["invocation"]["--command-packet"])
        )
