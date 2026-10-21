"""Focused immutable-manifest checks for the BL E4/E2 handoff."""

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_ppo.manifests import (
    ManifestRefused,
    e2_transitions,
    git_head,
    make_e2_manifests,
    make_e2_templates,
    make_e4_manifest,
)

COMMIT = git_head()


def _roster(tmp_path: Path) -> tuple[Path, ...]:
    rows = []
    for ordinal in range(5):
        path = tmp_path / f"roster-{ordinal}.pth"
        path.write_bytes(bytes([ordinal]))
        rows.append(path)
    return tuple(rows)


def _complete(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row))
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
    )
    path.with_name(path.name + ".complete").write_text("complete\n")


def _admitted_e9(tmp_path: Path) -> Path:
    """Synthetic test-only OC fixture; never an operator admission artifact."""
    from research.bl_ppo.rule import RULE

    counts = {"scripted": 128, "frozen": 48, "mixed": 48}
    path = tmp_path / "test-only-e9.json"
    _complete(
        path,
        {
            "schema": "bl-ppo-e9-oc/v1",
            "rule": RULE,
            "measured_inputs": {
                "sha256": "a" * 64,
                "mixes": dict.fromkeys(
                    ("scripted", "frozen", "mixed"),
                    {
                        "paired_mass_variance": 1.0,
                        "paired_alive_variance": 0.01,
                        "paired_episode_variance": 0.01,
                    },
                ),
            },
            "S1_inputs": {
                "sha256": "b" * 64,
                "mixes": dict.fromkeys(
                    ("scripted", "frozen", "mixed"),
                    {
                        "paired_mass_variance": 1.0,
                        "paired_alive_variance": 0.01,
                        "paired_episode_variance": 0.01,
                    },
                ),
            },
            "selection": {"counts": counts, "dispatch_admitted": True},
            "summaries": [{"counts": counts, "passes_probability_bars": True}],
            "scenarios": [
                {
                    "counts": counts,
                    "rho_shared_B8": rho,
                    "true_mass_and_early_dividend": effect,
                    "true_survival_delta": delta,
                    "family_funding_probability": 0.05 if effect == 0 else 0.7,
                }
                for rho in (0.0, 0.5, 0.9)
                for effect in (0.0, 20.0, 40.0, 60.0, 80.0)
                for delta in (-0.05, 0.0, 0.03, 0.06)
            ],
        },
    )
    return path


def test_e4_has_two_process_ruler_and_whole_session_cap(tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    manifest = make_e4_manifest(
        commit=COMMIT,
        output_dir=tmp_path / "out",
        b8_checkpoint=b8,
        roster_checkpoints=_roster(tmp_path),
    )
    assert manifest["env"]["BL_PPO_PROCESS_COUNTS"] == "6,12"
    assert manifest["env"]["BL_PPO_WORLDS"] == "128"
    assert manifest["timeout_minutes"] == 30
    assert manifest["budget"]["unit_cap"] == 2.0


def test_e2_requires_complete_e4_and_uses_4m_below_kill_threshold(tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    receipt = tmp_path / "e4.json"
    row = {
        "schema": "bl-ppo-e4-ruler/v1",
        "status": "COMPLETE",
        "scientific": True,
        "worlds": 128,
        "long_bodies": True,
        "selected_processes": 6,
        "configs": [
            {"processes": 6, "allocated_hero_transitions_per_second": 999.0},
            {"processes": 12, "allocated_hero_transitions_per_second": 800.0},
        ],
    }
    _complete(receipt, row)
    assert e2_transitions(receipt) == 4_000_000
    # The amended 2M refit cannot fit the conservative envelope at 999/s.
    with pytest.raises(ManifestRefused, match="per-arm"):
        make_e2_manifests(
            commit=COMMIT,
            output_root=tmp_path / "e2",
            b8_checkpoint=b8,
            e4_receipt=receipt,
            e9_receipt=_admitted_e9(tmp_path),
            roster_checkpoints=_roster(tmp_path),
        )


def test_e2_requires_e4_and_explicit_budget_fitting_20m_switch(tmp_path: Path) -> None:
    receipt = tmp_path / "e4.json"
    receipt.write_text("{}")
    with pytest.raises(ManifestRefused):
        e2_transitions(receipt)
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    row = {
        "schema": "bl-ppo-e4-ruler/v1",
        "status": "COMPLETE",
        "scientific": True,
        "worlds": 128,
        "long_bodies": True,
        "selected_processes": 12,
        "configs": [
            {"processes": 6, "allocated_hero_transitions_per_second": 4100.0},
            {"processes": 12, "allocated_hero_transitions_per_second": 6000.0},
        ],
    }
    _complete(receipt, row)
    assert e2_transitions(receipt) == 8_000_000
    jobs = make_e2_manifests(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        e4_receipt=receipt,
        e9_receipt=_admitted_e9(tmp_path),
        roster_checkpoints=_roster(tmp_path),
        request_scale=True,
    )
    assert {job["env"]["BL_PPO_TRANSITIONS"] for job in jobs} == {"20000000"}
    assert all(2 * job["requirements"]["est_minutes"] <= job["timeout_minutes"] for job in jobs)


def test_templates_have_no_invented_e4_input_and_runtime_demands_sidecars(tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    templates = make_e2_templates(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        roster_checkpoints=_roster(tmp_path),
    )
    assert all(item["bl_ppo"]["template_only"] for item in templates)
    assert all(
        not any(row["dest"] == "e4-ruler.json" for row in item["inputs"]) for item in templates
    )
    script = (Path(__file__).with_name("colab.sh")).read_text()
    assert "${payload}.sha256" in script
    assert "${payload}.complete" in script


def test_materialized_e4_transport_includes_verifiable_sidecars(tmp_path: Path) -> None:
    import shutil

    from research.bl_ppo.exports import verify_export

    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    receipt = tmp_path / "e4.json"
    _complete(
        receipt,
        {
            "schema": "bl-ppo-e4-ruler/v1",
            "status": "COMPLETE",
            "scientific": True,
            "worlds": 128,
            "long_bodies": True,
            "selected_processes": 6,
            "configs": [
                {"processes": 6, "allocated_hero_transitions_per_second": 3000},
                {"processes": 12, "allocated_hero_transitions_per_second": 2900},
            ],
        },
    )
    job = make_e2_manifests(
        commit=COMMIT,
        output_root=tmp_path / "e2",
        b8_checkpoint=b8,
        e4_receipt=receipt,
        e9_receipt=_admitted_e9(tmp_path),
        roster_checkpoints=_roster(tmp_path),
    )[0]
    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    for row in job["inputs"]:
        target = uploaded / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["path"], target)
    verify_export(uploaded / job["env"]["BL_PPO_E4_BASENAME"])
    assert job["env"]["BL_PPO_DEVICE"] == "cuda"


def test_e2_forecast_charges_refit_and_fixed_ordinary_bank(tmp_path: Path) -> None:
    from research.bl_ppo.manifests import CRITIC_REFIT_TRANSITIONS, ORDINARY_BANK_TRANSITIONS

    assert CRITIC_REFIT_TRANSITIONS == 2_000_000
    assert ORDINARY_BANK_TRANSITIONS == 10_000
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    template = make_e2_templates(
        commit=COMMIT,
        output_root=tmp_path / "out",
        b8_checkpoint=b8,
        roster_checkpoints=_roster(tmp_path),
    )[0]
    assert template["bl_ppo"]["critic_refit_transitions_min"] == 2_000_000
    assert len(template["bl_ppo"]["e8_amendment_sha256"]) == 64


def test_e2_materialization_refuses_missing_e9_before_any_manifest(tmp_path: Path) -> None:
    from research.bl_ppo.manifests import make_e2_manifests

    receipt = tmp_path / "e4.json"
    _complete(
        receipt,
        {
            "schema": "bl-ppo-e4-ruler/v1",
            "status": "COMPLETE",
            "scientific": True,
            "worlds": 128,
            "long_bodies": True,
            "selected_processes": 6,
            "configs": [
                {"processes": 6, "allocated_hero_transitions_per_second": 3000},
                {"processes": 12, "allocated_hero_transitions_per_second": 2900},
            ],
        },
    )
    with pytest.raises(ManifestRefused, match="E9"):
        make_e2_manifests(
            commit=COMMIT,
            output_root=tmp_path / "blocked",
            b8_checkpoint=tmp_path / "absent.pth",
            e4_receipt=receipt,
        )
    assert not (tmp_path / "blocked").exists()
