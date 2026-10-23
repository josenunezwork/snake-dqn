import json
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_json, export_torch
from research.bl_ppo.p1_source_repair import build_origin_import, diff_scope


def _origin(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "origin"
    root.mkdir()
    export_torch(root / "p1-y3-origin-pause-snapshot-00000.pt", {"state": "pause"})
    export_json(
        root / "training-status.json",
        {
            "arm": "F",
            "counters": {
                "critic_refit_transitions": 2523136,
                "critic_refit_fit_transitions": 2010624,
                "critic_fixed_holdout_transitions": 60000,
            },
            "critic_refit": {"status": "PASS"},
        },
    )
    failed = tmp_path / "failed-receipt.json"
    export_json(failed, {"status": "INCOMPLETE"})
    offline = tmp_path / "offline-audit-claude.json"
    export_json(offline, {"status": "HOLD_MISSING_OPERATIONAL_PROOF"})
    control = tmp_path / "control.json"
    export_json(
        control,
        {
            "registration_sha256": "a" * 64,
            "evaluation_bank_sha256": "b" * 64,
            "e8_critic_convergence_amendment_04_sha256": "c" * 64,
        },
    )
    return root, failed, control, offline


def test_build_origin_import_binds_full_catalog_and_refit(tmp_path: Path) -> None:
    root, failed, control, offline = _origin(tmp_path)
    value = build_origin_import(
        origin_root=root, failed_receipt=failed, offline_audit=offline, source_control=control
    )
    assert value["origin_snapshot"]["path"] == "p1-y3-origin-pause-snapshot-00000.pt"
    assert value["expected_counters"]["critic_refit_transitions"] == 2523136
    assert value["offline_audit"]["path"] == "offline-audit-claude.json"
    assert value["expected_identity"]["natural_collection_world_id_offset"] == 3011000


def test_build_origin_import_refuses_missing_counter(tmp_path: Path) -> None:
    root, failed, control, offline = _origin(tmp_path)
    status = json.loads((root / "training-status.json").read_text())
    status["counters"].pop("critic_refit_transitions")
    (root / "training-status.json").unlink()
    for suffix in (".sha256", ".complete"):
        Path(str(root / "training-status.json") + suffix).unlink()
    export_json(root / "training-status.json", status)
    with pytest.raises(ValueError, match="exact refit counters"):
        build_origin_import(
            origin_root=root, failed_receipt=failed, offline_audit=offline, source_control=control
        )


@pytest.mark.parametrize("bad", [False, True])
def test_diff_scope_records_pinned_tools_and_refuses_science(tmp_path, monkeypatch, bad):
    import subprocess

    from research.bl_ppo import p1_source_repair

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "base").write_text("base")
    git("add", "base")
    git("commit", "-qm", "base")
    monkeypatch.setattr(p1_source_repair, "ORIGIN_COMMIT", git("rev-parse", "HEAD"))
    name = "src/science.py" if bad else "research/bl_ppo/p1_source_repair.py"
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    target.write_text("fixture = True\n")
    git("add", name)
    git("commit", "-qm", "change")
    commit = git("rev-parse", "HEAD")
    if bad:
        with pytest.raises(ValueError, match="out-of-scope"):
            diff_scope(tmp_path, commit)
    else:
        result = diff_scope(tmp_path, commit)
        assert result["repair_commit"] == commit
        assert result["diff_stat"]
        assert result["paths"][0]["path"] == name
