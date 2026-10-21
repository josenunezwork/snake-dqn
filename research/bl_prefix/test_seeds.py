import json
from pathlib import Path

import pytest

from research.bl_prefix import seeds


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for rel in ("docs", "research", "src", "scripts"):
        (repo / rel).mkdir(parents=True)
    (tmp_path / "snake-dqn-artifacts" / seeds.SEALED_REL).mkdir(parents=True)
    return repo


def _empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seeds, "_registered_history", lambda _repo: ({"complete": True}, set()))


def test_registered_domains_are_deterministic_disjoint_and_sized() -> None:
    allocation = seeds.frozen_allocation()
    assert {name: len(values) for name, values in allocation.items()} == seeds.FROZEN_COUNTS
    assert len(set().union(*map(set, allocation.values()))) == sum(seeds.FROZEN_COUNTS.values())
    assert seeds.DOMAIN == "bl-prefix-e6e7-development/20261021/v2"
    assert all(seeds.SEED_OFFSET <= value < 2**63 for values in allocation.values() for value in values)
    assert seeds.namespace_seeds("source", 8, round_index=1) != seeds.namespace_seeds("source", 8)


def test_real_scanner_excludes_sealed_tree_without_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fixture_repo(tmp_path)
    sealed = tmp_path / "snake-dqn-artifacts" / seeds.SEALED_REL
    (sealed / "forbidden.json").write_text('{"seeds": [1]}\n')
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == sealed:
            raise AssertionError("scanner attempted to enumerate sealed study")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    result = seeds._visible_text_scan(
        repo,
        tmp_path / "snake-dqn-artifacts",
        set(seeds.namespace_seeds("preflight", 2)),
        (sealed,),
    )
    assert result["hits"] and str(sealed) not in result["roots"]


def test_audit_rejects_visible_numeric_list_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fixture_repo(tmp_path)
    _empty_history(monkeypatch)
    collision = seeds.namespace_seeds("preflight", 1)[0]
    monkeypatch.setattr(
        seeds,
        "_visible_text_scan",
        lambda *_args: {
            "hits": {"fixture": [collision]},
            "progress": {},
            "roots": [],
            "method": "test",
        },
    )
    with pytest.raises(seeds.SeedAuditFailed, match="whole allocation rejected"):
        seeds.audit({"preflight": 1}, repo=repo)


def test_bound_bank_collects_numeric_json_lists(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    bank = repo / "research/bank.json"
    receipt = repo / "research/audit.json"
    bank.parent.mkdir(parents=True)
    bank.write_text(json.dumps({"rows": [{"world_seed": 7}], "numbers": [9, {"more": [11]}]}))
    receipt.write_text(json.dumps({"status": "CLEAR", "bank_sha256": seeds.sha256(bank)}))
    _row, values = seeds._bound_bank(repo, "research/bank.json", "research/audit.json", {"CLEAR"})
    assert {7, 9, 11}.issubset(values)


def test_supervisor_export_uses_complete_and_sha256_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fixture_repo(tmp_path)
    _empty_history(monkeypatch)
    monkeypatch.setattr(
        seeds,
        "_visible_text_scan",
        lambda *_args: {"hits": {}, "progress": {}, "roots": [], "method": "test"},
    )
    output = tmp_path / "snake-dqn-artifacts" / "bl-e6e7" / "audit.json"
    receipt = seeds.audit({"preflight": 2}, repo=repo, out=output)
    from research.bl_ppo.exports import export_json, verify_export

    export_json(output, receipt)
    assert verify_export(output) == seeds.sha256(output)
    assert Path(str(output) + ".complete").read_text() == "complete\n"


def test_verify_receipt_reconstructs_round_one_preflight_namespace() -> None:
    values = seeds.namespace_seeds("preflight", seeds.PREFLIGHT_COUNT, round_index=1)
    receipt = {
        "schema": "bl-prefix-seed-audit/v2",
        "status": "PASS",
        "domain": seeds.DOMAIN,
        "round_index": 1,
        "allocation": {
            "preflight": {
                "namespace": seeds.NAMESPACES["preflight"] + "/round1",
                "count": len(values),
                "seeds": values,
                "seed_digest": seeds.seed_digest(values),
            }
        },
        "registered_history": {
            "historical_frp_exclusions": [],
            "sealed_round0_coverage": {"covered": True},
        },
        "registered_collision_seeds": [],
    }
    seeds.verify_receipt(receipt, preflight_only=True, round_index=1)
    receipt["allocation"]["preflight"]["seeds"] = list(reversed(values))
    with pytest.raises(seeds.SeedAuditFailed, match="does not reconstruct"):
        seeds.verify_receipt(receipt, preflight_only=True, round_index=1)
