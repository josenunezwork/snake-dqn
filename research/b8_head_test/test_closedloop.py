"""Action semantics, endpoint accounting and immutable resume failure checks."""

import json
from pathlib import Path

import pytest
import torch

from research.b8_head_test.closedloop import (
    ARMS,
    NAMESPACE,
    actions,
    frozen_features,
    ledger,
    publish,
    seed,
    statistics,
    validate_ledger,
    verified,
)


def test_deployment_features_match_b8_forward() -> None:
    from src.model.ego2s_network import Ego2sNet

    torch.manual_seed(9)
    b8 = Ego2sNet(local_channels=7, n_scalars=60).eval()
    local = torch.randint(256, (2, 7, 31, 31), dtype=torch.uint8)
    glob = torch.randint(256, (2, 4, 39, 39), dtype=torch.uint8)
    scalars = torch.rand(2, 60)
    with torch.inference_mode():
        phi = frozen_features(b8, local, glob, scalars)
        adv = b8.adv(phi)
        rebuilt = b8.value(phi) + adv - adv.mean(1, keepdim=True)
        torch.testing.assert_close(rebuilt, b8(local, glob, scalars), atol=1e-6, rtol=1e-6)


def test_margin_is_relative_strict_and_legal() -> None:
    values = torch.tensor(
        [[100.0, 110.0, 999.0, 0.0, 0.0, 0.0], [-20.0, -9.0, 999.0, 0.0, 0.0, 0.0]]
    )
    mask = torch.tensor([[True, True, False, False, False, False]] * 2)
    b8 = torch.tensor([0, 0])
    assert actions(values, mask, b8, torch.tensor([10.0, 10.0])).tolist() == [0, 1]
    assert actions(values, mask, b8, torch.tensor([0.0, float("inf")])).tolist() == [1, 0]
    assert actions(torch.zeros_like(values), mask, b8, torch.zeros(2)).tolist() == [0, 0]
    values[0, 1] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        actions(values, mask, b8, torch.zeros(2))


def test_seed_domains_are_disjoint() -> None:
    keys = [
        seed(mode, w, opp)
        for mode in ("main", "canary", "preflight")
        for w in range(192)
        for opp in (False, True)
    ]
    assert len(keys) == len(set(keys))
    assert all(0 <= k < 2**63 for k in keys)


def test_immutable_units_and_foreign_resume(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    row = publish(root, "batch-0000", "fit/main", [dict(world_id=0)])
    value = dict(ledger("fit/main"), units=[row])
    validate_ledger(value, "fit/main", {"batch-0000"})
    assert verified(root, row, "fit/main")["rows"] == [dict(world_id=0)]
    with pytest.raises(FileExistsError):
        publish(root, "batch-0000", "fit/main", [])
    with pytest.raises(ValueError, match="foreign"):
        validate_ledger(dict(value, namespace=NAMESPACE + "/foreign"), "fit/main", {"batch-0000"})
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(dict(value, units=[row, row]), "fit/main", {"batch-0000"})
    (root / row["path"]).write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        verified(root, row, "fit/main")


def test_bootstrap_keeps_paired_worlds_and_three_mixes() -> None:
    rows = []
    for w in range(6):
        for arm in ARMS:
            rows.append(
                dict(
                    world_id=w,
                    arm=arm,
                    mix=("scripted", "frozen", "mixed")[w % 3],
                    initial_sha256=str(w),
                    mass=w + (10 if arm != "b8" else 0),
                    deaths=0,
                    alive_fraction=1.0,
                    overrides=int(arm != "b8"),
                    decisions=2,
                )
            )
    result = statistics(rows)
    assert result["m10"]["all"]["ci95"] == [10.0, 10.0]
    assert result["m10"]["all"]["override_rate"] == 0.5
    assert result["b8"]["all"]["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="duplicate"):
        statistics(rows + [rows[0]])
    rows[1]["initial_sha256"] = "foreign"
    with pytest.raises(ValueError, match="unpaired"):
        statistics(rows)


def test_receipt_ledger_is_outcome_blind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research.b8_head_test.closedloop_readout import build_ledger
    from research.doom_auc_smoke.evidence import digest

    output = tmp_path.resolve()
    row = publish(
        output / "exports/closedloop", "batch-0000", "fit/main", [dict(secret="outcomes")]
    )
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(dict(commit="a" * 40, env=dict(B8_HEAD_MODE="main"))))
    receipt = dict(
        stop_verified=True,
        commit="a" * 40,
        manifest_sha256=digest(manifest),
        exports=[dict(path="closedloop/" + row["path"], size=row["size"], sha256=row["sha256"])],
    )
    (output / "receipt.json").write_text(json.dumps(receipt))
    original = Path.read_text

    def guarded(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "batch-0000.json":
            raise AssertionError("outcome payload decoded")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    result, _ = build_ledger([output], "fit/main")
    assert result["units"] == [row]
    receipt["stop_verified"] = False
    (output / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="stopped"):
        build_ledger([output], "fit/main")
