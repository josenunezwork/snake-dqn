"""Risk diagnostics preserve historical identities, metrics and independent units."""

from pathlib import Path

import numpy as np
import pytest

from research.t5b_risk_and_fuse_20261016.risk import Pool, estimate, fail_probability, observations


def test_duplicate_enriches_unknown_without_inflating_n(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    pool = Pool()
    pool.add("B8_s0", "scripted", 1, 0.0, None, 12.0, source, "round52")
    pool.add("B8_s0", "scripted", 1, 0.0, 0.45, 12.0, source, "round52")
    assert len(pool.rows) == 1
    assert pool.duplicates == 1
    assert pool.rows[("B8_s0", "scripted", 1)]["alive"] == 0.45
    with pytest.raises(ValueError, match="conflicting duplicate"):
        pool.add("B8_s0", "scripted", 1, 1.0, 0.45, 12.0, source, "round52")


def test_nullable_survival_never_becomes_episode_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    pool = Pool()
    pool.add("B8_s0", "mixed", 1, 0.0, 0.8, 12.0, source, "round52")
    pool.add("incumbent", "mixed", 1, 0.0, None, 9.0, source, "round52")
    assert observations(pool, "mixed", "alive", "incumbent") == []
    assert observations(pool, "mixed", "episode", "incumbent") == [(0, 1, 0.0)]
    assert observations(pool, "mixed", "episode", "champion") == []


def test_shared_world_clustering_preserves_reference_dependence() -> None:
    # Five policies replayed on the same worlds contain the information of two worlds.
    obs = [(s, w, v) for s in range(5) for w, v in ((10, -1.0), (20, 1.0))]
    est = estimate(obs, 96)
    assert est["observed_rows"] == 10
    assert est["distinct_worlds"] == 2
    assert est["observed_cluster_se"] == pytest.approx(1.0)
    assert est["future_se"] == pytest.approx(np.sqrt(2 / 480))


def test_equal_final_means_despite_unequal_comparator_coverage() -> None:
    est = estimate([(0, 1, 0.0), (0, 2, 0.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)], 96)
    assert est["mean"] == 0.5
    assert est["future_se"] == 0
    assert fail_probability(0.5, 0, 0.5) == 0
    assert fail_probability(0.49, 0, 0.5) == 1
    assert fail_probability(0, 1, 0) == 0.5


def test_reference_binding_rejects_wrong_sha_and_veto() -> None:
    from research.t5b_risk_and_fuse_20261016.risk import REFERENCES, validate_reference

    engine = dict(veto="v8", sim_engine="grid", vector61_forward="rowwise")
    validate_reference("incumbent", dict(sha256=REFERENCES["incumbent"]), engine)
    with pytest.raises(ValueError, match="wrong reference identity"):
        validate_reference("incumbent", dict(sha256=REFERENCES["champion"]), engine)
    with pytest.raises(ValueError, match="wrong reference identity"):
        validate_reference(
            "incumbent", dict(sha256=REFERENCES["incumbent"]), dict(engine, veto=None)
        )


def test_required_bank_coverage_rejects_missing_comparator() -> None:
    from research.t5b_risk_and_fuse_20261016.risk import MIXES, validate_coverage

    pool = Pool()

    def row(policy: str, mix: str, world: int, family: str, alive: float | None) -> None:
        pool.rows[policy, mix, world] = dict(
            policy=policy, mix=mix, world=world, family=family, alive=alive, episode=1.0, mass=1.0
        )

    for s in range(5):
        for mix in MIXES:
            for w in range(s * 1000 + 1000, s * 1000 + 1032):
                row(f"B8_s{s}", mix, w, "M4", 1.0)
                row("incumbent", mix, w, "M4", 1.0)
                if s == 0:
                    row("champion", mix, w, "M4", 1.0)
            for w in range(96 if mix == "scripted" else 32):
                row(f"B8_s{s}", mix, w, "round52", 1.0)
                if w < 32:
                    row("incumbent", mix, w, "round52", 1.0 if mix == "scripted" else None)
                    if mix == "scripted":
                        row("champion", mix, w, "round52", 1.0)
    validate_coverage(pool)
    del pool.rows["incumbent", "scripted", 1000]
    with pytest.raises(ValueError, match="missing M4 served reference"):
        validate_coverage(pool)
    row("incumbent", "scripted", 1000, "M4", 1.0)
    del pool.rows["B8_s0", "scripted", 1000]
    with pytest.raises(ValueError, match="incomplete M4 candidate bank"):
        validate_coverage(pool)
