"""Focused lossless/cache-integrity coverage for the T34 fit replay cache."""

from __future__ import annotations

import json

import numpy as np
import pytest

from research.t34_exit_b8_fit.cache import (
    CachedReplaySource,
    CacheValidationError,
    build_cache,
)
from research.t34_exit_b8_fit.data import StreamingReplayLoader
from research.t34_exit_b8_fit.interface import FakeReplaySource, ReplayWorld

IDENTITY = {"export_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}


def _flatten(source, epoch: int) -> list[dict[str, np.ndarray]]:
    return list(StreamingReplayLoader(source, batch_size=3, seed=17).epoch(epoch))


def test_cache_is_byte_lossless_and_keeps_loader_sampling_order(tmp_path) -> None:
    raw = FakeReplaySource(worlds=2, rows_per_world=6, seed=31)
    cache = build_cache(raw, tmp_path / "fit-cache", identity=IDENTITY, include_tower=True)
    for epoch in (1, 4):
        before, after = _flatten(raw, epoch), _flatten(cache, epoch)
        assert len(before) == len(after)
        for left, right in zip(before, after):
            assert left.keys() == right.keys()
            for key in left:
                assert left[key].dtype == right[key].dtype
                assert np.array_equal(left[key], right[key])
    first_row = next(cache.replay_epoch(1)).rows[0]
    assert isinstance(first_row["local"], np.memmap)
    assert first_row["local"].dtype == np.uint8
    assert first_row["q_prev"].dtype == np.float32


def test_cache_includes_heldout_once_when_source_has_it(tmp_path) -> None:
    class Source(FakeReplaySource):
        calls = 0

        def replay_heldout(self, epoch):
            self.calls += 1
            yield from super().replay_epoch(epoch)

    source = Source(worlds=1, rows_per_world=4)
    cache = build_cache(source, tmp_path / "cache", identity=IDENTITY)
    assert source.calls == 1
    assert len(list(cache.replay_heldout(1))) == 1
    assert len(list(cache.replay_heldout(99))) == 1
    stats = cache.manifest["stats"]
    assert stats["rows"] == stats["training_rows"] + stats["heldout_rows"] == 8
    assert stats["bytes_per_row"] == stats["bytes"] / stats["rows"]


def test_cache_does_not_invent_inventory_for_source_without_it(tmp_path) -> None:
    cache = build_cache(FakeReplaySource(rows_per_world=4), tmp_path / "cache", identity=IDENTITY)
    assert not callable(getattr(cache, "training_row_counts", None))

    class Counted(FakeReplaySource):
        def training_row_counts(self, epoch):
            return (2, 2)

    counted = build_cache(
        Counted(worlds=1, rows_per_world=4), tmp_path / "counted", identity=IDENTITY
    )
    assert counted.training_row_counts(100) == (2, 2)


def test_cache_refuses_incomplete_corrupt_or_stale_identity(tmp_path) -> None:
    target = tmp_path / "cache"
    build_cache(FakeReplaySource(rows_per_world=4), target, identity=IDENTITY)
    with pytest.raises(FileExistsError, match="create-only"):
        build_cache(FakeReplaySource(rows_per_world=4), target, identity=IDENTITY)
    with pytest.raises(CacheValidationError, match="identity mismatch"):
        CachedReplaySource(target, identity={"export_sha256": "c" * 64})
    npy = next((target / "training").glob("world-*/*.npy"))
    with npy.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(CacheValidationError, match="hash mismatch"):
        CachedReplaySource(target, identity=IDENTITY)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(CacheValidationError, match="incomplete"):
        CachedReplaySource(incomplete, identity=IDENTITY)
    (target / "manifest.sha256").write_text("", encoding="ascii")
    with pytest.raises(CacheValidationError, match="sha sidecar"):
        CachedReplaySource(target, identity=IDENTITY)


def test_optional_tower_is_preserved_and_reported(tmp_path) -> None:
    cache = build_cache(
        FakeReplaySource(rows_per_world=4),
        tmp_path / "tower",
        identity=IDENTITY,
        include_tower=True,
    )
    assert cache.manifest["include_tower"] is True
    assert cache.manifest["stats"]["bytes_per_row"] > 0
    assert "full_sight" in next(cache.replay_epoch(1)).rows[0]
    no_tower = build_cache(
        FakeReplaySource(rows_per_world=4), tmp_path / "plain", identity=IDENTITY
    )
    assert "full_sight" not in next(no_tower.replay_epoch(1)).rows[0]


def test_build_rejects_inventory_drift_and_chunk_drop_rate(tmp_path) -> None:
    class Drift(FakeReplaySource):
        def training_row_counts(self, epoch):
            return (999, 999)

    with pytest.raises(CacheValidationError, match="counts must describe"):
        build_cache(Drift(rows_per_world=4), tmp_path / "drift", identity=IDENTITY)

    class Bad(FakeReplaySource):
        def replay_epoch(self, epoch):
            yield ReplayWorld("bad", (), False, chunk_id="chunk")

    with pytest.raises(CacheValidationError, match="2%"):
        build_cache(Bad(rows_per_world=4), tmp_path / "bad", identity=IDENTITY)


def test_cache_accepts_v2_lad_and_shadow_metadata_with_different_scalar_keys(tmp_path) -> None:
    class V2(FakeReplaySource):
        def replay_epoch(self, epoch):
            world = next(super().replay_epoch(epoch))
            lad, shadow = world.rows[:2]
            lad.update(
                label_kind="lad",
                t_star=12,
                decision=12,
                a_star=int(lad["action_star"]),
                F=[int(lad["action_star"])],
                per_action={str(lad["action_star"]): {"survives": True, "mass": 12.5}},
            )
            lad["forbidden_mask"] = [False] * 6
            shadow.update(label_kind="shadow_confirm", shadow_digest="confirm-12")
            yield world

    cache = build_cache(V2(rows_per_world=4), tmp_path / "v2", identity=IDENTITY)
    rows = next(cache.replay_epoch(1)).rows
    assert rows[0]["label_kind"] == "lad"
    assert rows[0]["F"] == [rows[0]["action_star"]]
    assert rows[0]["per_action"][str(rows[0]["action_star"])]["mass"] == 12.5
    assert rows[1]["kl_anchor_only"] is True


def test_cache_preserves_masked_auxiliary_nan(tmp_path) -> None:
    class Aux(FakeReplaySource):
        def replay_epoch(self, epoch):
            world = next(super().replay_epoch(epoch))
            for index, row in enumerate(world.rows):
                row["hindsight_log_free_region"] = 1.0 if index == 0 else float("nan")
                row["hindsight_log_free_region_mask"] = index == 0
            yield world

    cache = build_cache(
        Aux(rows_per_world=4), tmp_path / "aux", identity=IDENTITY, include_tower=True
    )
    rows = next(cache.replay_epoch(1)).rows
    assert np.isnan(rows[1]["hindsight_log_free_region"])
    assert bool(rows[0]["hindsight_log_free_region_mask"])


def test_cache_requires_production_world_sha_proof(tmp_path) -> None:
    class Production(FakeReplaySource):
        is_fake = False

    with pytest.raises(CacheValidationError, match="sha256 proof"):
        build_cache(Production(rows_per_world=4), tmp_path / "production", identity=IDENTITY)


def test_manifest_metadata_tampering_is_refused(tmp_path) -> None:
    target = tmp_path / "cache"
    build_cache(FakeReplaySource(rows_per_world=4), target, identity=IDENTITY)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training"].reverse()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CacheValidationError, match="manifest hash mismatch"):
        CachedReplaySource(target, identity=IDENTITY)
