"""Stats-only tests for the rating + drift tooling (no game, no checkpoints).

Covers the pure math in ``src/scripts/rate_pool.py`` (death order -> ranks,
Plackett-Luce updates, ratings persistence) and in
``src/scripts/obs_histogram_diff.py`` (KS statistic, per-dim comparison,
self-test mode). Episode running against real checkpoints is exercised by the
scripts themselves, not here.
"""

import numpy as np
import pytest

from src.scripts.obs_histogram_diff import (
    compare_observation_sets,
    feature_label,
    histogram_summary,
    ks_statistic,
    markdown_report,
    self_test,
)
from src.scripts.rate_pool import (
    Participant,
    discover_participants,
    load_ratings,
    ranks_from_episode,
    ratings_markdown,
    save_ratings,
    update_ratings,
)

# =============================================================================
# ranks_from_episode: death order -> OpenSkill ranks
# =============================================================================


class TestRanksFromEpisode:
    def test_survivor_ranks_first(self):
        # Snake 1 survived (None); snakes 0 and 2 died.
        ranks = ranks_from_episode([100, None, 50], [10.0, 5.0, 20.0])
        assert ranks == [1, 0, 2]

    def test_later_death_ranks_better(self):
        ranks = ranks_from_episode([10, 20, 30], [5.0, 5.0, 5.0])
        assert ranks == [2, 1, 0]

    def test_same_frame_tie_broken_by_final_mass(self):
        # Both die at frame 40; the bigger snake places higher.
        ranks = ranks_from_episode([40, 40, None], [8.0, 12.0, 3.0])
        assert ranks == [2, 1, 0]

    def test_multiple_survivors_tie_broken_by_mass(self):
        # Frame-cap episode: both survive, mass decides the winner.
        ranks = ranks_from_episode([None, None, 5], [7.0, 30.0, 50.0])
        assert ranks == [1, 0, 2]

    def test_exact_tie_shares_rank(self):
        ranks = ranks_from_episode([40, 40, 10], [8.0, 8.0, 8.0])
        assert ranks == [0, 0, 2]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ranks_from_episode([1, 2], [1.0])


# =============================================================================
# update_ratings: Plackett-Luce update math
# =============================================================================


class TestUpdateRatings:
    def test_winner_gains_loser_drops(self):
        ratings = {}
        update_ratings(ratings, ["winner", "loser"], [0, 1])
        assert ratings["winner"]["mu"] > 25.0 > ratings["loser"]["mu"]
        assert ratings["winner"]["sigma"] < 25.0 / 3.0
        assert ratings["winner"]["games"] == ratings["loser"]["games"] == 1
        assert ratings["winner"]["wins"] == 1
        assert ratings["loser"]["wins"] == 0

    def test_ordinal_is_mu_minus_three_sigma(self):
        ratings = update_ratings({}, ["a", "b", "c"], [0, 1, 2])
        for entry in ratings.values():
            assert entry["ordinal"] == pytest.approx(entry["mu"] - 3.0 * entry["sigma"])

    def test_repeated_wins_separate_ordinals(self):
        ratings = {}
        for _ in range(10):
            update_ratings(ratings, ["strong", "weak"], [0, 1])
        assert ratings["strong"]["ordinal"] > ratings["weak"]["ordinal"]
        assert ratings["strong"]["games"] == 10
        assert ratings["strong"]["wins"] == 10

    def test_existing_ratings_are_continued_not_reset(self):
        ratings = {"vet": {"mu": 30.0, "sigma": 4.0, "ordinal": 18.0, "games": 7, "wins": 5}}
        update_ratings(ratings, ["vet", "rookie"], [1, 0])
        assert ratings["vet"]["games"] == 8
        assert ratings["vet"]["wins"] == 5  # lost this one
        assert ratings["vet"]["mu"] < 30.0  # upset loss pulls the veteran down
        assert ratings["rookie"]["wins"] == 1

    def test_tied_ranks_accepted(self):
        ratings = update_ratings({}, ["a", "b", "c"], [0, 0, 2])
        # Both tied winners count a win and stay symmetric.
        assert ratings["a"]["wins"] == ratings["b"]["wins"] == 1
        assert ratings["a"]["mu"] == pytest.approx(ratings["b"]["mu"])


# =============================================================================
# Ratings persistence
# =============================================================================


class TestRatingsPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "ratings.json"
        ratings = update_ratings({}, ["a", "b", "anchor:greedy_food"], [0, 1, 2])
        save_ratings(path, ratings, meta={"episodes": 1})
        loaded = load_ratings(path)
        assert loaded == ratings

    def test_missing_file_is_empty(self, tmp_path):
        assert load_ratings(tmp_path / "nope.json") == {}

    def test_nightly_continuation(self, tmp_path):
        """A second run resumes from the persisted store instead of resetting."""
        path = tmp_path / "ratings.json"
        first = update_ratings({}, ["a", "b"], [0, 1])
        save_ratings(path, first)
        second = update_ratings(load_ratings(path), ["a", "b"], [0, 1])
        assert second["a"]["games"] == 2
        assert second["a"]["mu"] > first["a"]["mu"]

    def test_markdown_sorted_by_ordinal(self):
        ratings = {}
        for _ in range(5):
            update_ratings(ratings, ["top", "mid", "low"], [0, 1, 2])
        table = ratings_markdown(ratings)
        lines = table.splitlines()
        assert lines[0].startswith("| rank |")
        body = lines[2:]
        assert "top" in body[0] and "mid" in body[1] and "low" in body[2]


# =============================================================================
# Participant discovery (filesystem only, no game)
# =============================================================================


class TestDiscoverParticipants:
    def test_checkpoints_and_anchors(self, tmp_path):
        (tmp_path / "b.pth").write_bytes(b"x")
        (tmp_path / "a.pth").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("ignored")
        pool = discover_participants(str(tmp_path), ["anchor:greedy_food"])
        assert [p.participant_id for p in pool] == ["a.pth", "b.pth", "anchor:greedy_food"]
        assert pool[0] == Participant("a.pth", "checkpoint", str(tmp_path / "a.pth"))
        assert pool[2].kind == "anchor" and pool[2].source == "greedy_food"

    def test_bad_anchor_id_raises(self, tmp_path):
        with pytest.raises(ValueError):
            discover_participants(str(tmp_path), ["greedy_food"])

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_participants(str(tmp_path / "missing"), [])


# =============================================================================
# KS statistic + comparison (obs_histogram_diff)
# =============================================================================


class TestKsStatistic:
    def test_identical_samples_zero(self):
        x = np.linspace(-1.0, 1.0, 500)
        assert ks_statistic(x, x) == 0.0

    def test_disjoint_samples_one(self):
        assert ks_statistic(np.zeros(50), np.ones(50)) == pytest.approx(1.0)

    def test_same_distribution_small(self):
        rng = np.random.default_rng(7)
        ks = ks_statistic(rng.normal(size=5000), rng.normal(size=5000))
        assert ks < 0.05

    def test_shifted_distribution_large(self):
        rng = np.random.default_rng(7)
        ks = ks_statistic(rng.normal(size=5000), rng.normal(loc=1.0, size=5000))
        # Analytic KS for a 1-sigma mean shift is 2*Phi(0.5)-1 ~ 0.383.
        assert ks == pytest.approx(0.383, abs=0.05)

    def test_empty_sample_raises(self):
        with pytest.raises(ValueError):
            ks_statistic(np.array([]), np.ones(10))


class TestCompareObservationSets:
    def test_flags_only_the_shifted_dim(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=(3000, 4))
        b = rng.normal(size=(3000, 4))
        b[:, 2] += 1.0
        result = compare_observation_sets(a, b, threshold=0.25)
        assert result["flagged"] == [2]
        assert not result["passed"]
        assert result["max_ks"] == result["dims"][2]["ks"]

    def test_passes_when_below_threshold(self):
        rng = np.random.default_rng(11)
        a = rng.normal(size=(3000, 3))
        b = rng.normal(size=(3000, 3))
        result = compare_observation_sets(a, b, threshold=0.25)
        assert result["passed"] and result["flagged"] == []

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            compare_observation_sets(np.zeros((10, 3)), np.zeros((10, 4)))

    def test_markdown_report_mentions_flagged_dim(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(2000, 2))
        b = rng.normal(size=(2000, 2))
        b[:, 1] += 2.0
        result = compare_observation_sets(a, b, threshold=0.25)
        report = markdown_report(result, "A", "B")
        assert "DRIFT DETECTED" in report
        assert "YES" in report

    def test_histogram_summary_fields(self):
        s = histogram_summary(np.arange(101, dtype=float))
        assert s["n"] == 101
        assert s["min"] == 0.0 and s["max"] == 100.0
        assert s["p50"] == pytest.approx(50.0)

    def test_feature_labels(self):
        assert feature_label(0) == "direction[0]"
        assert feature_label(4) == "length"
        assert feature_label(58) == "free_space[0]"
        assert feature_label(99) == "dim_99"


def test_self_test_passes():
    assert self_test() == 0
