"""Tests for the shared eval CLI helpers and the promotion-copy safety guard."""

import argparse
import json
import math

import pytest

from src.scripts.eval_cli import parse_seed_list


class TestParseSeedList:
    """The --seeds type must accept the comma form AND the range form."""

    def test_comma_form(self):
        assert parse_seed_list("0,1,2") == [0, 1, 2]

    def test_range_form_is_inclusive(self):
        assert parse_seed_list("0-15") == list(range(16))

    def test_single_seed(self):
        assert parse_seed_list("7") == [7]

    def test_degenerate_range_is_one_seed(self):
        assert parse_seed_list("4-4") == [4]

    def test_mixed_forms_preserve_order(self):
        assert parse_seed_list("0-3,7,10-11") == [0, 1, 2, 3, 7, 10, 11]

    def test_whitespace_and_empty_tokens_tolerated(self):
        # The legacy evaluate_checkpoints contract.
        assert parse_seed_list("0, 2,5") == [0, 2, 5]

    @pytest.mark.parametrize("value", ["", " , ", ",,"])
    def test_no_seeds_rejected(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="at least one seed"):
            parse_seed_list(value)

    @pytest.mark.parametrize("value", ["abc", "1.5", "0-", "-5", "1-2-3", "0..3", "0:3"])
    def test_malformed_rejected_with_clear_error(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="invalid seed"):
            parse_seed_list(value)

    def test_backwards_range_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="before start"):
            parse_seed_list("9-2")


class TestSharedAcrossScripts:
    """All three eval scripts must resolve to the one implementation."""

    def test_tournament_eval_uses_shared_parser(self):
        from src.scripts.tournament_eval import parse_seed_list as tournament_parser

        assert tournament_parser is parse_seed_list

    def test_evaluate_checkpoints_uses_shared_parser(self):
        from src.scripts.evaluate_checkpoints import (
            parse_seed_list as checkpoints_parser,
        )

        assert checkpoints_parser is parse_seed_list

    def test_ensemble_eval_legacy_alias_uses_shared_parser(self):
        from src.scripts.ensemble_eval import parse_seeds

        assert parse_seeds is parse_seed_list

    def test_set_seed_is_shared(self):
        from src.scripts.eval_cli import set_seed
        from src.scripts.tournament_eval import set_seed as tournament_set_seed

        assert tournament_set_seed is set_seed


class TestEnsembleEvalDefaults:
    """FIX A: the default config must actually exist on disk."""

    def test_default_config_exists(self):
        from pathlib import Path

        from src.scripts.ensemble_eval import DEFAULT_CONFIG

        assert Path(DEFAULT_CONFIG).exists(), f"{DEFAULT_CONFIG} is missing"


def _strict_loads(text: str):
    """json.loads that refuses NaN/Infinity/-Infinity, the way jq does."""

    def _reject(constant: str):
        raise ValueError(f"non-finite JSON constant: {constant}")

    return json.loads(text, parse_constant=_reject)


class TestJsonSafe:
    """FIX B: the -inf failure sentinel must not leak into the JSON dump."""

    def test_failed_summary_dump_parses_strictly(self):
        from src.scripts.evaluate_checkpoints import _failed_summary, json_safe

        summary = _failed_summary("broken.pth", RuntimeError("shape mismatch"))
        text = json.dumps(json_safe([summary]), indent=2)

        parsed = _strict_loads(text)
        assert parsed[0]["avg_reward"] is None
        assert parsed[0]["avg_deaths"] is None
        assert parsed[0]["error"] == "shape mismatch"
        assert parsed[0]["checkpoint"] == "broken.pth"

    def test_raw_dump_is_not_strict_json(self):
        # Guards the regression: without json_safe the dump emits -Infinity.
        from src.scripts.evaluate_checkpoints import _failed_summary

        text = json.dumps([_failed_summary("broken.pth", RuntimeError("x"))])
        assert "-Infinity" in text
        with pytest.raises(ValueError):
            _strict_loads(text)

    def test_finite_values_and_types_preserved(self):
        from src.scripts.evaluate_checkpoints import json_safe

        payload = {"a": 1.5, "b": [0.0, -2.25], "c": "text", "d": 3, "e": True, "f": None}
        assert json_safe(payload) == payload

    def test_non_finite_nested_values_nulled(self):
        from src.scripts.evaluate_checkpoints import json_safe

        assert json_safe({"x": [math.inf, -math.inf, math.nan, 1.0]}) == {
            "x": [None, None, None, 1.0]
        }


class TestCopyBestGuard:
    """FIX B: a sweep in which every candidate failed must not promote anything."""

    def _run_main(self, monkeypatch, tmp_path, summaries, argv_extra):
        from src.scripts import evaluate_checkpoints as mod

        monkeypatch.setattr(mod, "configure_project", lambda *a, **k: None)
        monkeypatch.setattr(mod, "evaluate_checkpoints", lambda *a, **k: summaries)
        return mod.main(["a.pth", "b.pth", *argv_extra])

    def test_all_failed_skips_copy_and_returns_nonzero(self, monkeypatch, tmp_path, capsys):
        from src.scripts.evaluate_checkpoints import _failed_summary

        target = tmp_path / "promoted" / "best.pth"
        summaries = [
            _failed_summary("a.pth", RuntimeError("bad a")),
            _failed_summary("b.pth", RuntimeError("bad b")),
        ]

        code = self._run_main(monkeypatch, tmp_path, summaries, ["--copy-best-to", str(target)])

        assert code == 1
        assert not target.exists()
        captured = capsys.readouterr()
        assert "no checkpoint evaluated successfully" in captured.err
        assert "Best checkpoint" not in captured.out
        assert "Copied best checkpoint" not in captured.out

    def test_all_failed_still_writes_strict_json(self, monkeypatch, tmp_path):
        from src.scripts.evaluate_checkpoints import _failed_summary

        out = tmp_path / "summary.json"
        summaries = [_failed_summary("a.pth", RuntimeError("bad a"))]

        code = self._run_main(monkeypatch, tmp_path, summaries, ["--json-output", str(out)])

        assert code == 1
        parsed = _strict_loads(out.read_text(encoding="utf-8"))
        assert parsed[0]["error"] == "bad a"

    def test_surviving_checkpoint_is_still_copied(self, monkeypatch, tmp_path, capsys):
        from src.scripts.evaluate_checkpoints import _failed_summary, summarize_rollouts

        source = tmp_path / "a.pth"
        source.write_bytes(b"weights")
        target = tmp_path / "promoted" / "best.pth"
        good = summarize_rollouts(
            str(source),
            [{"episode": {"reward": 5.0, "food_eaten": 2, "deaths": 0, "length": 30, "kills": 1}}],
        )
        summaries = [good, _failed_summary("b.pth", RuntimeError("bad b"))]

        code = self._run_main(monkeypatch, tmp_path, summaries, ["--copy-best-to", str(target)])

        assert code == 0
        assert target.read_bytes() == b"weights"
        assert "Best checkpoint" in capsys.readouterr().out
