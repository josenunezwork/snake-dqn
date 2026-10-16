"""Focused contract tests for the binding T4f early-warning evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.t4f_return_20261016.early_warning import (
    EarlyWarningConfig,
    OutsideObservations,
    evaluate,
    main,
)


class _Bank:
    def __init__(self, *, family: int, died: list[bool], actions: list[int], b8: int = 0) -> None:
        n = len(actions)
        self.state_packed = np.zeros((n, 2), np.uint8)
        self.state_scalars = np.zeros((n, 60), np.float32)
        self.local = np.zeros((n, 7, 31, 31), np.uint8)
        self.global_ = np.zeros((n, 4, 37, 37), np.uint8)
        self.family = np.full(n, family, np.uint8)
        self.died = np.asarray(died, dtype=bool)
        self.action = np.asarray(actions, dtype=np.int64)
        self.b8_action = np.full(n, b8, np.int64)


class _Net(torch.nn.Module):
    def __init__(self, q: list[float]) -> None:
        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float32)

    def forward(self, local, glob, scalars):
        return self.q.expand(len(local), -1)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    current, initial, telemetry = (
        tmp_path / "current.pth",
        tmp_path / "initial.pth",
        tmp_path / "log.jsonl",
    )
    current.write_bytes(b"current")
    initial.write_bytes(b"initial")
    telemetry.write_text("")
    return current, initial, telemetry


def _config(tmp_path: Path, *, transitions: int = 1_000_000) -> EarlyWarningConfig:
    current, initial, telemetry = _paths(tmp_path)
    return EarlyWarningConfig(
        current, initial, Path("decisive"), Path("outside"), telemetry, transitions
    )


def _evaluate(
    tmp_path: Path,
    *,
    q: list[float],
    initial: list[float],
    error: float = 1.0,
    baseline: float = 2.0,
    boost: float = 0.0,
    transitions: int = 1_000_000,
):
    config = _config(tmp_path, transitions=transitions)
    config.telemetry_jsonl.write_text(
        json.dumps(
            {
                "transitions": transitions,
                "return_mean_abs_contrast_error": error,
                "return_initial_mean_abs_contrast_error": baseline,
                "greedy_boost_rate_lt_700": boost,
            }
        )
        + "\n"
    )
    decisive = _Bank(family=0, died=[True, False], actions=[0, 1])
    outside = OutsideObservations(
        local=np.zeros((1, 7, 31, 31), np.uint8),
        global_=np.zeros((1, 4, 37, 37), np.uint8),
        scalars=np.zeros((1, 60), np.float32),
        b8_action=np.array([0], np.int64),
        legal_mask=np.array([[True, True, False, False, False, False]]),
    )
    banks = {"decisive": decisive}
    nets = {"current.pth": _Net(q), "initial.pth": _Net(initial)}
    return evaluate(
        config,
        bank_loader=lambda path, _: banks[path.name],
        outside_loader=lambda _: outside,
        network_loader=lambda path: nets[path.name],
        torch_module=torch,
    )


def test_one_million_kills_nonmoving_decisive_gap_when_train_error_fell(tmp_path: Path) -> None:
    got = _evaluate(tmp_path, q=[4, 2, 0, 0, 0, 0], initial=[4, 2, 0, 0, 0, 0])
    assert got["status"] == "KILL"
    assert got["decisive"]["median_gap"] == 2
    assert got["train"]["fallen"] is True
    assert got["reasons"] == ["heldout_decisive_gap_nonnegative_while_train_error_fell"]


def test_one_million_kills_outside_spillover_and_low_length_boost(tmp_path: Path) -> None:
    got = _evaluate(
        tmp_path, q=[0, 2, 0, 0, 0, 0], initial=[0, 0, 0, 0, 0, 0], error=3, baseline=2, boost=0.011
    )
    assert got["status"] == "KILL"
    assert set(got["reasons"]) == {
        "heldout_outside_gap_delta_exceeds_0.5",
        "lt_700_greedy_boost_rate_exceeds_1pct",
    }


def test_half_million_records_metrics_without_early_kill(tmp_path: Path) -> None:
    got = _evaluate(
        tmp_path,
        q=[3, 0, 0, 0, 0, 0],
        initial=[0, 0, 0, 0, 0, 0],
        error=1,
        baseline=2,
        boost=0.2,
        transitions=500_000,
    )
    assert got["status"] == "PASS"
    assert got["reasons"] == []


def test_reports_macro_changed_nondecisive_heldout_states(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.telemetry_jsonl.write_text(
        json.dumps(
            {
                "transitions": 1_000_000,
                "return_mean_abs_contrast_error": 1.0,
                "return_initial_mean_abs_contrast_error": 2.0,
                "greedy_boost_rate_lt_700": 0.0,
            }
        )
        + "\n"
    )
    # B8 action 0 survives in the first state; it must be excluded rather than
    # silently redefining fatal as action 1.  The second state stays decisive.
    decisive = _Bank(family=0, died=[False, True, True, False], actions=[0, 1, 0, 1])
    decisive.state_packed[2:] = 1
    decisive.state_scalars[2:] = 1
    outside = OutsideObservations(
        np.zeros((1, 7, 31, 31), np.uint8),
        np.zeros((1, 4, 37, 37), np.uint8),
        np.zeros((1, 60), np.float32),
        np.array([0]),
        np.array([[True, True, False, False, False, False]]),
    )
    nets = {"current.pth": _Net([0, 2, 0, 0, 0, 0]), "initial.pth": _Net([0] * 6)}
    got = evaluate(
        config,
        bank_loader=lambda *_: decisive,
        outside_loader=lambda _: outside,
        network_loader=lambda path: nets[path.name],
        torch_module=torch,
    )
    assert got["decisive"]["n_states"] == 1
    assert got["decisive"]["excluded_nondecisive"] == 1


def test_cli_fails_closed_and_writes_not_evaluated_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "receipt.json"
    code = main(
        [
            "--checkpoint",
            str(tmp_path / "missing.pth"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pth"),
            "--holdout-bank",
            "missing",
            "--outside-bank",
            "missing",
            "--telemetry-jsonl",
            "missing",
            "--transitions",
            "500000",
            "--out",
            str(out),
        ]
    )
    assert code == 2
    assert json.loads(out.read_text())["status"] == "NOT_EVALUATED"
    assert "NOT_EVALUATED" in capsys.readouterr().out


def test_requires_exact_look_telemetry_row(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.telemetry_jsonl.write_text(json.dumps({"transitions": 500_000}) + "\n")
    with pytest.raises(ValueError, match="100k transition window"):
        evaluate(
            config,
            bank_loader=lambda *_: _Bank(family=0, died=[True, False], actions=[0, 1]),
            outside_loader=lambda _: OutsideObservations(
                np.zeros((1, 7, 31, 31), np.uint8),
                np.zeros((1, 4, 37, 37), np.uint8),
                np.zeros((1, 60), np.float32),
                np.array([0]),
                np.array([[True, True, False, False, False, False]]),
            ),
            network_loader=lambda _: _Net([0] * 6),
            torch_module=torch,
        )


def test_uses_first_telemetry_row_after_packet_crossing(tmp_path: Path) -> None:
    config = _config(tmp_path, transitions=500_000)
    config.telemetry_jsonl.write_text(
        "\n".join(
            json.dumps(
                {
                    "transitions": transitions,
                    "return_mean_abs_contrast_error": 1.0,
                    "return_initial_mean_abs_contrast_error": 2.0,
                    "greedy_boost_rate_lt_700": 0.0,
                }
            )
            for transitions in (400_000, 500_128, 600_000)
        )
        + "\n"
    )
    decisive = _Bank(family=0, died=[True, False], actions=[0, 1])
    outside = OutsideObservations(
        np.zeros((1, 7, 31, 31), np.uint8),
        np.zeros((1, 4, 37, 37), np.uint8),
        np.zeros((1, 60), np.float32),
        np.array([0]),
        np.array([[True, True, False, False, False, False]]),
    )
    got = evaluate(
        config,
        bank_loader=lambda *_: decisive,
        outside_loader=lambda _: outside,
        network_loader=lambda _: _Net([0] * 6),
        torch_module=torch,
    )
    assert got["boost"]["source_transition"] == 500_128


def test_rejects_telemetry_only_after_the_scheduled_100k_window(tmp_path: Path) -> None:
    config = _config(tmp_path, transitions=500_000)
    config.telemetry_jsonl.write_text(json.dumps({"transitions": 600_000}) + "\n")
    with pytest.raises(ValueError, match="100k transition window"):
        from research.t4f_return_20261016.early_warning import _read_telemetry

        _read_telemetry(config.telemetry_jsonl, config.transitions)
