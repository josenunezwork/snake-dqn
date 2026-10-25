"""Post-F4 successor-gate bindings remain narrow and decision-bearing."""

import numpy as np
import pytest

from research.bl_round30.analyze import require_global_life_identity
from research.bl_round30.protocol import (
    AMENDMENT_SHA,
    JUDGMENT_SHA,
    require_successor_gate,
)


def _sha(letter: str) -> str:
    return letter * 64


def _diagnosis() -> dict:
    return {
        "ruling_sha256": JUDGMENT_SHA,
        "amendment_sha256": AMENDMENT_SHA,
        "corpus_allocation": _sha("a"),
        "development_only": True,
        "label": "OPTIMIZER_ADEQUATE",
        "diagnostics": {"certificate": {"certified": True}},
        "conversion": {
            "label": "GO",
            "control": {
                "name": "adam",
                "procedure": "adam",
                "endpoint_sha256": _sha("b"),
                "parameters_sha256": _sha("e"),
                "convergence_binding": {
                    "procedure": "adam",
                    "initializer": "zero-residual",
                    "initializer_parameters_sha256": _sha("f"),
                    "recipe_sha256": _sha("1"),
                    "batch_plan_sha256": _sha("2"),
                    "cache_sha256": _sha("3"),
                },
            },
        },
    }


def _gate() -> dict:
    return {
        "ruling_sha256": JUDGMENT_SHA,
        "amendment_sha256": AMENDMENT_SHA,
        "allocation": _sha("a"),
        "f4_terminal": True,
        "f4_r9_label": 5,
        "f4_continuation_permitted": True,
        "label5_exception": {
            "amendment_sha256": AMENDMENT_SHA,
            "f4_r9_label": 5,
            "scope": "r30-2-rescue-survivor-mass-ranking",
            "terminal_label": "STOP_FUNDING",
        },
        "optimizer_resolved": True,
        "conversion_trigger_passed": True,
        "archive_admitted": True,
        "optimizer": "adam",
        "convergence_procedure": "adam",
        "control_name": "adam",
        "control_procedure": "adam",
        "control_endpoint_sha256": _sha("b"),
        "control_parameters_sha256": _sha("e"),
        "convergence_binding": _diagnosis()["conversion"]["control"]["convergence_binding"],
        "diagnosis_sha256": _sha("c"),
        "allow_successor": True,
    }


def test_successor_gate_binds_original_and_amendment_and_resolved_adam_control() -> None:
    assert require_successor_gate(_gate(), _sha("a"), _diagnosis()) == "adam"


@pytest.mark.parametrize(
    "change, match",
    [
        ({"f4_r9_label": 4}, "only original R9 label 5"),
        ({"amendment_sha256": _sha("d")}, "foreign successor gate"),
        ({"control_name": "lbfgs"}, "conversion control"),
    ],
)
def test_successor_gate_refuses_a_broadened_label5_exception_or_control(change, match) -> None:
    gate = _gate()
    gate.update(change)
    with pytest.raises(ValueError, match=match):
        require_successor_gate(gate, _sha("a"), _diagnosis())


def test_global_life_ids_cannot_collapse_tranche_local_identifiers() -> None:
    table = {
        "q": np.zeros((3, 6)),
        "source_life_id": np.asarray(["fold1/p1/l1", "fold1/p1/l1", "fold2/p2/l1"]),
        "global_life_index": np.asarray([0, 0, 1]),
        "parent_world": np.asarray(["p1", "p1", "p2"]),
        "life_id": np.asarray(["fold1/p1/l1", "fold1/p1/l1", "fold2/p2/l1"]),
    }
    require_global_life_identity(table)
    table["global_life_index"] = np.asarray([0, 0, 0])
    with pytest.raises(ValueError, match="integer map collision"):
        require_global_life_identity(table)


def test_global_life_identity_refuses_zip_truncation_and_noninteger_map() -> None:
    table = {
        "q": np.zeros((2, 6)),
        "source_life_id": np.asarray(["f/p/1"]),
        "global_life_index": np.asarray([0, 1]),
        "parent_world": np.asarray(["p", "q"]),
        "life_id": np.asarray(["f/p/1", "f/q/1"]),
    }
    with pytest.raises(ValueError, match="unequal lengths"):
        require_global_life_identity(table)
    table["source_life_id"] = np.asarray(["f/p/1", "f/q/1"])
    table["global_life_index"] = np.asarray([0.0, 1.0])
    with pytest.raises(ValueError, match="invalid dtypes"):
        require_global_life_identity(table)
