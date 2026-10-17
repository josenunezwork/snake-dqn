"""Portable evaluator inputs for tests that otherwise require Mac-only research artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


def evaluator_context(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Build a real GridBatchSim context backed only by temporary checkpoints.

    The production identity contexts deliberately verify historical checkpoint
    digests at absolute Mac paths.  Unit tests exercise the evaluator's public
    construction path, so use deterministic local weights and a small design
    authority instead of weakening that production provenance check.
    """
    import torch

    from research.redesign_m2_20261008 import ni_spec
    from src.core.config_loader import load_and_initialize_config
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.training.apex_policy import ApexPolicy

    load_and_initialize_config("research/apex_safety_20260926/deployment.yaml")
    torch.manual_seed(1717)
    carrier = tmp_path / "carrier.pth"
    torch.save(ApexPolicy(61, 64, 6, training=False).get_state_dict(), carrier)
    student = tmp_path / "student.pth"
    save_ego2s_checkpoint(student, Ego2sNet(local_channels=7, n_scalars=60))

    # Different stable member ids preserve roster rotation assertions while all
    # slots use lightweight scripted opponents.  The evaluator still builds its
    # normal runtime, policies, action masks, and GridBatchSim state.
    member_ids = [f"fixture-anchor-{index}" for index in range(6)]
    lookup = {member: ("scripted", "greedy_food") for member in member_ids}

    def design_rows(seeds: list[int]) -> list[dict[str, Any]]:
        rows = []
        for world_index, seed in enumerate(seeds):
            for offset, mix in enumerate(("scripted", "mixed", "frozen")):
                ids = [
                    member_ids[(world_index + offset + slot) % len(member_ids)] for slot in range(5)
                ]
                rows.append(
                    {
                        "world_seed": int(seed),
                        "mix": mix,
                        "slots": [{"member_sha256": member} for member in ids],
                    }
                )
        return rows

    return (
        {
            "profile": evaluation_profile_for_name("promotion-v2-watch-rect", 5000),
            "lookup": lookup
            | {
                "fixture-carrier": ("checkpoint", str(carrier)),
                ni_spec.FRP3_S12[1]: ("checkpoint", str(carrier)),
            },
            "ds": SimpleNamespace(_design_rows=design_rows),
        },
        student,
    )


def subprocess_evaluator_sitecustomize(tmp_path: Path) -> Path:
    """Write a startup shim for CLI smoke tests that spawn fresh interpreters.

    The smoke test's subprocess and spawned actor cannot receive pytest's
    in-process monkeypatches.  The shim substitutes only the historical
    identity context with the same lightweight scripted roster authority used
    by :func:`evaluator_context`; production code keeps its fail-closed digest
    checks whenever this test-only startup path is absent.
    """
    hooks = tmp_path / "python-startup"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text("""from types import SimpleNamespace

from research.apex_safety_20260926 import dev_screen
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from src.core.config_loader import load_and_initialize_config
from src.scripts.tournament_eval import evaluation_profile_for_name

load_and_initialize_config("research/apex_safety_20260926/deployment.yaml")
_members = [f"cli-fixture-anchor-{index}" for index in range(6)]
_lookup = {member: ("scripted", "greedy_food") for member in _members}

def _design_rows(seeds):
    rows = []
    for world_index, seed in enumerate(seeds):
        for offset, mix in enumerate(("scripted", "mixed", "frozen")):
            ids = [_members[(world_index + offset + slot) % len(_members)] for slot in range(5)]
            rows.append({"world_seed": int(seed), "mix": mix,
                         "slots": [{"member_sha256": member} for member in ids]})
    return rows

_context = {"profile": evaluation_profile_for_name("promotion-v2-watch-rect", 5000),
            "lookup": _lookup, "ds": SimpleNamespace(_design_rows=_design_rows)}
gid._context = lambda _threads: _context
dev_screen._design_rows = _design_rows
""")
    return hooks
