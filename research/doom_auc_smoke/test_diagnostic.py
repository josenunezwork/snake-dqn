"""Mechanical action-decision telemetry and frozen-runtime parity checks."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from research.doom_auc_smoke import run as runner
from research.doom_auc_smoke.evidence import empty_ledger
from research.doom_auc_smoke.fixtures import synthetic_dataset

BASE = "eefbb767d7ec5505aaefa88bed1eec3894f5a75f"
ROOT = Path(__file__).resolve().parents[2]


def test_legal_mask_lowest_doom_ties_and_no_mutation() -> None:
    logits = torch.tensor([[-100.0, -4.0, 3.0, -4.0, 0.0, 2.0], [0.0] * 6], requires_grad=True)
    legal = torch.tensor([[False, True, True, True, False, False], [False] * 6])
    before, rng = logits.detach().clone(), torch.get_rng_state().clone()
    result = runner.diagnostic_decision(logits, legal)
    assert result["action"] == 1 and result["doom_logit"] == -4
    assert result["legal"][result["action"]] and result["device"] == "cpu"
    assert torch.equal(logits, before) and logits.grad is None
    assert torch.equal(torch.get_rng_state(), rng)
    # The first row with no legal actions is skipped, never assigned an illegal action.
    legal[0] = False
    legal[1, 5] = True
    assert runner.diagnostic_decision(logits, legal)["action"] == 5
    with pytest.raises(ValueError, match="no legal action"):
        runner.diagnostic_decision(logits, torch.zeros_like(legal))
    with pytest.raises(ValueError, match="matching"):
        runner.diagnostic_decision(logits[:, :5], legal[:, :5])


@pytest.mark.parametrize("arm", ["spatial", "b8"])
def test_diagnostic_preserves_frozen_training_and_prediction(tmp_path: Path, arm: str) -> None:
    """Compare against the unedited pre-fix entrypoint, not a disabled new helper."""
    reference_path = tmp_path / "reference.py"
    reference_path.write_bytes(
        subprocess.check_output(["git", "show", f"{BASE}:research/doom_auc_smoke/run.py"], cwd=ROOT)
    )
    spec = importlib.util.spec_from_file_location("_doom_pre_diagnostic", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(1)
    data = synthetic_dataset(8, 8)
    for name, module in (("reference", reference), ("candidate", runner)):
        output = tmp_path / name
        output.mkdir()
        assert module.train_arm(
            arm,
            data,
            output,
            tmp_path,
            empty_ledger("id"),
            "id",
            device="cpu",
            updates=3,
            batch=2,
            deadline=time.monotonic() + 60,
            stopped=[False],
        )
    left, right = [
        torch.load(tmp_path / name / f"units/a0000-{arm}-fit-0003.pt", weights_only=True)
        for name in ("reference", "candidate")
    ]

    def exact(a: object, b: object) -> None:
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.numpy().tobytes() == b.numpy().tobytes()
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                exact(x, y)
        else:
            assert a == b

    exact(left, right)  # Includes model, optimizer, sampler, CPU/CUDA RNG and step.
    left, right = [
        torch.load(tmp_path / name / f"units/a0000-{arm}-prediction.pt", weights_only=True)
        for name in ("reference", "candidate")
    ]
    exact(left["logits"], right["logits"])
    assert right["diagnostic_decision"]["legal"][right["diagnostic_decision"]["action"]]


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA observer proof needs CUDA")
def test_real_cuda_diagnostic_qualifies_observer(tmp_path: Path) -> None:
    # Match the observer's real, nonsymlinked runtime repository layout.
    package = tmp_path / "repo/research/doom_auc_smoke"
    package.mkdir(parents=True)
    (package / "run.py").write_bytes((ROOT / "research/doom_auc_smoke/run.py").read_bytes())
    code = """
import importlib.util
import os
import torch
from research.colab_cli import canary_observer as observer
path = os.path.join(os.environ['COLAB_CLI_ROOT'], 'repo/research/doom_auc_smoke/run.py')
spec = importlib.util.spec_from_file_location('_doom_cuda_diagnostic', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
observer.install()
result = module.diagnostic_decision(torch.tensor([[0., -9., 2., 3., 4., 5.]], device='cuda'),
                                    torch.tensor([[True, False, True, True, True, True]]))
assert result['action'] == 0
rows = observer.collect(__import__('os').environ['COLAB_CLI_ROOT'], 'research/doom_auc_smoke')
assert rows[0]['operation'] == 'Tensor.argmax' and rows[0]['shape'] == [1, 6]
assert '/run.py:' in rows[0]['caller'] and rows[0]['device'].startswith('cuda')
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        env=dict(
            os.environ,
            COLAB_CLI_ROOT=str(tmp_path),
            SNAKE_COLAB_CANARY="1",
            SNAKE_COLAB_CANARY_PACKAGE="research/doom_auc_smoke",
        ),
        timeout=60,
    )
