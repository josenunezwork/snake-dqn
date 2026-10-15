"""T4 learner-cadence contract: default parity and explicit cadence scaling."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008 import learner_m3c as lm  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as m3c  # noqa: E402
from src.model.ego2s_network import Ego2sNet  # noqa: E402
from tests.test_perf_learner import _args, _st, _world  # noqa: E402


def _learner(*, batch_scale: int, update_every: int, fast: bool = False):
    nscal, multi, replay, labels, demos = _world("bv8")
    online = Ego2sNet(7, nscal).train()
    target = Ego2sNet(7, nscal).eval()
    target.load_state_dict(online.state_dict())
    args = _args("bv8")
    opts = lm.LearnerOpts(
        batch_scale=batch_scale, update_every=update_every, fast=fast, fused_adam=False
    )
    return lm.Learner(
        torch,
        online=online,
        target=target,
        opt=lm.make_optimizer(torch, online, opts, "cpu"),
        replay=replay,
        multi=multi,
        demos=demos,
        labels=labels,
        rng=np.random.default_rng(0),
        st=_st(),
        args=args,
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=opts,
        agent_rows=24,
        anchor_rows=8,
        min_rows=4,
    )


def test_update_every_default_preserves_existing_intervals():
    learner = _learner(batch_scale=1, update_every=1)
    assert (learner.update_every, learner.target_every, learner.publish_every) == (128, 1000, 25)


@pytest.mark.parametrize("batch_scale,update_every", [(1, 2), (2, 2), (4, 3)])
def test_update_every_scales_ingest_cadence_only(batch_scale, update_every):
    learner = _learner(batch_scale=batch_scale, update_every=update_every)
    baseline = _learner(batch_scale=batch_scale, update_every=1)
    assert learner.update_every == 128 * batch_scale * update_every
    assert (learner.a_rows, learner.d_rows, learner.min_rows) == (
        baseline.a_rows,
        baseline.d_rows,
        baseline.min_rows,
    )
    assert learner.target_every == max(1, round(lm.TARGET_EVERY / batch_scale))
    assert learner.publish_every == max(1, round(25 / batch_scale))


def test_update_every_is_validated_and_recorded_as_a_nondefault_option():
    with pytest.raises(ValueError, match="--update-every"):
        lm.LearnerOpts(update_every=0)
    with pytest.raises(ValueError, match="--update-every"):
        lm.LearnerOpts(update_every=-1)
    args = m3c.build_parser().parse_args(
        [
            "--seed",
            "0",
            "--init",
            "x",
            "--demo-data",
            ".",
            "--out",
            "x",
            "--actors",
            "2",
            "--update-every",
            "2",
        ]
    )
    assert lm.LearnerOpts.from_args(args).update_every == 2
    assert m3c.m3c_flags(args)["update_every"] == 2
    with pytest.raises(SystemExit, match="multi-actor"):
        m3c.main(
            [
                "--seed",
                "0",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "x",
                "--update-every",
                "2",
            ]
        )
    with pytest.raises(SystemExit, match="--update-every"):
        m3c.main(
            [
                "--seed",
                "0",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "x",
                "--actors",
                "2",
                "--update-every",
                "-1",
            ]
        )


@pytest.mark.parametrize("fast", [False, True])
def test_ingest_credit_calls_reference_and_fast_learners_at_selected_cadence(fast):
    def credit_updates(update_every):
        learner = _learner(batch_scale=1, update_every=update_every, fast=fast)
        since_update = 256
        while since_update >= learner.update_every:
            since_update -= learner.update_every
            learner.update(0.5)
        learner.flush()
        return learner.st["updates"], since_update

    assert credit_updates(1) == (2, 0)
    assert credit_updates(2) == (1, 0)
