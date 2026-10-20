import pytest

from research.t34_pathb.timing_v2 import ProbeFault, probe


class _Session:
    def __init__(self):
        self.n = 0
        self.shadow_upkeep_ms = 1.0
        self.first_result_latency_ms = None

    def step(self):
        self.n += 1
        self.last_update_ms = 2.0
        if self.n == 3:
            self.first_result_latency_ms = 7.0


def test_probe_reports_on_tick_distributions_and_overruns():
    result = probe(_Session(), fps=24, ticks=3, warmup=0, contended=True)
    assert result["contended"] is True
    assert result["update_ms"]["p95"] == 2.0
    assert result["first_result_latency_ms"]["n"] == 1
    assert result["overruns"] == 0


@pytest.mark.parametrize("warmup", [0, 1])
def test_probe_rejects_background_nonfinite_q(warmup):
    class FaultSession(_Session):
        def controller_state(self):
            return {"unconditional_faults": [{"marker": "nonfinite_q"}]}

    with pytest.raises(ProbeFault, match="nonfinite_q"):
        probe(FaultSession(), fps=24, ticks=1, warmup=warmup)


def test_probe_worker_failure_is_incomplete_timing():
    class BrokenSession(_Session):
        def controller_state(self):
            return {"counts": {"worker_errors": 1}}

    with pytest.raises(RuntimeError, match="not verified") as error:
        probe(BrokenSession(), fps=24, ticks=1, warmup=0)
    assert not isinstance(error.value, ProbeFault)
