"""R6 E-scaling: ladder contract, pre-registered reducer branches, microbatch pinning."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from research.gpu_sim_spike import r6_analyze, r6_units
from research.gpu_sim_spike.r6_bench import Microbatched, pin_microbatches

COMMIT = "c" * 40


def _content(unit, rate, *, exact=True, headroom=0.5, status="MEASURED"):
    if status == "OOM":
        return {"status": "OOM", "memory": {"headroom_fraction": None}}
    return {
        "status": "MEASURED",
        "median_complete_world_steps_per_second": rate,
        "blocks": [{"complete_world_steps_per_second": rate}] * 3,
        "mapped_proof": {"exact": exact, "mismatched_world_ticks": 0 if exact else 7},
        "memory": {"headroom_fraction": headroom, "peak_reserved_bytes": 1},
    }


def _tree(tmp_path, *, gain=0.4, exact=True, headroom=0.5, oom=None, drop=None, native="PASS"):
    root = tmp_path / "exports"
    cells, ns = r6_units.allocation(), r6_units.namespace(COMMIT)
    for unit in cells:
        if unit == drop:
            continue
        _mix, worlds = r6_units.parse(unit)
        rate = 6000.0 * (1 + gain * (worlds > 1024))
        content = _content(
            unit,
            rate,
            exact=exact or worlds == 1024,
            headroom=headroom if worlds == 4096 else 0.8,
            status="OOM" if unit == oom else "MEASURED",
        )
        r6_units.publish_unit(root, ns, cells, unit, content)
    r6_units.build_ledger(root, root / "verified-ledger.json", commit=COMMIT)
    r6_units.publish_json(
        root / "native-parity.json",
        {"status": native, "native_cuda_executed": native == "PASS", "commit": COMMIT},
    )
    return root


def test_ladders_and_allocation_order():
    assert r6_units.ladder() == (1024, 2048, 4096)
    assert r6_units.allocation()[:3] == (
        "cell-frozen-e1024",
        "cell-frozen-e2048",
        "cell-frozen-e4096",
    )
    assert r6_units.allocation(canary=True)[-1] == "cell-scripted-e256"
    assert r6_units.parse("cell-mixed-e2") == ("mixed", 2)
    with pytest.raises(ValueError):
        r6_units.parse("cell-mixed-e8192")


def test_keep_requires_all_three_bars_on_every_roster(tmp_path):
    assert r6_analyze.aggregate(_tree(tmp_path / "a"), COMMIT)["verdict"] == "KEEP_E4096"
    assert r6_analyze.aggregate(_tree(tmp_path / "b", gain=0.24), COMMIT)["verdict"] == "DROP_E4096"
    assert (
        r6_analyze.aggregate(_tree(tmp_path / "c", exact=False), COMMIT)["verdict"] == "DROP_E4096"
    )
    value = r6_analyze.aggregate(_tree(tmp_path / "d", headroom=0.19), COMMIT)
    assert value["verdict"] == "DROP_E4096" and value["descriptive_e2048"]


def test_oom_or_missing_is_incomplete_and_native_failure_invalid(tmp_path):
    oom = _tree(tmp_path / "o", oom="cell-mixed-e4096")
    assert r6_analyze.aggregate(oom, COMMIT)["verdict"] == "INCOMPLETE"
    missing = _tree(tmp_path / "m", drop="cell-frozen-e2048")
    assert r6_analyze.aggregate(missing, COMMIT)["verdict"] == "INCOMPLETE"
    bad = _tree(tmp_path / "n", native="INVALID")
    assert r6_analyze.aggregate(bad, COMMIT)["verdict"] == "INVALID"
    failed = _tree(tmp_path / "f")
    r6_units.publish_json(failed / "r6-failure.json", {"status": "INVALID"})
    assert r6_analyze.aggregate(failed, COMMIT)["verdict"] == "INVALID"


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 6)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + y.sum(1, keepdim=True)


def test_microbatched_forward_equals_per_chunk_forward():
    torch.manual_seed(0)
    net = _Net()
    x, y = torch.randn(10, 5), torch.randn(10, 3)
    pinned = Microbatched(net, 4)(x, y)
    expected = torch.cat([net(x[i : i + 4], y[i : i + 4]) for i in range(0, 10, 4)])
    assert torch.equal(pinned, expected)
    assert torch.equal(Microbatched(net, 16)(x, y), net(x, y))


def test_pin_microbatches_preserves_shared_network_identity():
    class Policy:
        pass

    net, other = _Net(), _Net()
    policy = Policy()
    policy.b8_network, policy.opponent_networks = net, (other, None, other)
    pin_microbatches(policy, 8)
    assert policy.opponent_networks[0] is policy.opponent_networks[2]
    assert policy.opponent_networks[1] is None and policy.b8_network.size == 8
