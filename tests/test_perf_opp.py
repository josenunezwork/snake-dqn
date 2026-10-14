"""perf-opp (2026-10-13): ``--opp-forward merged-cpu`` is the per-group opponent math.

* :class:`StackedApexForward` (one ``baddbmm`` per layer for G stacked ApexNetworks) gives Q
  bitwise equal to each network's own forward on its rows, for uneven / single-row / empty
  groups padded into one block. Bitwise is asserted on Apple Accelerate (darwin), where it
  was verified; other BLAS builds only get a tolerance check here (run
  ``research/perf_opp_20261013/bench_opp.py --parity`` there before relying on it).
* The comparison catches a one-ulp weight change (the parity check can fail).
* In a real training segment (pinned checkpoints), :class:`MergedVector61Opponents` picks
  the same action and bitwise-equal Q as the per-group ``Vector61SimdPolicy.actions`` on
  every row of every frame, also on random row subsets (uneven and single-row groups).
* End to end: ``actor_main`` with ``--opp-forward merged-cpu`` emits the same frame messages
  and ring rows as the default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008 import train_m3c as T
from src.model.apex_network import ApexNetwork
from src.simd_env.vector61_merged import (
    EXACT_MIN_ROWS,
    MergedVector61Opponents,
    StackedApexForward,
    apex_signature,
    mergeable,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT  # noqa: E402

DARWIN = sys.platform == "darwin"


def _nets(G=4, D=61, H=512, seed=0):
    torch.manual_seed(seed)
    nets = [ApexNetwork(D, H, 6).eval() for _ in range(G)]
    with torch.no_grad():  # trained-looking scales, distinct per net
        for g, n in enumerate(nets):
            for p in n.parameters():
                p.mul_(1.0 + 0.1 * g)
    return nets


def _bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32).view(np.uint32)


def _stacked_q(stack, xs):
    """Pad per-group inputs into one block (as MergedVector61Opponents does)."""
    G, D = len(xs), xs[0].shape[1]
    nmax = max(max(len(x) for x in xs), 1)
    block = torch.zeros(G, nmax, D)
    for g, x in enumerate(xs):
        block[g, : len(x)] = x
    out = stack(block)
    return [out[g, : len(x)] for g, x in enumerate(xs)]


def _assert_same(a, b, tag):
    assert a.shape == b.shape, tag
    if DARWIN:
        assert np.array_equal(_bits(a), _bits(b)), tag
    else:
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "counts", [(64, 64, 64, 64), (16, 64, 17, 33), (16, 16, 16, 16), (0, 20, 128, 16)]
)
def test_stacked_forward_matches_each_network(counts):
    """The padded block (>= EXACT_MIN_ROWS rows per group) reproduces each network."""
    nets = _nets()
    stack = StackedApexForward(nets)
    gen = torch.Generator().manual_seed(sum(counts))
    xs = [torch.rand(n, 61, generator=gen) * 2.0 - 0.5 for n in counts]
    with torch.no_grad():
        ref = [net(x) if len(x) else torch.zeros(0, 6) for net, x in zip(nets, xs)]
    got = _stacked_q(stack, xs)
    for g, (a, b) in enumerate(zip(ref, got)):
        _assert_same(a, b, (counts, g))
    if not DARWIN:
        pytest.skip("bitwise parity is asserted on Apple Accelerate only (checked: close)")


class _FakeRuntime:
    """``Vector61Runtime.selection`` over fixed arrays (read-only, as the real one)."""

    def __init__(self, state, mask):
        self.sel_state, self.sel_mask = state, mask

    def selection(self, slots):
        slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        return self.sel_state[slots[:, 0], slots[:, 1]], self.sel_mask[slots[:, 0], slots[:, 1]]


def _fake_policy(net, runtime):
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    pol = Vector61SimdPolicy.__new__(Vector61SimdPolicy)
    pol._configure_veto((), "v2")
    pol._torch, pol.network, pol.device = torch, net, torch.device("cpu")
    pol.input_size, pol.runtime, pol.forward = 61, runtime, "batched"
    pol.checkpoint_path = "fake"
    return pol


@pytest.mark.parametrize(
    "counts",
    [
        (64, 64, 64, 64),
        (1, 64, 7, 33),
        (1, 1, 1, 1),
        (0, 5, 128, 2),
        (200, 3, 0, 1),
        (15, 16, 17, 0),
    ],
)
def test_merged_opponents_match_policies_actions(counts):
    """Merged == per-group ``Vector61SimdPolicy.actions`` (Q and actions), small groups too."""
    rng = np.random.default_rng(sum(counts))
    E, S = 64, 6
    state = (rng.random((E, S, 61)) * 2.0 - 0.5).astype(np.float32)
    mask = rng.random((E, S, 6)) < 0.6
    mask[rng.random((E, S)) < 0.1] = False  # exercises the normal-only fallback
    runtime = _FakeRuntime(state, mask)
    nets = _nets()
    cells = rng.permutation(np.argwhere(np.ones((E, S), dtype=bool)))
    groups, policies, start = {}, {}, 0
    for g, n in enumerate(counts):
        spec = ("checkpoint", f"net{g}")
        groups[spec] = cells[start : start + max(n, 1)].astype(np.int64)
        start += max(n, 1)
        policies[spec] = _fake_policy(nets[g], runtime)
    alive = np.zeros((E, S), dtype=bool)
    for spec, n in zip(groups, counts):
        if n:
            alive[groups[spec][:, 0], groups[spec][:, 1]] = True
    merged = MergedVector61Opponents(policies, groups, runtime, "cpu")
    assert merged.specs == frozenset(groups) and len(merged.parts) == 1
    ref = np.full((E, S), -1, dtype=np.int64)
    ref_q = []
    for spec, cg in groups.items():
        live = cg[alive[cg[:, 0], cg[:, 1]]]
        if len(live):
            ref[live[:, 0], live[:, 1]] = policies[spec].actions(None, None, live)
            ref_q.append(policies[spec].q_values(runtime.selection(live)[0]))
    out = np.full((E, S), -1, dtype=np.int64)
    pend = merged.launch(alive, with_q=True)
    merged.finish(pend, out)
    _assert_same(torch.cat([p["q"] for p in pend]), torch.cat(ref_q), counts)
    assert np.array_equal(out, ref), counts
    if not DARWIN:
        pytest.skip("bitwise parity is asserted on Apple Accelerate only (checked: close)")


def test_parity_check_detects_one_ulp_weight_change():
    """A deliberate break (net 2's first-layer weights moved by one ulp) must fail it."""
    nets = _nets()
    xs = [torch.rand(16, 61, generator=torch.Generator().manual_seed(g)) for g in range(4)]
    stack = StackedApexForward(nets)
    w = stack.w[0]
    w[2] = torch.from_numpy(np.nextafter(w[2].numpy(), np.float32(np.inf)))
    with torch.no_grad():
        ref = nets[2](xs[2])
    got = _stacked_q(stack, xs)[2]
    assert not np.array_equal(_bits(ref), _bits(got))


def test_stack_rejects_mixed_architectures_and_non_apex():
    with pytest.raises(ValueError):
        StackedApexForward([ApexNetwork(61, 512, 6), ApexNetwork(61, 256, 6)])
    with pytest.raises(TypeError):
        StackedApexForward([torch.nn.Linear(61, 6)])
    assert apex_signature(ApexNetwork(61, 512, 6)) == apex_signature(ApexNetwork(61, 512, 6))


def test_only_plain_batched_cpu_policies_are_merged():
    """Rowwise (batch-1 floats), vetoed and (in exact mode) off-CPU policies keep their call."""
    E, S = 8, 6
    runtime = _FakeRuntime(np.zeros((E, S, 61), np.float32), np.ones((E, S, 6), bool))
    nets = _nets()
    pols = [_fake_policy(n, runtime) for n in nets]
    pols[1].forward = "rowwise"
    pols[2]._configure_veto((0,), "v2")
    meta = ApexNetwork(61, 512, 6).to("meta")  # a network that does not live on the CPU
    pols[3].network = meta
    assert [mergeable(p) for p in pols] == [True, False, False, True]
    cells = np.argwhere(np.ones((E, S), dtype=bool))
    groups = {("c", g): cells[g * 6 : g * 6 + 6] for g in range(4)}
    policies = {("c", g): p for g, p in enumerate(pols)}
    merged = MergedVector61Opponents(policies, groups, runtime, "cpu")
    assert merged.specs == frozenset({("c", 0)})


def test_opp_forward_env_is_recorded_for_multi_actor_runs(monkeypatch):
    base = ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x"]
    parse = T.build_parser().parse_args
    monkeypatch.setenv(T.OPP_FORWARD_ENV, "merged-device")
    args = parse(base + ["--actors", "1"])
    T.resolve_opp_forward_env(args)
    assert T.m3c_flags(args) == {"actors": 1, "opp_forward": "merged-device"}
    args = parse(base + ["--actors", "1", "--opp-forward", "merged-cpu"])
    T.resolve_opp_forward_env(args)  # the flag wins
    assert args.opp_forward == "merged-cpu"
    args = parse(base)  # single-process trainer: the env is not read
    T.resolve_opp_forward_env(args)
    assert args.opp_forward is None and T.m3c_flags(args) == {}
    monkeypatch.setenv(T.OPP_FORWARD_ENV, "per-group")
    args = parse(base + ["--actors", "1"])
    T.resolve_opp_forward_env(args)
    assert args.opp_forward is None  # the default: nothing recorded
    monkeypatch.setenv(T.OPP_FORWARD_ENV, "bogus")
    with pytest.raises(SystemExit, match="M3C_OPP_FORWARD"):
        T.resolve_opp_forward_env(parse(base + ["--actors", "1"]))


def test_parser_flags_and_sync_refusal(tmp_path):
    d = T.default_args()
    assert d.opp_forward is None and T.m3c_flags(d) == {}
    args = T.build_parser().parse_args(
        ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x"]
        + ["--actors", "1", "--opp-forward", "merged-cpu"]
    )
    assert T.m3c_flags(args) == {"actors": 1, "opp_forward": "merged-cpu"}
    with pytest.raises(SystemExit, match="opp-forward needs the multi-actor"):
        T.main(
            ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", str(tmp_path / "o")]
            + ["--opp-forward", "merged-cpu"]
        )


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned pool checkpoints")
def test_merged_opponents_equal_per_group_in_a_segment():
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    args = T.build_parser().parse_args(
        ["--seed", "3", "--init", "x", "--demo-data", ".", "--out", "x", "--actors", "1"]
        + ["--worlds", "40", "--frozen-frac", "0.34", "--mixed-frac", "0.33"]
        + ["--v61-backend", "numba"]
    )
    ctx = gid._context(1)
    sim, runtime, policies, groups = T.build_segment(args, ctx, 0, 0.99, hero_v61=True)
    merged = MergedVector61Opponents(policies, groups, runtime, "cpu")
    v61 = [s for p in merged.parts for s in p.specs]
    assert len(merged.parts) == 1 and len(v61) == 4  # the strict pool: 4 checkpoints, 1 arch
    assert all(s[0] == "scripted" for s in groups if s not in merged.specs)
    assert all(mergeable(policies[s]) for s in v61)
    rng = np.random.default_rng(0)
    stats = {"rows": 0, "subset_rows": 0, "single": 0, "small": 0, "block": 0}

    def compare(prepared, alive, ref_actions, tag):
        out = np.full_like(ref_actions, -1)
        pend = merged.launch(alive, with_q=True)
        merged.finish(pend, out)
        q = torch.cat([p["q"] for p in pend])
        ref_q = []
        for s in v61:
            c = groups[s]
            live = c[alive[c[:, 0], c[:, 1]]]
            if len(live):
                ref_q.append(policies[s].q_values(runtime.selection(live)[0]))
        ref_q = torch.cat(ref_q)
        assert np.array_equal(_bits(q), _bits(ref_q)), tag
        rows = np.concatenate([p["rows"] for p in pend])
        assert np.array_equal(out[rows[:, 0], rows[:, 1]], ref_actions[rows[:, 0], rows[:, 1]])
        untouched = np.ones(out.shape, dtype=bool)
        untouched[rows[:, 0], rows[:, 1]] = False
        assert (out[untouched] == -1).all(), tag  # only the merged rows are written
        return len(rows)

    def choose(prepared):
        masks = prepared.get_resolved_action_mask()
        alive = prepared.get_alive()
        actions = np.ones((sim.E, prepared.S), dtype=np.int64)
        runtime.prepare(prepared)
        hero = np.argwhere(alive[:, :1])
        m = masks[hero[:, 0], 0].copy()
        m[~m.any(axis=1), :3] = True
        actions[hero[:, 0], 0] = T.sample_legal(rng, m)
        for spec, cells in groups.items():
            live = cells[alive[cells[:, 0], cells[:, 1]]]
            if len(live):
                mk = masks[live[:, 0], live[:, 1]]
                actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
        stats["rows"] += compare(prepared, alive, actions, "all")
        # random row subsets: uneven groups (block and own-call), single rows, empty groups
        for keep in (0.8, 0.5, 0.1):
            sub = alive & (rng.random(alive.shape) < keep)
            per_group = []
            for s in v61:
                c = groups[s]
                live = c[sub[c[:, 0], c[:, 1]]]
                per_group.append(len(live))
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[s].actions(mk, prepared, live)
            stats["single"] += sum(n == 1 for n in per_group)
            stats["small"] += sum(0 < n < EXACT_MIN_ROWS for n in per_group)
            stats["block"] += sum(n >= EXACT_MIN_ROWS for n in per_group)
            if sum(per_group):
                stats["subset_rows"] += compare(prepared, sub, actions, f"subset {keep}")
        return actions

    for _ in range(60):
        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    assert stats["rows"] > 5000 and stats["subset_rows"] > 5000, stats
    assert stats["single"] > 5 and stats["small"] > 100 and stats["block"] > 100, stats


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_actor_messages_identical_with_merged_cpu(tmp_path, monkeypatch):
    from test_perf_actor import _run_actor

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    monkeypatch.setattr(T.Schedules, "eps", lambda self, t: 0.3)
    torch.manual_seed(0)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    frames = 100  # crosses two segment ends (45 frames each)
    blocks = []  # rows per stacked block: proves the merged path drove the run
    launch = MergedVector61Opponents.launch

    def counting_launch(self, alive, with_q=False):
        pend = launch(self, alive, with_q)
        blocks.append(sum(len(p["rows"]) for p in pend))
        return pend

    monkeypatch.setattr(MergedVector61Opponents, "launch", counting_launch)
    runs, n_launch = {}, {}
    for mode in ("per-group", "merged-cpu"):
        d = tmp_path / mode
        d.mkdir()
        # 40 worlds: ~27 rows per pool checkpoint, so the stacked block is exercised
        extra = ("--opp-forward", mode, "--worlds", "40")
        runs[mode] = _run_actor(d, init, "vectorized", "numba", frames, extra)
        n_launch[mode] = len(blocks)
    assert n_launch["per-group"] == 0 and n_launch["merged-cpu"] == frames
    assert min(blocks) >= 4 * EXACT_MIN_ROWS
    ref_msgs, ref_ring = runs["per-group"]
    msgs, ring = runs["merged-cpu"]
    assert len(ref_msgs) == len(msgs) == frames
    skip = {"env_s", "timing", "j"}
    for f, (a, b) in enumerate(zip(ref_msgs, msgs)):
        assert a.keys() == b.keys(), f
        for k in a:
            if k in skip:
                continue
            x, y = a[k], b[k]
            if isinstance(x, np.ndarray):
                assert x.dtype == y.dtype and np.array_equal(x, y), (f, k)
            else:
                assert x == y, (f, k)
    for k in ref_ring:
        assert np.array_equal(ref_ring[k], ring[k]), k
