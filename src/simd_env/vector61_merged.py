"""Opt-in merged forwards for a training actor's frozen / mixed vector61 opponents (perf-opp).

A training segment (``research/redesign_m3c_20261008/train_m3c.build_segment``) already keys
its opponent groups by checkpoint: every row of one checkpoint is ONE
:class:`~src.simd_env.vector61_policy.Vector61SimdPolicy` call per frame (``forward="batched"``).
With the strict pool that is four calls per frame (four distinct ApexNetwork checkpoints of
identical architecture, different weights), each with its own selection, mask, forward,
masked argmax and host copy.

:class:`MergedVector61Opponents` replaces those calls with one per *architecture*: the rows
of every same-architecture checkpoint are gathered with one ``Vector61Runtime.selection``,
padded into a ``(G, N_max, D)`` block and run through :class:`StackedApexForward` (one
``baddbmm`` per layer, the G weight sets stacked), followed by one masked argmax.

* ``device="cpu"`` (``--opp-forward merged-cpu``): Q is bitwise equal to the per-group
  forwards on Apple Accelerate (``tests/test_perf_opp.py``: every row, every frame, uneven,
  single-row and empty groups). ``baddbmm`` runs one GEMM per batch entry with the bias
  pre-filled, as ``addmm`` does. Accelerate's GEMM is row-count invariant only from 16 rows
  up (:data:`EXACT_MIN_ROWS`), so smaller groups keep their own network call. Other BLAS
  builds (MKL / oneDNN on x86) may differ: run
  ``research/perf_opp_20261013/bench_opp.py --parity`` there before calling it same-math.
* ``device=<actor device>`` (``merged-device``): the same forward on MPS / CUDA, every group
  stacked. Float results differ from the CPU forwards (different kernels); the bench reports
  the max |dQ| and the selected-action flip rate. Not same-math.

Only plain greedy ``forward="batched"`` groups are merged (no veto slots, exact
``ApexNetwork`` modules; in exact CPU mode also only networks that live on the CPU, which
``train_m3c``'s default ``SNAKE_DQN_DEVICE=cpu`` gives); anything else stays on its own
policy. Nothing imports this unless the caller opts in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.simd_env.vector61_policy import NUM_ACTIONS, NUM_DIRECTIONS, Vector61SimdPolicy

__all__ = [
    "EXACT_MIN_ROWS",
    "OPP_FORWARD_MODES",
    "MergedVector61Opponents",
    "StackedApexForward",
    "apex_signature",
    "mergeable",
]

#: ``per-group`` = one ``Vector61SimdPolicy.actions`` call per checkpoint (the default).
OPP_FORWARD_MODES = ("per-group", "merged-cpu", "merged-device")
#: Apple Accelerate (torch 2.9.1, M-series, one thread) uses a different GEMM kernel below 16
#: rows: an ApexNetwork's Q for a row then depends on how many rows share the call (measured:
#: M = 1..15 differ in the last bits from M >= 16, and M >= 16 are row-count invariant). Such
#: a group cannot be reproduced inside a padded block, so in exact (CPU) mode it keeps its own
#: network call.
EXACT_MIN_ROWS = 16


def _apex_class():
    from src.model.apex_network import ApexNetwork

    return ApexNetwork


def _layers(net) -> List[Any]:
    f, v, a = net.feature_layer, net.value_stream, net.advantage_stream
    return [f[0], f[2], v[0], v[2], a[0], a[2]]


def apex_signature(net) -> Tuple[Tuple[int, ...], ...]:
    """Weight / bias shapes of an ``ApexNetwork`` (equal signatures can be stacked)."""
    return tuple(tuple(p.shape) for layer in _layers(net) for p in (layer.weight, layer.bias))


def mergeable(policy: Any) -> bool:
    """A plain greedy vector61 opponent: ``forward="batched"``, no veto, an exact
    ``ApexNetwork`` (a ``"rowwise"`` policy runs batch-1 forwards: other floats)."""
    return (
        isinstance(policy, Vector61SimdPolicy)
        and policy.forward == "batched"
        and not policy.veto_slots
        and type(policy.network) is _apex_class()
    )


class StackedApexForward:
    """``ApexNetwork.forward`` for G same-architecture networks in one batched call.

    ``forward(x)`` maps ``(G, N, D)`` to ``(G, N, A)`` with ``out[g] == networks[g](x[g])``:
    ``Linear`` -> ``baddbmm(bias, x, W^T)`` (the transposed weight view, as ``F.linear``
    passes it to ``addmm``), the same ReLUs, and the dueling combination of
    :func:`src.model.base_network.dueling_q`. The weights are copied (detached) at
    construction; the source networks are not referenced afterwards.
    """

    def __init__(self, networks: Sequence[Any], device: Any = "cpu") -> None:
        import torch

        if not networks:
            raise ValueError("StackedApexForward needs at least one network")
        apex = _apex_class()
        sigs = set()
        for net in networks:
            if type(net) is not apex:
                raise TypeError(f"only exact ApexNetwork modules stack, got {type(net).__name__}")
            sigs.add(apex_signature(net))
        if len(sigs) != 1:
            raise ValueError("networks differ in architecture; stack each signature separately")
        self._torch = torch
        self.device = torch.device(device)
        self.G = len(networks)
        self.input_size = int(networks[0].feature_layer[0].in_features)
        self.output_size = int(networks[0].advantage_stream[2].out_features)
        with torch.no_grad():
            per_layer = list(zip(*[_layers(n) for n in networks]))
            # weight (G, out, in) stored contiguous; the forward uses its (G, in, out) view
            self.w = [
                torch.stack([lay.weight.detach().float() for lay in ls]).to(self.device)
                for ls in per_layer
            ]
            self.b = [
                torch.stack([lay.bias.detach().float() for lay in ls])[:, None, :].to(self.device)
                for ls in per_layer
            ]

    def _linear(self, i: int, x):
        return self._torch.baddbmm(self.b[i], x, self.w[i].transpose(1, 2))

    def forward(self, x):
        torch = self._torch
        with torch.no_grad():
            h = torch.relu(self._linear(0, x))
            h = torch.relu(self._linear(1, h))
            value = self._linear(3, torch.relu(self._linear(2, h)))
            adv = self._linear(5, torch.relu(self._linear(4, h)))
            return value + (adv - adv.mean(dim=-1, keepdim=True))

    __call__ = forward


class _Part:
    """The groups of one architecture signature."""

    def __init__(
        self,
        specs: List[Any],
        cells: List[np.ndarray],
        stack: StackedApexForward,
        policies: List[Any],
    ) -> None:
        self.specs = specs
        self.cells = cells
        self.stack = stack
        self.policies = policies


class MergedVector61Opponents:
    """Act for every mergeable vector61 opponent group of one segment at once.

    Args:
        policies: ``{spec: policy}`` of the segment (as ``build_segment`` returns them).
        groups: ``{spec: (n, 2) int64 cells}`` of the segment.
        runtime: The segment's shared :class:`Vector61Runtime` (read-only here).
        device: Where the stacked forward runs (``"cpu"`` or the actor's MPS / CUDA device).
        exact_min_rows: Groups with fewer live rows run their own network call instead of
            the padded block. Default :data:`EXACT_MIN_ROWS` on CPU (the per-group floats),
            ``0`` on a device (everything stacked; not same-math anyway).

    ``specs`` is the set of specs handled here; the caller keeps driving the others (scripted
    anchors, vetoed policies) through their own ``actions``. :meth:`launch` builds the inputs
    and enqueues the forward and masked argmax (asynchronous on MPS / CUDA), :meth:`finish`
    copies the actions back and writes them, so host work can run in between.
    """

    def __init__(
        self,
        policies: Mapping[Any, Any],
        groups: Mapping[Any, np.ndarray],
        runtime: Any,
        device: Any = "cpu",
        exact_min_rows: Optional[int] = None,
    ) -> None:
        import torch

        self._torch = torch
        self.runtime = runtime
        self.device = torch.device(device)
        if exact_min_rows is None:
            exact_min_rows = EXACT_MIN_ROWS if self.device.type == "cpu" else 0
        self.exact_min_rows = int(exact_min_rows)
        exact = self.device.type == "cpu"
        by_sig: Dict[Any, List[Any]] = {}
        for spec, pol in policies.items():
            if spec not in groups or not mergeable(pol):
                continue
            if exact and next(pol.network.parameters()).device.type != "cpu":
                continue  # its per-group forward runs elsewhere: a CPU stack is other floats
            by_sig.setdefault(apex_signature(pol.network), []).append(spec)
        self.parts: List[_Part] = []
        for specs in by_sig.values():
            pols = [policies[s] for s in specs]
            stack = StackedApexForward([p.network for p in pols], self.device)
            cells = [np.asarray(groups[s], dtype=np.int64).reshape(-1, 2) for s in specs]
            self.parts.append(_Part(specs, cells, stack, pols))
        self.specs = frozenset(s for p in self.parts for s in p.specs)

    def launch(self, alive: np.ndarray, with_q: bool = False) -> List[Dict[str, Any]]:
        """Enqueue this frame's forwards; ``alive`` is the sim's ``(E, S)`` alive grid."""
        from src.training.action_mask import INVALID_Q_VALUE

        torch = self._torch
        dev = self.device
        pending: List[Dict[str, Any]] = []
        for part in self.parts:
            lives = [c[alive[c[:, 0], c[:, 1]]] for c in part.cells]
            counts = np.array([len(x) for x in lives], dtype=np.int64)
            if not counts.any():
                continue
            rows = np.concatenate(lives)
            states, exact = self.runtime.selection(rows)
            D = part.stack.input_size
            if states.shape[1] != D:
                raise ValueError(
                    f"checkpoint input_size={D} does not match the active featurizer width "
                    f"{states.shape[1]}"
                )
            mask = exact.copy()
            mask[~mask.any(axis=1), :NUM_DIRECTIONS] = True  # live allow_fallback
            G = len(lives)
            offs = np.concatenate([[0], np.cumsum(counts)])
            small = (counts > 0) & (counts < self.exact_min_rows)
            big = (counts > 0) & ~small
            with torch.no_grad():
                block = None
                if big.any():
                    nmax = int(counts[big].max())
                    x = np.zeros((G, nmax, D), dtype=np.float32)
                    for g in np.flatnonzero(big):
                        x[g, : counts[g]] = states[offs[g] : offs[g + 1]]
                    block = part.stack(torch.from_numpy(x).to(dev))
                qs = []
                for g in range(G):
                    if small[g]:  # the per-group call itself
                        q_own = part.policies[g].q_values(states[offs[g] : offs[g + 1]])
                        qs.append(q_own.to(dev))
                    elif big[g]:
                        qs.append(block[g, : counts[g]])
                qn = qs[0] if len(qs) == 1 else torch.cat(qs)
                mask_t = torch.from_numpy(mask).to(dev)
                masked = torch.where(mask_t, qn, torch.full_like(qn, INVALID_Q_VALUE))
                act = masked.argmax(dim=1)
            item = {"rows": rows, "act": act}
            if with_q:
                item["q"] = qn
                item["mask"] = mask
            pending.append(item)
        return pending

    @staticmethod
    def finish(pending: List[Dict[str, Any]], actions: np.ndarray) -> None:
        """Copy the actions back (syncs a device forward) into the ``(E, S)`` grid."""
        for item in pending:
            a = item["act"].cpu().numpy().astype(np.int64)
            rows = item["rows"]
            actions[rows[:, 0], rows[:, 1]] = np.clip(a, 0, NUM_ACTIONS - 1)

    def act(self, alive: np.ndarray, actions: np.ndarray) -> None:
        """:meth:`launch` then :meth:`finish`."""
        self.finish(self.launch(alive), actions)
