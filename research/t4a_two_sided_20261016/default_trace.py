"""Compare the T4a default learner trace with the e6331c9 baseline.

The harness runs only bounded synthetic CPU updates. It records hashes for the
online and target tensors, both RNG streams, and stable learner statistics. The
stored-bank modes deliberately use the bank's default sampler so they exercise
the pre-T4a inert path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _canonical(value: Any) -> Any:
    """Convert test values into JSON-compatible deterministic values."""
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _canonical(value.detach().cpu().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest_object(value: Any) -> str:
    """Hash a canonical JSON representation."""
    payload = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _digest_network(network: torch.nn.Module) -> tuple[str, dict[str, str]]:
    """Hash a network state dict and each tensor's raw bytes."""
    overall = hashlib.sha256()
    tensors: dict[str, str] = {}
    for name, value in network.state_dict().items():
        array = value.detach().cpu().contiguous()
        raw = array.numpy().tobytes()
        overall.update(name.encode())
        overall.update(str(array.dtype).encode())
        overall.update(repr(tuple(array.shape)).encode())
        overall.update(raw)
        tensors[name] = hashlib.sha256(raw).hexdigest()
    return overall.hexdigest(), tensors


def _stable_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Remove timing fields while retaining deterministic learner stats."""
    timing_tokens = (
        "time",
        "clock",
        "elapsed",
        "duration",
        "throughput",
        "latency",
        "wall",
        "learn_s",
    )
    return {
        key: _canonical(value)
        for key, value in stats.items()
        if not any(token in key.lower() for token in timing_tokens)
    }


def _snapshot(
    label: str,
    update: int,
    online: torch.nn.Module,
    target: torch.nn.Module,
    stats: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Capture one hash-bound learner snapshot."""
    online_hash, online_tensors = _digest_network(online)
    target_hash, target_tensors = _digest_network(target)
    stable = _stable_stats(stats)
    return {
        "label": label,
        "update": update,
        "online_sha256": online_hash,
        "target_sha256": target_hash,
        "online_tensor_sha256": online_tensors,
        "target_tensor_sha256": target_tensors,
        "rng_sha256": _digest_object(rng.bit_generator.state),
        "torch_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "stats_sha256": _digest_object(stable),
        "stable_stats": stable,
    }


def _run_bank_off(
    learner_module: Any,
    ego2s_net: Any,
    world: Any,
    ingest: Any,
    stats_factory: Any,
    args_factory: Any,
    fast: bool,
) -> dict[str, Any]:
    """Run the bank-off reference or fast learner path."""
    torch.manual_seed(3)
    nscal, multi, replay, labels, demos = world("bv8")
    online, target = ego2s_net(7, nscal).train(), ego2s_net(7, nscal).eval()
    target.load_state_dict(online.state_dict())
    learner_options = learner_module.LearnerOpts(fast=fast, fused_adam=False)
    stats, rng = stats_factory(), np.random.default_rng([7, 17])
    learner = learner_module.Learner(
        torch,
        online=online,
        target=target,
        opt=learner_module.make_optimizer(torch, online, learner_options, "cpu"),
        replay=replay,
        multi=multi,
        demos=demos,
        labels=labels,
        rng=rng,
        st=stats,
        args=args_factory("bv8"),
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=learner_options,
        agent_rows=24,
        anchor_rows=8,
        min_rows=4,
    )
    learner.target_every = 3
    feed = np.random.default_rng(99)
    trace = []
    for update in range(1, 9):
        learner.update(0.5)
        learner.flush()
        trace.append(_snapshot("bank_off", update, online, target, stats, rng))
        ingest(replay, feed, 40)
    learner.flush()
    return {"trace": trace, "final": _snapshot("bank_off_final", 8, online, target, stats, rng)}


def _run_bank_inert(
    learner_module: Any,
    ego2s_net: Any,
    label_bank: Any,
    write_bank: Any,
    world: Any,
    ingest: Any,
    stats_factory: Any,
    args_factory: Any,
    fast: bool,
    fraction: float,
) -> dict[str, Any]:
    """Run the stored-bank path with T4a's default inert mode."""
    with tempfile.TemporaryDirectory(prefix="t4a-bank-") as directory:
        bank = label_bank(write_bank(Path(directory)), 60)
        torch.manual_seed(3)
        nscal, multi, replay, labels, demos = world("bv8")
        if fraction == 1.0:
            labels = None
        online, target = ego2s_net(7, nscal), ego2s_net(7, nscal)
        target.load_state_dict(online.state_dict())
        learner_options = learner_module.LearnerOpts(fast=fast, fused_adam=False)
        stats, rng = stats_factory(), np.random.default_rng([7, 17])
        args = args_factory(
            "bv8",
            label_bank_frac=fraction,
            label_frac=1.0 - fraction,
            label_bank_margin_scale=1.7,
            onpolicy_labels=fraction != 1.0,
        )
        args.label_bank_counter_mode = "inert"
        args.label_bank_counter_ratio = None
        learner = learner_module.Learner(
            torch,
            online=online,
            target=target,
            opt=learner_module.make_optimizer(torch, online, learner_options, "cpu"),
            replay=replay,
            multi=multi,
            demos=demos,
            labels=labels,
            rng=rng,
            st=stats,
            args=args,
            nscal=nscal,
            margin=0.3,
            device="cpu",
            opts=learner_options,
            agent_rows=24,
            anchor_rows=8,
            min_rows=4,
            bank=bank,
        )
        learner.target_every = 3
        feed = np.random.default_rng(99)
        trace = []
        for update in range(1, 9):
            learner.update(0.5)
            learner.flush()
            trace.append(_snapshot(f"bank_inert_{fraction}", update, online, target, stats, rng))
            ingest(replay, feed, 40)
        learner.flush()
        return {
            "trace": trace,
            "final": _snapshot(f"bank_inert_{fraction}_final", 8, online, target, stats, rng),
        }


def main() -> None:
    """Run the trace harness for a selected source root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="source tree to import and execute")
    parser.add_argument("output", type=Path, help="JSON output path")
    options = parser.parse_args()

    root = options.root.resolve()
    output = options.output.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    # Import helpers only after selecting the source tree.
    from research.redesign_m3c_20261008 import (  # noqa: E402
        learner_m3c as learner_module,
    )
    from research.t3_rollout_labels_20261015.bank import LabelBank  # noqa: E402
    from src.model.ego2s_network import Ego2sNet  # noqa: E402
    from tests.test_perf_learner import _args, _ingest, _st, _world  # noqa: E402
    from tests.test_t3_label_bank_trainer import _write_bank  # noqa: E402

    torch.set_num_threads(1)
    modes = {
        "bank_off_ref": _run_bank_off(
            learner_module, Ego2sNet, _world, _ingest, _st, _args, fast=False
        ),
        "bank_off_fast": _run_bank_off(
            learner_module, Ego2sNet, _world, _ingest, _st, _args, fast=True
        ),
    }
    for fraction in (0.5, 1.0):
        for fast in (False, True):
            name = f"bank_inert_{fraction}_{'fast' if fast else 'ref'}"
            modes[name] = _run_bank_inert(
                learner_module,
                Ego2sNet,
                LabelBank,
                _write_bank,
                _world,
                _ingest,
                _st,
                _args,
                fast,
                fraction,
            )

    result = {
        "root": str(root),
        "commit": os.environ.get("TRACE_COMMIT", "unknown"),
        "cpu_only": True,
        "torch_num_threads": torch.get_num_threads(),
        "modes": modes,
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"root": str(root), "out": str(output), "modes": list(modes)}, sort_keys=True))


if __name__ == "__main__":
    main()
