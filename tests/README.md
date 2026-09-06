# Test Suite

Tests for the Snake DQN project. Every command below is run from the **repo root**.

`pytest.ini` sets `testpaths = tests`, so a bare `pytest` already collects this
directory — you rarely need to pass a path.

## Running Tests

Use the project venv interpreter so you get the pinned toolchain:

```bash
./venv/bin/python -m pytest -m "not slow" -q
```

That is the gate. It is exactly what CI runs (`.github/workflows/ci.yml`), and it
must stay green.

Narrower selections while iterating:

```bash
# One file
./venv/bin/python -m pytest tests/test_apex_policy.py

# One test
./venv/bin/python -m pytest tests/test_apex_policy.py::test_local_policy_uses_small_replay_warmup

# Everything matching a keyword
./venv/bin/python -m pytest -k parity
```

`pytest.ini` already sets `-v`, `-ra`, and `--showlocals` via `addopts`, so there is
no need to add them yourself.

## Markers

`pytest.ini` sets `strict_markers = true`: an unregistered marker is a collection
**error**, not a warning. To use a new marker, register it in `pytest.ini` first.
This has to be the ini key — `--strict-markers` in `addopts` alone is silently
inert on pytest 9, and warns instead of failing.

Registered markers: `slow`, `integration`, `unit`, `requires_gpu`. Only `slow`
currently has tests attached to it — the other three are reserved, and selecting on
them today deselects everything.

`slow` is the one that matters, because CI and the documented gate both filter on it:

```bash
# The gate: skip slow tests
./venv/bin/python -m pytest -m "not slow" -q

# Only the slow tests
./venv/bin/python -m pytest -m "slow" -q
```

Mark a test `slow` when it trains, or otherwise costs seconds rather than
milliseconds, so the default gate stays fast.

## Coverage

Coverage is **not** part of the gate and `pytest-cov` is not installed in `./venv`,
so `--cov` flags fail with `unrecognized arguments` as-is. Install it into the venv
first if you want a report; `[tool.coverage.*]` in `pyproject.toml` configures the
output. The same applies to `pytest-xdist` (`-n auto`) — listed in
`requirements-dev.txt` but absent from the venv.

## Test Groups

The suite is large and grows steadily, so this is a structural map rather than a
file list — run `ls tests/` for the current inventory.

- **Game & mechanics** — the core simulation: snake movement, collisions, food,
  kill attribution, speed boost, relative actions, circular arena, curriculum, and
  the v2 mechanics/reward surface (`test_mechanics_v2.py`, `test_reward_v2.py`,
  `test_reward_events.py`, `test_behavior_probes.py`).
- **State & features** — the observation vector: the 58-D/61-D featurizer, the GPU
  featurizer, free-space features, enemy features, per-action danger, action masking.
- **Apex stack** — the distributed trainer: policy, actor, buffer, learner, the
  actor hot path, replay structures (`test_sum_tree.py`, `test_base_buffer.py`,
  `test_multistep_buffer.py`), n-step targets, and the opponent pool.
- **SIMD env & parity** — the vectorized sim and the parity checks that pin it to
  the reference implementation (`test_simd_env.py`, `test_simd_parity.py`,
  `test_obs_parity.py`). Most of these run under the default gate; the only `slow`
  tests in the suite are the 3 parametrizations of
  `test_simd_parity.py::test_parity_bit_exact_full_battery`.
- **PQN & raster** — the raster network, PQN trainer, raster inference/serving, and
  the PQN command-line path.
- **Inference, eval & promotion** — `InferenceAgent`, tournament eval (the promotion
  gate), eval stats, checkpoint evaluation, and the checkpoint contract.
- **Data & persistence** — replay DB handler, score store, replay audit/quality, and
  experience generation.
- **Web-adjacent** — human play, checkpoint handling, controls, metrics, and raster
  serving through the web backend.
- **Config & infra** — config schema parity, reconciliation, PQN config block,
  device selection, and CLI guards.

Shared fixtures and helpers live in `tests/conftest.py` (`temp_db`, `setup_config`,
`make_test_snake`).

## Adding New Tests

Add tests next to the group they belong to, or add a new `test_*.py` file.

Policies are constructed **directly** — there is no policy factory. `ApexPolicy`
reads global config, so initialize config and pin the device first, then restore
both in a `finally` so state cannot leak into later tests. `AISnake` receives a
policy via `SnakeFactory` (see `src/game/snake_factory.py`).

```python
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import (
    ApexSettings,
    AppConfig,
    NetworkSettings,
    TrainingSettings,
    initialize_config,
)
from src.training.apex_policy import ApexPolicy


def test_apex_policy_learns_from_replay():
    """ApexPolicy trains once the replay warmup is satisfied."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(batch_size=2, min_buffer_size=50, learning_rate=0.001),
        )
    )
    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        loss = None
        for i in range(8):
            loss, _ = policy.update(
                state=torch.full((4,), float(i)),
                action=i % 3,
                reward=0.1,
                next_state=torch.full((4,), float(i + 1)),
                done=False,
                snake_id=0,
            )
        assert loss is not None
        assert policy.update_counter > 0
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()
```

Tiny `input_size`/`hidden_size` values keep unit tests fast; use the real 58-D/61-D
sizes only when the test is actually about the production state layout. See
`tests/test_apex_policy.py` for more of this pattern and `tests/test_ai_snake.py`
for policy-plus-snake integration.
