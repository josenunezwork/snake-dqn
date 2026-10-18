---
name: snake-dqn-apex-training
description: |
  Use when modifying Ape-X DQN, ApexPolicy, BaseDQNPolicy, replay buffers, SumTree, multi-step returns, sequence replay, GRU/DRQN networks, distributed actors, buffer processes, learners, checkpointing, or training config in /Users/josenunez/Projects/ml/snake-dqn.
---

# Snake DQN Ape-X Training

Use this skill for training-stack changes, including single-process DQN behavior
and distributed Ape-X runs.

## Key Files

- Policy base and implementation: `src/training/base_dqn_policy.py`,
  `src/training/apex_policy.py`.
- Replay buffers: `src/training/base_buffer.py`, `src/training/replay_buffer.py`,
  `src/training/multistep_buffer.py`, `src/training/sequence_buffer.py`,
  `src/training/sum_tree.py`.
- Distributed components: `src/training/apex_actor.py`,
  `src/training/apex_buffer.py`, `src/training/apex_learner.py`.
- Networks: `src/model/apex_network.py`, `src/model/gru_network.py`.
- Coordinator: `src/scripts/apex_train.py`.

## Contracts To Preserve

- `BaseDQNPolicy` subclasses provide `select_action()`, `update()`,
  `get_state_dict()`, and `load_state_dict()`.
- `ApexPolicy` supports training mode and inference-only mode. Inference mode
  must not require an optimizer, target network, or replay buffer.
- Standard network forward pass returns Q-values shaped `(batch, 6)`.
- GRU/DRQN forward pass returns `(q_values, hidden_state)` and supports both
  `(batch, 58)` and `(batch, seq_len, 58)` inputs.
- Priorities must remain positive and numerically stable. TD-error updates,
  importance-sampling weights, beta annealing, and SumTree totals are coupled.
- Multi-step return code must flush episode-ending transitions correctly.
- Sequence replay must respect `sequence_length` and `burn_in_length` and
  reset hidden state at episode boundaries.

## Distributed Training Notes

- Actor processes generate experience on CPU and receive learner weights through
  queues.
- The learner owns GPU training and checkpoint state.
- Do not send CUDA tensors through multiprocessing queues.
- Keep queue payloads picklable and reasonably small.
- Preserve graceful shutdown paths for actors, learner cleanup, and
  `BufferProcess.shutdown()`.
- When changing config loading in `src/scripts/apex_train.py`, verify that the
  `--config` argument actually affects the values used by actors and learner.

## Verification

- Buffers and SumTree: `pytest tests/test_sum_tree.py tests/test_multistep_buffer.py tests/test_sequence_buffer.py`.
- Policy and learner: `pytest tests/test_apex_policy.py tests/test_apex_learner.py`.
- Actors/distributed edges: `pytest tests/test_apex_actor.py`.
- GRU/DRQN: `pytest tests/test_gru_network.py tests/test_drqn_integration.py`.
- Shared training changes: run `pytest -m "not slow"` after the focused tests.
- For runtime smoke testing, keep runs small, use temporary checkpoint/log
  directories, and avoid expensive long training unless explicitly requested.

