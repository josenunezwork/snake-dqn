# snake-dqn

Multi-agent reinforcement learning that teaches snakes to play a slither.io-style game with a
distributed **Apex DQN** built from scratch — a hand-rolled actor/learner/buffer system, a
disciplined paired-seed promotion gate, and a live web app to watch the trained policy think in
real time.

![architecture](docs/architecture.svg)

![python](https://img.shields.io/badge/python-3.12-blue)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![code style](https://img.shields.io/badge/code%20style-black-000000)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## The headline result

The interesting finding wasn't a reward tweak — it was a diagnosis. The agent kept dying by
**self-collision once it got long**, and no amount of reward shaping fixed it, because it wasn't a
reward problem: the snake **couldn't see the trap in its observation**. Adding three bounded
BFS flood-fill **"free-space"** features (reachable open area for turn-left / straight / turn-right)
gave it that signal.

The model is a feedforward Dueling DQN. The original free-space evidence used a
pre-repair, alive-conditioned, narrow-opponent gate, so it is not comparable to
the current promotion metric. Read the dated results, reward-label mismatch, and
corrected CNN/GRU record in [Project history and durable findings](docs/project_history_and_findings.md).
The champion remains the incumbent baseline until a candidate passes the current gate.

## Quickstart

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# Watch the trained champion play — or play against it yourself — in the browser
cd web/frontend && npm install && npm run build && cd ../..
python web/serve.py            # -> http://localhost:8000   (or: make web)
#                                open the "Play" tab to steer a snake and top the leaderboard

# Train headless (no GUI needed)
python src/main.py --headless --episodes 100000

# Gate a candidate against the incumbent champion (paired seeds, all 3 opponent mixes).
# --gate exits 0 iff the candidate passes the promotion rule, so CI can branch on it.
SNAKE_DQN_DEVICE=cpu python src/scripts/tournament_eval.py saved_snakes/best_apex_fs.pth \
  --gate --frames 3000 --seeds 0,1,2,3,4,5,6,7,8,9 \
  --json-output logs/gate_candidate.json

# How many seeds do you actually need? (power analysis on the baseline)
python src/scripts/tournament_eval.py --pilot --frames 3000
```

## The web app

A **server-authoritative** UI: the Python engine is the single source of truth, and the React
frontend is a pure view that renders frames streamed over a WebSocket — no game logic or model
inference runs in the browser. Seven panels — the arena is always on screen, with six tabs beside it:

- **Game** — the live arena; the hero snake is outlined.
- **Play** — *take the controls yourself.* Steer a snake with the arrow keys / WASD against the
  trained AI, boost with space, and submit your final score to a **SQLite-backed leaderboard**
  (see [Human play](#human-play--leaderboard)).
- **Inspector** — the hero's 58-/61-D state vector (grouped + labeled), live Q-values → chosen
  action, and the free-space features.
- **Raster** — what a `raster31v2` snake actually sees: the heading-rotated **ego-centric 31×31
  tactical stack** (head centred, facing up), as a colour-coded composite or one isolated channel.
  The redesign's candidate observation (`src/simd_env/featurizer.py`); shows a placeholder for the
  61-D vector champion.
- **Network** — live `forward_with_activations` heat-mapped across input / hidden / output layers.
- **Dashboard** — the eval leaderboard parsed from `logs/eval_*.json` + the checkpoint inventory.
- **Controls** — play/pause, speed, ε, hero selector, checkpoint loader, watch↔train toggle.

### Human play + leaderboard

The **Play** tab pits *you* against the trained snakes on the same engine. Movement is decoupled
from any UI toolkit ([`HumanSnake.apply_direction_input`](src/game/human_snake.py)), so the browser
drives it over the WebSocket. A run freezes until your first key ("press an arrow to start"), your
death ends the run (AI opponents keep respawning), and the score is computed **server-side** from
length, food, kills, and survival — so it can't be spoofed by the client.

Scores persist in a small dedicated SQLite database ([`ScoreStore`](src/data/score_store.py),
`scores.db`), separate from the large experience-replay DB. REST surface:
`GET /api/leaderboard`, `GET /api/players/{name}`, `POST /api/scores`.

> A screenshot/GIF lives best at `docs/web_app.png` — the app is one command away (`make web`).
> See [web/README.md](web/README.md) for the design.

## Architecture

### Distributed Apex DQN (`src/training/`)

N actor processes (CPU, varied ε) → a `SumTree`-backed prioritized buffer (O(log N) sampling,
n-step returns) → one GPU learner; weights broadcast back to actors. Implements:

- **Double DQN** (online selects, target evaluates)
- **Dueling architecture** (separate value / advantage streams)
- **Prioritized Experience Replay** with importance sampling
- **N-step returns** and target networks

A `Policy` ABC and a `BaseReplayBuffer` ABC keep the pieces swappable; `SnakeFactory` injects the
shared policy into snakes via DI; configuration is immutable frozen dataclasses.

### Models and state representations

The **production** model is the feedforward Dueling DQN `ApexNetwork` (`src/model/apex_network.py`)
— the only one with a promoted checkpoint. Its state is selectable via `use_free_space`: the 58-D
hand-crafted vector, or the 61-D variant that appends the three free-space features
(`input_size: 61`, `use_free_space: true`).

Alongside it, the in-progress redesign adds `RasterDuelingNetwork` (`src/model/raster_network.py`),
a conv encoder over the ego-raster observation. It is **not** yet promoted — the gate decides.

A note on history, since earlier docs (including this README) got it wrong: **GRU/DRQN** was trained
and lost the older paired frozen-opponent trials to the feedforward 61-D model. Those trials predate
the repaired gate, so they do not settle future recurrent experiments. The oft-repeated claim that
**CNN** variants "were evaluated and lost" is **false** — git archaeology found the CNN code was
written and deleted within a week, never trained, with zero checkpoints. The CNN question remains
open, which is why the redesign re-tests it against an MLP control arm
([blueprint](docs/ml_redesign_blueprint_2026-07.md) Appendix B, claims 1 and 10).

The `Snake` class is split by concern: entity/lifecycle (`snake.py`), observation
(`snake_state.py` → `SnakeStateMixin`), and reward (`snake_reward.py` → `SnakeRewardMixin`).

### Evaluation: the promotion gate

`src/scripts/tournament_eval.py` is the differentiator. Naïve self-play eval inflates skill (a Red
Queen effect — opponents improve too, and survival saturates at the frame cap). This harness runs
greedy (ε=0, deterministic) rollouts on **paired seeds** — every candidate plays the identical
worlds as a `--baseline` (default: the incumbent champion) — and reports per-seed deltas with a
t-distribution 95% CI.

The headline metric is the **mass integral**: mean per-frame mass over the **total** horizon, with
dead frames contributing **0**. This replaced an alive-conditioned mean mass under which *dying rich
outranked surviving* — that metric survives only as the `mean_mass_alive` legacy diagnostic.

Opponents are **diverse** rather than five clones of one checkpoint. The candidate is evaluated
round-robin over three mixes:

| mix | opponent slots |
|---|---|
| `frozen` | cycled over the `--opponents` checkpoint pool |
| `scripted` | all `greedy_food` scripted anchors — ungameable, and it can't drift |
| `mixed` | alternating frozen-pool checkpoints and `random_safe` scripted slots |

The scripted anchors are why the gate can calibrate *itself* (champion > greedy anchor > random_safe)
and why its tests run hermetically with no checkpoints at all.

**Promotion rule** (printed always, machine-enforced with `--gate`, which exits 0 iff it passes):
promote iff the paired mass-integral delta is **> 0 at 95% CI on ≥ 2 mixes** *and* there is **no
regression vs the scripted anchor mix**. `--pilot` runs the baseline alone and recommends a seed
count for a 3% minimum detectable effect. Behavioral probes (boost fraction, death causes, kills,
entrapment) ride along on every run.

This harness is what caught a mislabeled reward contract during development.

## Train your own on an H100

The winner is feedforward, so it runs on the fast distributed path (1 GPU learner + many CPU
actors). See [docs/h100_training_recipe.md](docs/h100_training_recipe.md) for the exact warm-start,
config, and gating recipe.

## Project structure

```
src/
├── core/        immutable config, device manager
├── model/       ApexNetwork (Dueling DQN) + mixins, RasterDuelingNetwork, InferenceAgent
├── training/    Apex policy, actor, learner, SumTree/PER buffers, curriculum, PQN trainer
├── game/        snake entity + state/reward mixins, game loop, factory (DI)
├── simd_env/    the vectorized redesign env:
│                  batch_sim      NumPy-vectorized, cell-exact batch simulator
│                  featurizer     dual-scale ego-raster observation (raster31v2)
│                  gpu_featurizer the same featurizer in torch, built on-device
│                  eval_engine    batched rollouts for the gate
│                  parity         golden-replay harness: live game vs batch sim
│                  live_adapter   live GameState -> ObsInputs bridge
└── scripts/     apex_train, train_pqn, sweep, tournament_eval (promotion gate),
                 bench_simd, widen_input + warm-start tooling
web/             FastAPI backend + React/TS frontend (the live UI)
configs/         YAML configs (free_space_v2 = production; eval_free_space = the gate arena)
docs/            algorithm, redesign blueprint, SIMD env spec, H100 recipe, findings
saved_snakes/    champion + frozen eval opponents
```

### Redesign in progress

The redesign provides a NumPy-vectorized simulator, an ego-raster observation, and
a PQN trainer alongside the incumbent Apex path. The candidate is unpromoted: the
champion stays incumbent until something beats it under the repaired gate. See the
[redesign blueprint](docs/ml_redesign_blueprint_2026-07.md), the
[SIMD contract](docs/simd_env_spec.md), and the
[dated findings record](docs/project_history_and_findings.md).

## Development

```bash
make install-dev    # deps + pre-commit hooks
make test           # pytest
make lint           # black --check + isort --check + flake8
make format         # auto-format
cd web/frontend && npm test    # frontend unit tests (vitest)
```

CI runs lint + Python tests + the frontend build **and its vitest suite** on every push
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## References

- **[docs/ml_algorithm.md](docs/ml_algorithm.md)** — the algorithm design in depth: dueling network,
  Double-DQN + n-step targets, PER, action masking, the distributed topology, and design directions.
- **[docs/project_history_and_findings.md](docs/project_history_and_findings.md)** — dated results,
  corrected historical claims, and their provenance.
- [Ape-X](https://arxiv.org/abs/1803.00933) — Distributed Prioritized Experience Replay
- [Double DQN](https://arxiv.org/abs/1509.06461) · [Dueling Networks](https://arxiv.org/abs/1511.06581) · [PER](https://arxiv.org/abs/1511.05952)

## License

MIT
