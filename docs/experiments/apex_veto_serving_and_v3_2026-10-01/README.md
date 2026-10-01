# Apex veto: web serving qualification and v3 tail-aware screen, 2026-10-01

Source: `apex-safety-and-eval-tooling` @ `7a65b20`, local only.

Authorization: the user's standing loop instruction ("make the best decisions
in a loop for me until i say stop"). Both runs were on AC, with shared CPU
slots 1 and 2.

## 1. Web serving run: **SERVING_PASS (serving_path_qualified = true)**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-serving-20261001/run-v1` |
| Audit | `audit/audit.json` (sha256 `c7f612e8f38b8913…`) |
| Pre-registered criteria | S1–S6, all PASS, 0 failures |
| Episodes | 50, 0 failures |

What ran:

- 50 episodes through the real `web.backend` `GameSession` and
  `app._apply_control` dispatch: 1 Watch episode as served, plus 49 Play
  episodes with a scripted stand-in for the human.
- Parity probes against `tournament_eval.rollout`.
- The opt-in hook wraps only the receipt-bound champion (`43d4e2c5…`) with
  the receipt-bound v2 veto (source `1b62d15c…`). It fails closed for any
  other checkpoint or wrapper.

The veto is therefore **release-ready** for the Watch hero. The pending
release step needs explicit user approval:

- **Release:** restart the server with
  `SNAKE_SERVE_VETO_WATCH_HERO=1 ./venv/bin/python web/serve.py`. Startup logs
  `safety-veto-serving: active=True … strict_checkpoint_match=True`.
- **Rollback:** restart without the variable.

`SNAKE_SERVE_VETO_PLAY_AI` stays off. The strict gate measured one wrapped
hero only, so wrapping every Play AI snake would need its own evaluation.

## 2. v3 tail-aware veto Tier-1 screen: **NOT_ADVANCED**

| Item | Value |
|---|---|
| Root | `snake-dqn-artifacts/apex-veto-v3-screen-20261001/run-v1` |
| Receipt | sha256 `9b1b9b43ff7932ff…` |
| Namespace | `apex-veto-v3-screen-v2` (v1 consumed by a runaway test, quarantined unread; see protocol revision 4) |
| Episodes | all 264 completed; determinism 24/24 identical |

Arm A is champion + v2 (the strict-passed configuration); arm B is champion +
v3.

| Mix | A (v2) | B (v3) | Δ (95% CI) | Worlds better / equal / worse |
|---|---:|---:|---|---|
| frozen | 121.8 | 125.5 | +3.7 [−3.8, 11.3] | 1 / 39 / 0 |
| mixed | 113.8 | 113.8 | 0.0 | 0 / 40 / 0 |
| scripted | 153.2 | 148.6 | −4.6 [−13.1, 3.9] | 1 / 37 / 2 |

Holm passes in no mix.

**Interpretation.** Tail-aware counting almost never changes a decision; it
differed in 4 of 120 worlds. The remaining self-collision deaths occur after
the safe-and-spacious set is already empty, so the trap is entered earlier
than any one-step reachability check can see. The next veto idea, if pursued,
needs multi-step look-ahead (a short search over the next few moves), not a
better one-step count. v3 is retired.
