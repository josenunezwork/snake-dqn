# B2 snapshot process-configuration correction — pre-data

This is an additive correction to the published preparation at `4971234d`, not a
study result or an amendment to PLAN_v3's frozen design. No B2 family canary,
production trajectory, source snapshot export, bank look or Colab allocation has
run. The pending design and blocked Mac preflight remain pending. Prior receipts
and the reviewed HANDOFF are retained unchanged.

A fresh spawned branch worker starts with the process-wide default `AppConfig`
(mechanics v1). The natural-root simulator is constructed from the deployment
configuration (mechanics v2). Native v8's `check_live_rule_config(sim)` checks the
simulator against process-wide `GameConfig`, so restoring only `sim.cfg` and
policy carry would refuse a real branch in a fresh worker.

`b2_engine.WatchDriver.snapshot()` now emits `bl-b2-state/v2` and includes the
immutable `AppConfig`. `restore()` requires that configuration and installs it
before any branch action or native-rule check. It does not re-open a YAML file or
resolve a checkpoint path. Simulator geometry, policy carry, RNG and padded-grid
alias restoration retain their prior behavior. Version-1 snapshots are refused;
none were produced by B2 production work.

The mocked regression `test_restore_replaces_spawn_defaults_with_snapshot_game_config`
serializes a mechanics-v2 state, resets the process to v1 defaults, proves the
real native configuration guard refuses that mismatch, then restores and proves
the guard accepts v2. It advances no game and performs no model inference.
The complete B2 mocked suite passes 52 tests. This is not real transition parity;
the required tiny Mac entrypoint preflight and same-pin natural-root family
canary still must run under Claude.

Restoration intentionally sets process-wide game configuration. Use it in the
isolated branch/fixture worker or exclusive offline merger, not a process hosting
a separate game with different configuration. C15 construction remains inside
its worker budget as required by PLAN_v3.

Future manifests must pin the corrected published source. No old preflight or
family proof may be inherited. The already doubled budget remains canary <=1 u,
original phase including canary <=20 u, all attempts <=40 u. No cap, N, seed,
root-selection rule, H300/H1440 interpretation, endpoint or serving state changed.
