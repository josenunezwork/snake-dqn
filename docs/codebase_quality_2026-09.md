# Codebase quality handoff — September 2026

**Status:** complete for the scoped quality pass. This note describes the
incremental work only; the checkout had broad pre-existing changes.

**Audience and outcome.** This is for maintainers reviewing or continuing the
work. It assumes the two independent stacks: Apex uses `vector61`, replay, and
a target network; raster/PQN uses `raster31v2` and no replay, PER, or target
network. The outcome is four explicit operational boundaries: checkpoint writes,
web checkpoint selection, browser identity storage, and script/package startup.

The coordinating pass recorded
`/tmp/snake-quality-start-files.tar`,
`/tmp/snake-quality-start.patch`, and the incremental
`/tmp/snake-quality-changes.patch`. Use them to separate this pass from
existing work; they are temporary operator artifacts, not repository inputs.

## Durable rolling checkpoint writes

[`src/model/checkpoint_io.py`](../src/model/checkpoint_io.py) owns the
rolling-checkpoint protocol. `atomic_torch_save()` writes beside the
destination, flushes and syncs the temporary file, then swaps it in with
`os.replace`. If serialization, sync, or replacement fails, including through
an interruption, it removes the temporary file and retains the old checkpoint.

[`CheckpointManager`](../src/model/checkpoint_manager.py) uses that helper for
Apex, and [`PQNTrainer`](../src/training/pqn_trainer.py) uses it for
`latest_pqn.pth`. The two formats retain their envelopes and metadata, while
the write mechanism is no longer duplicated. `save_checkpoint_dict()` adds
the Apex marker to a copy, preserving its caller's mapping.

The focused coverage in
[`tests/test_checkpoint_io.py`](../tests/test_checkpoint_io.py) and
[`tests/test_pqn_trainer.py`](../tests/test_pqn_trainer.py) verifies replacement,
cleanup, parent-directory creation, and failed saves.

## Canonical web checkpoint catalog

[`web/backend/checkpoints.py`](../web/backend/checkpoints.py) owns the
catalog and resolver used by the API inventory
([`web/backend/metrics.py`](../web/backend/metrics.py)) and WebSocket loader
([`web/backend/session.py`](../web/backend/session.py)). The only loadable
names are direct `saved_snakes/*.pth` files and
`runs/**/latest_pqn.pth` outputs.

The resolver accepts only a name it has advertised. It rejects traversal,
absolute paths, and symlinks whose canonical targets leave the allowed roots.
For an in-root symlink it retains the discovered lexical
`runs/**/latest_pqn.pth` name while using the canonical target for containment
and loading, so a client can reuse the exact alias it received.

Web saves deliberately have a different contract from rolling trainer saves:
[`GameSession.save_weights`](../web/backend/session.py) exclusively creates a
fresh `web_train_*.pth` name, adds a suffix on collision, and removes a
partially written new file if saving raises any exception. No existing web save
is overwritten. Session locking and state snapshots are separate from this
catalog boundary.

The catalog narrows file selection; it does not make arbitrary checkpoint
payloads safe to deserialize. The existing compatibility `torch.load` paths
remain, and a safe-loading migration is deferred.

## Browser identity storage

[`web/frontend/src/identity.ts`](../web/frontend/src/identity.ts) is the
single owner of player-name normalization, stored display names, and the opaque
per-browser client ID used by the play API. This removes divergent whitespace
and storage handling across components.

When browser storage is disabled, blocked, or full, the open page still receives
one runtime client ID and remains usable. The identifier is browser-local
convenience, not authentication or cross-device identity. Focused coverage is
in [`identity.test.ts`](../web/frontend/src/identity.test.ts) and
[`api.test.ts`](../web/frontend/src/api.test.ts).

## Consistent package and configuration startup

[`pyproject.toml`](../pyproject.toml) now installs `snake-train` from
`src.main:main` and explicitly discovers `src` packages. The installed
command therefore follows the repository's actual import layout.

Affected operational scripts call
`load_and_initialize_config()` from
[`src/core/config_loader.py`](../src/core/config_loader.py) instead of each
repeating load, global initialization, and legacy application. YAML validation
and the `GameConfig` compatibility view now initialize through one startup
operation. The compatibility helper remains for external callers that already
have a constructed configuration.

## Invariants retained

- Apex remains the incumbent `vector61` path with replay, PER, and target
  networks.
- Raster/PQN remains `raster31v2`, synchronous, and replay-free; its
  checkpoint contract continues to serve inference and the SIMD gate.
- [`src/scripts/tournament_eval.py`](../src/scripts/tournament_eval.py)
  remains the promotion authority. Its paired-seed, total-horizon
  mass-integral, opponent-mix, and scripted-anchor rules did not change.

## Verification evidence

| Check | Result |
| --- | --- |
| CPU non-slow Python suite | 1,881 passed, 3 deselected in 90.52 s; one Starlette warning |
| Frontend tests and build | 166 tests passed; production build passed |
| Python focused regression set | 339 tests passed |
| Static checks | `make lint` and `git diff --check` passed |
| Installed command | Built wheel installed and `snake-train` smoke-tested from an external working directory |
| Independent review | Frontend and backend/checkpoint reviews found no remaining actionable issue |

The final Python rerun is logged at
`/tmp/snake-quality-final-post-review-pytest-rerun-20260906T202700Z.log`.
The immediately preceding full run had one failure in the unchanged
`tests/test_apex_buffer.py::TestBufferProcessActorRejections::test_rejects_malformed_actor_batch_and_keeps_processing`:
its two-second stats deadline returned an empty mapping. Five isolated reruns
passed, the subsequent full run above passed, and the relevant files were
unchanged. Treat that as a timing-sensitive test flake to monitor; no production
change was made to suppress it.

Typed PQN rollout containers are a separate readability improvement and were
intentionally deferred.
