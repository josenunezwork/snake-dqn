# Codex and Claude Operational & Research Skills (`agent-skills`)

**Dates:** 2026-10-18
**Verdict:** Active repository backup mirror (`18` kept files = `169,999` bytes under `docs/codex-skills/`) of procedural `~/.codex/skills/snake-dqn-*` runbooks, with campaign-specific budgets (`Q0/Q1`), legacy queues (`colab_queue`), and pre-`2026-10-30` ceremony superseded by `AGENTS.md` and `docs/STATUS.md`.

## What was tried
- Built and reviewed (`2026-10-18` Fable + Claude pass) a `docs/codex-skills/` mirror (`README.md` + `17` `SKILL.md`s; `snake-dqn-colab-jobs` tracked under `colab-ops`) synced from `~/.codex/skills/snake-dqn-*`.
- Codified codebase, `58`-D state / `6`-action mechanics, `GameState` / `BatchSim` / `GridBatchSim` parity, Mac-vs-x86 float32 tie divergence (`~1` in `50` H10000 episodes), and immutable SHA-256 governance docs.
- Codified M5 Pro (`18` cores, `20`-core GPU, `64` GB) thread caps, slot locks, AC/`caffeinate -dims` rules, served-checkpoint pins (`b8-s3`, `frp3-s12`, `champion`), `ego2s` fps ceilings, sequential OBF/Phase R gates, and throughput baselines.

## What happened
- **Durable operational baselines (`ACTIVE`):** Preserves M5 Pro single-thread env exports (`OMP`/`OPENBLAS`/`MKL`/`NUMBA`/`VECLIB=1`), `2`-slot strict / `3`-slot Tier-1 CPU locks, and `ego2s` speed ceilings (`24` fps for `<= 6` driven AI, `12` fps for `7–12`).
- **Gate & throughput references:** Codifies survival band `v2` (`0.05` pooled NI, `0.075` per-mix, `0.30` floor) and reference speeds (`~5.2–5.5k tr/s` at `~21 upd/s` Mac MPS, `78–83 upd/s` isolated / `~62 upd/s` = `3.0x` full A100, `14.9 -> 8.4 ms` = `1.74x` RTX `4090`).
- **Aged campaign parameters (`SUPERSEDED`):** Hard-coded `Q0+Q1` unit caps (`<= 150` +`25` units, `Q0` `40`, `Q1` `<= 60`), legacy Drive `research/colab_queue/`, RunPod-for-T34, and pre-`2026-10-30` pre-registration ceremony were overridden by `research/colab_cli` and the Fast-process rule in `AGENTS.md`.

## Why
Procedural skills encoding timeless codebase/hardware contracts (`6` relative actions, parity tests, M5 Pro thread caps, SHA-256 immutability) remain durable across campaigns, whereas skills embedding point-in-time credit budgets or transport queues go stale whenever operating rules advance.

## What to reuse / what not to repeat
- Keep procedural invariants (state/action specs, parity checks, Mac thread caps, serving fallbacks) in `SKILL.md` files, while treating `AGENTS.md` and `docs/STATUS.md` as the sole authority on active campaign rules, budgets, and ceremony.

## Full record
Original: `git show 90b3db94:docs/research/history/agent-skills.md` | Primary sources: [README.md](../../codex-skills/README.md), [SKILL.md](../../codex-skills/snake-dqn-mac-ops/SKILL.md), [SKILL.md](../../codex-skills/snake-dqn-evaluation-gates/SKILL.md)
