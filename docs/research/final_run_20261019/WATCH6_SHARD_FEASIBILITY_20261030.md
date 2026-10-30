# Watch6 main-shard feasibility — 2026-10-30

Outcome: NOT_READY — no valid main shard packets could be prepared under the unchanged
registered forecast and whole-pair execution contract. No dispatch or allocation.

The task requested fresh main packets from the genuine transport01 measurements,
with exactly 96 paired sessions, the frozen seeds, bars and one Mac look. Those
scientific constraints remain intact. All 18 supported operational layouts fail;
even one pair per shard exceeds the registered per-session wall. This records the
blocker rather than changing the registration, relabelling a failed canary, or
creating packets that cannot pass admission.

## Evidence

- Source/runtime pin: `79668a17962606f12bb674352c5e3bc04aa1702f`.
- Transport tree: `d2992d6a24dd548d2be1cd546b324e33aeb454cb`.
- Science identity: `8ac54d173be3bdc85b2f280f03d35a4eeb69aa6307aa96d182ee85e14a3e47ff`.
- Namespace: `t34-watch6-exact-stop-efficacy-20261030-v2`.
- Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/transport-qualify/watch6-canary-transport01/output-canary/receipt.json`, SHA256 `eacbda01243305e692c895e9d520d8b050e1194e6b6c0e86510f695290945575`.
- Performance export: same output root, `exports/performance.json`, SHA256 `d715c16ec076747d98cc6bba590f6448151e5e46cd8cf287b11f1895c1b17041`.
- Log: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/watch6-search-canary-watch6-canary-transport01.log`.
- Audit artifacts and complete 18-layout enumeration: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/watch6-shards/feasibility.json`.

All eight recovered export payloads were verified against receipt path/size/SHA,
their sidecars and completion markers, without opening pair outcomes. All nine
uploaded input bytes and their verified transfer records match. Manifest binding,
current science/allocation and current/pinned transport identities agree. The
three completed pairs were excluded operational fixtures, not main scientific
units. Staging: **113.554491s**, transfers **41.713365s**, source delivery
**69.546083s**. Both original timing bars pass. Stop is verified and cleanup error
is null. The retained measurements are genuine; none were fabricated or altered.

## Measured costs and exhaustive layout result

| Profile | Branch seconds | Bare main s/frame | Exact-stop main s/frame | Pair forecast seconds | Minimum base minutes | Doubled wall minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 | 4.017958751000 | 0.044284285503 | 0.044432721723 | 70707.85212134 | 1494 | 2988 |
| W6 | 4.144422236000 | 0.045433177810 | 0.044650674470 | 72912.82371091 | 1540 | 3080 |
| W8 | 5.379818594000 | 0.060529387857 | 0.061153582753 | 94715.50008110 | 1994 | 3988 |

The existing `performance.forecast` rule is unchanged:

```text
pair_seconds = 14400 * (bare_main_s/frame + exact_stop_main_s/frame)
               + ceil(14400/5) * 6 * measured_branch_seconds
base_minutes = ceil(20 + 1.25 * ceil(pairs_per_shard/workers) * pair_seconds/60)
effective_wall_minutes = 2 * base_minutes
```

The branch measurement is the largest whole H300 batch, deliberately used as a
singleton bound without division by branch count. The physical six-branch maximum,
25% forecast margin, 20-minute setup/export allowance and owner's separate 2x
wall/unit/reserve rule are retained. These are conservative fixture-derived
forecasts, not observed full-pair durations or all-state guarantees.

| Workers / base minutes | 3 shards | 6 shards | 12 shards | 24 shards | 48 shards | 96 shards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 | 11805 | 5913 | 2967 | 1494 | 1494 | 1494 |
| W6 | 9135 | 4578 | 3059 | 1540 | 1540 | 1540 |
| W8 | 7913 | 3967 | 1994 | 1994 | 1994 | 1994 |

Every nonempty shard needs at least one whole pair/wave. The best one-wave
profile, W4, needs **1,494 base minutes / 2,988 effective minutes (49.8h)**.
Both `performance.MAX_BASE_MINUTES` and `admission.admit` enforce base<=660min /
effective<=1,320min (22h); REGISTRATION_V2 explicitly retains room below a 24h
provider lifetime. More shards cannot shorten an indivisible pair. Longer walls
sufficient to meet the measured forecast would violate that session guard.
W4 would need at least **2.301688x** lower forecast pair cost to fit the current
one-wave ceiling. Additional Colab units alone do not resolve this constraint.

For comparison only, the rejected W4/96-shard arrangement would budget 2,390.4
aggregate base runtime-hours and 4,780.8 aggregate doubled wall-hours. At the
existing planning rate 2.5u/h this is 5,976 base forecast units or 11,952 units at
the doubled walls; reserve guards add 2,390.4 units (20% of doubled wall pricing).
The equally long per-shard W4/24-shard arrangement runs four pairs concurrently:
597.6 aggregate base hours / 1,195.2 doubled hours, 1,494 / 2,988 units, plus
597.6 reserve units. Neither arrangement is feasible. Full wall/unit figures for
all layouts are in feasibility.json. These are aggregate host-hours, not elapsed
completion predictions under the account cap of three, and use planning prices,
not fresh account debit measurements. No operational layout has been adopted.

## Why the retained report does not produce dispatchable main packets

`performance.forecast(report)` reproduces the original exact NOT_READY refusal.
The receipt is INCOMPLETE / CHILD_NONZERO_EXIT because the final sizing print
exited 1 after publishing performance.json. It has no canary_evidence object.
It remains unchanged. `prepare.prepare` requires PASS and verified stop, and the
independent `research.colab_cli.canary.check` also requires clean PASS, actual
source-bound family evidence and matching recovered exports/inputs.

The actual one-pair W4/96 diagnostic with a 2,988-minute effective wall was sent
through the real package dispatch entrypoint with `--dry-run`. It exited 1 at
`admission.admit`: `forecast exceeds safe provider session length`, before the
operator dispatcher or allocation. Direct family-canary validation independently
refused `family canary did not complete with verified release`; evidence recovery
refused `canary terminal job/source/generation identity differs` because the
terminal is failed. The diagnostic JSON is named REJECTED-layout-diagnostic.json;
it is not a prepared main packet. Its output directory remains absent.

No additional unchanged canary was prepared or run: the complete measurements
already establish this operational infeasibility. A repeat does not fix a whole
pair that cannot fit the formula. The report can be retained for future planning,
but cannot be represented as a passing family receipt under current checks.
A prospective byte-identical performance optimization or full-state continuation
inside a pair could preserve the scientific design, but requires implementation,
qualification and independent Argon review. The latter changes the current rule
that partial pairs are abandoned rather than resumed. Neither is implemented or
silently authorized by this task. Changed runtime/source/input bytes would require
appropriate new same-pin qualification; the current failed receipt cannot be
promoted into proof. The confirmation N, seeds, bars and one look remain fixed.

## Verification and handoff

- Saved-evidence verifier: PASS; eight immutable recovered exports, nine verified
  inputs, current identities and all 18 infeasible layouts. Command, from the
  worktree: `PYTHONPATH=. /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/watch6-shards/verify-feasibility.py`.
  This create-only command already ran; use a fresh output copy if repeating.
- Existing forecast/canary regression selection: **14 passed, 36 deselected in
  2.07s**. Exact command: `env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_watch6_search_efficacy.py -k 'forecast or measured_canary or w6_can_win'`.
- Actual negative dry-run: expected exit1 before operational side effects.
  `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch6_search_efficacy.dispatch /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/watch6-shards/REJECTED-layout-diagnostic.json --dry-run`.
- Independent read-only architectural feasibility analysis agreed with the exact
  18-layout arithmetic and absence of an implemented within-pair resume path.
  This is not independent Argon product-code review; no product code was edited.

**Exact live dispatch commands: none.** Zero valid main packets exist. The
requested preparation is blocked; no live dispatch command is safe to hand off.

Only this new status document is committed. No runtime/product/transport source,
registration/ruling, immutable prior receipt/package, checkpoint, serving pin,
roster, main bank, or tmux session changed. No main merge or push is performed;
the status document is published only on astra-watch6-search-efficacy.
