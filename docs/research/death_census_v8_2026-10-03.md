# Death census: what still kills or limits the champion under the released v8 veto (2026-10-03)

This is a Tier-0 development diagnostic. It is not a screen, it is not gate evidence, and it
produces no candidate. It plays the released Watch configuration and records every hero
death, the veto's activity near death, the trap horizon and the hero's mass trajectory. The
released configuration is the champion plus `free-space-veto/v8-space-and-head(lambda=8.0)`.
The goal is to choose the next lever from evidence. It mirrors the v5 census
(`docs/research/death_census_v5_2026-10-02.md`) and reuses that census's analysis code
unchanged.

## Setup

| Item | Value |
|---|---|
| Code | `research/death_census_v8_20261003/census.py` (branch `census-v8`, run from clean commit `85041ac`). Read-only follow-ups: `post_analysis.py`, `length_aware.py`, `closure_check.py` |
| Tests | `tests/test_death_census_v8.py`: 32 tests plus 1 opt-in tiny-world SIMD integration test (`SNAKE_CENSUS_TINY_ROLLOUT=1`; it passed). Every test that calls `main` makes the runners fatal. v5 census tests still pass. |
| Review | Independent read-only review returned GO with no blockers. SHOULD_FIX 1–5 and the cheap NITs were applied before the run (`85041ac`). |
| Engine | SIMD: `run_simd_eval(vector61=True, hero_safety_veto="v8", hero_safety_veto_lambda=8.0, vector61_forward="rowwise")`. This path is bit-exact with the live hook (H5000 parity 24/24). Released identity checked at launch: `VARIANT_RELEASED_DEFAULT == "v8"`, `V8_LAMBDA == 8.0`. |
| Worlds | H5000, `promotion-v2-watch-rect`, strict balanced rosters, **60 worlds per mix** (frozen, scripted, mixed), shared across mixes, so 180 episodes |
| Namespace | Fresh `apex-veto-v8-census-v1/worlds`. Fail-closed disjointness check: the v8 serving lane's `seed_report` (every earlier bank, including v8 strict and its observed seeds) plus the first 1,000 seeds of the v8 web-serving, v5-census and v2 trap-horizon domains (and their smokes), the seeds saved in the v8 serving intent, and the smoke domain. Result: disjoint. |
| Output | `snake-dqn-artifacts/death-census-v8-20261003/run-v1/` (`shard-{1,2,3}/`, `merged/summary.json` sha256 `a85296e5b877609d…`, `merged/post_analysis.json`, `merged/length_aware.json`). Not committed. |
| Compute | 3 shards, one per CPU slot (pool 3: slots 3/1/2), 2 threads each, `caffeinate -dimsu` in tmux. AC power and lid open were checked at launch and polled every 15 s. 0 safety pauses. The thermal guard made 105 admission checks, 0 not OK. |
| Wall time | 17:47–18:22 UTC, **about 35 min**: simulation 14 min per shard, analysis 19–21 min per shard |

**Capture.** All capture is read-only.

- `Vector61SimdPolicy._apply_veto` is wrapped. The original runs first and its result is returned unchanged. For each hero decision the wrapper takes the same snapshot the v5 census did, plus v8's reason.
- v8's reason comes from the live hook's counter deltas around the call: `kept`, `no_spacious`, `v2_rule`, the landing reasons, `v7_rerank` or `head_veto`, plus head-risk flags. The per-decision cost is the hook's own `apply_seconds_total` delta.
- `BatchSim._resolve_collisions` is wrapped as well. After detection and before deaths are applied, it keeps the post-move world of every env whose hero dies on that frame.
- The `frame_observer` seam gives the per-frame mass, boost, food, kills, frames since food and nearest food.

**Checks that passed:**

- captured decisions equal probe decisions in all 180 episodes;
- the trajectory mass integral equals the record's in all 180;
- the post-move capture is at the fatal frame in all 69 deaths;
- the one-frame counterfactual reproduces the actual cause in **69 of 69** deaths;
- the hero-move model reproduces **8,211 of 8,211** captured transitions;
- the model predicts `self` for all 53 self-death fatal actions.

**Analysis.**

- **Every death:** the v5 census's exact one-frame counterfactual.
- **Self deaths:** `diagnose.analyze_death`, unchanged. It uses count-or-depth as the primary criterion, depth-only and count-only sensitivities, depth 40, and a 50,000-node budget.
- **New in this census:** a **300 s CPU cap per death**. After the cap every remaining search is reported as unknown, never guessed. 17 of 69 deaths hit the cap, all of them self deaths.

## Results

### Outcomes (180 episodes)

| Mix | self | head_on | enemy_body | wall | survived | Mean mass | Survival fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 20 | 1 | 3 | 0 | 36 | 311.5 | 0.824 |
| scripted | 20 | 0 | 6 | 0 | 34 | 350.9 | 0.828 |
| mixed | 13 | 0 | 4 | 2 | 41 | 349.9 | 0.888 |
| **pooled** | **53 (29.4%)** | **1 (0.6%)** | **13 (7.2%)** | **2 (1.1%)** | **111 (61.7%)** | **337.4** | 0.847 |

Comparison with v5 (descriptive only: different engine, namespace and sample):

| | self | head_on | enemy_body | wall | survived |
|---|---:|---:|---:|---:|---:|
| v5 census (48 episodes, live) | 75.0% | 10.4% | 2.1% | 2.1% | 10.4% |
| v5 strict final (807 episodes) | 77.6% | 7.8% | 3.7% | 0.7% | 10.2% |
| **v8 census (180 episodes)** | **29.4%** | **0.6%** | **7.2%** | **1.1%** | **61.7%** |

v5 census mass by mix was frozen 205.5, scripted 144.4 and mixed 265.4. v8's mass here is in
line with the v8 strict gate's descriptive means (337 / 356 / 369).

- **Head-on deaths are gone.** One remains (frozen, length 146): v8 was in `no_spacious` and
  no legal alternative had an escape.
- **Self deaths are now a minority.** The remaining deaths are all late enclosures:
  - fatal lengths 235–881 (median 517);
  - median frame of death 3,088.
- **Enemy-body deaths doubled in share** (3.7% to 7.2%). In all 13:
  - v8 was in `no_spacious`;
  - the hero hit a body cell that was already occupied before the move;
  - no legal action survived the frame.

  These are encirclements by other snakes, not steering mistakes.

### What the veto saw near death

| Cause | n | v8 in `no_spacious` at the fatal decision | Veto changed the action in the last 1 / 10 / 40 decisions | Legal alternative survives the frame (with escape) |
|---|---:|---:|---|---:|
| self | 53 | 53 | 0 / 1 / 3 | 1 (0) |
| enemy_body | 13 | 13 | 0 / 2 / 3 | 0 (0) |
| wall | 2 | 2 | 0 / 0 / 2 | 0 (0) |
| head_on | 1 | 1 | 0 / 0 / 1 | 1 (0) |

- **v8 had stopped acting before every death.** Every death occurred in v8's `no_spacious`
  regime, where it leaves the base action unchanged.
- **The regime starts well before death.** Count the consecutive `no_spacious` decisions that
  end at the fatal one:
  - self deaths: median **28**, range 1–81, never censored by the 120-decision window;
  - enemy-body deaths: median 4.
- **No fatal action was a boost.** At the PNR the taken action was a boost in 8 of 53 deaths.

### Self deaths: point of no return

| | v5 census | **v8 census** |
|---|---:|---:|
| Self deaths / episodes | 36 / 48 | 53 / 180 |
| PNR exact | 30 (median 7) | 36 (median **4**, q25 2, q75 12) |
| PNR ≤ 8 / 9–20 / 21–40 / > 40 / unknown | 17 / 9 / 4 / 0 / 6 | 23 / 9 / 4 / 0 / 17 |
| Fatal choice at the PNR: boost / normal speed | 10 / 14 | **5 / 12** |
| Taken action itself escaped at the PNR | 8 | **24** |
| Unresolved | 4 | 12 (budget or CPU cap) |
| Only escape at the PNR was surviving 40 frames | 33 | 41 |
| Count-only PNR exact (median) | 19 (21) | 29 (**12**, q25 3, q75 31) |

**The hero's own fatal choices fell from 67% of self deaths to 32%** (17 of 53).

- Every one of the 17 was in `no_spacious` at the PNR.
- 16 of the 17 could escape at the PNR only by surviving 40 frames inside a region smaller
  than `need`.
- 12 of the 17 were enclosed more than 20 frames earlier.
- The 5 boost fatal choices landed in 1–19 cells (`need` 160). Moving the same direction at
  normal speed escaped in 4 of them.

**Most remaining self deaths are closed by the world, not by the hero.**

- At the count-only PNR, the taken action still had a count-only escape in **33 of 35** known
  cases. The trap closed one frame later.
- `closure_check.py` covers the 27 exact cases. It re-searched from the hero's actual next
  state in the *previous* frame's world:
  - **16 of 27 still had an escape.** Other snakes moving closed the route. No static search
    at that frame could have seen it.
  - **11 of 27 were threshold cases.** The post-move state had a tail-aware count exactly at
    `need`, yet every continuation failed.

**v8's look-ahead had no signal.** At the count-only PNR, v8's one-step count called the taken
direction spacious in **27 of 29** exact cases.

**A length-aware gate would not have helped either** (`length_aware.py`).

- v8's hard gate uses `need = min(length, 160)` while the fatal lengths are 235–881. So I
  re-counted every legal action at every decision of every death window with the cap set to
  the hero's *length*.
- In only **4 of 69** deaths did some legal alternative reach a region of at least `length`
  while the taken action did not. Those 4 opportunities were 41–109 frames before death.
- In the crowded late game, almost no move leads to a region as large as the body. A bigger
  one-step cap is not the lever.

### Mass trajectory and non-death limits

| Pooled (per mix in `merged/summary.json`) | Value |
|---|---|
| Mean mass at frame 1,000 / 2,000 / 3,000 / 4,000 / 5,000 (dead = 0) | 178 / 334 / 426 / 485 / 509 |
| Survivors' final mass (n = 111) | mean 825 (frozen 787, mixed 811, scripted 882) |
| Survivors' late growth (frames 2,500–5,000) | median 0.15 mass per frame |
| Food eaten per 1,000 alive frames | 189 (scripted 201, frozen 184, mixed 184) |
| Net growth per 1,000 alive frames | 166 |
| Mass lost while alive (boost burn) | mean 74 per episode |
| Boost fraction | 7.2% |
| Nearest food (Manhattan, cells) | mean 3.6. More than 20 cells away on about 0% of frames. Over 100 frames without food on 0.1% of frames; over 300 frames never. |
| Food on the board | about 420 pellets |
| Kills credited | 10.4 per episode (scripted 22.2, mixed 5.6, frozen 3.5) |

- **Food is never the binding constraint.** The hero eats about one pellet every 5 frames,
  pellets are always about 3–4 cells away, and starvation stretches do not occur.
- **Growth is steady:** about 0.15 to 0.19 mass per frame.
- **Survivors are limited only by the horizon and the intake rate.** Their mass integral
  averages 432.
- **Kills are frequent** in the scripted mix: greedy_food snakes run into the hero's body.

### Per-decision veto latency (SIMD row views, v8 hook wall time, 762,078 decisions)

| | mean | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.51 ms | 0.46 | 0.89 | 1.31 | 1.89 | 29.7 |
| `kept` (758,625) | 0.51 | 0.46 | 0.89 | 1.30 | 1.80 | 29.7 |
| `v7_rerank` (306) | 1.87 | 1.85 | 2.67 | 3.67 | 3.92 | 3.9 |
| `head_veto` (130) | 0.45 | 0.37 | 0.72 | 2.12 | 2.41 | 2.4 |
| `v2_rule` (853) | 0.25 | 0.14 | 0.64 | 1.04 | 1.26 | 1.3 |
| `no_spacious` (1,891) | 0.01 | 0.01 | 0.02 | 0.02 | 0.04 | 0.4 |

- 189 decisions (0.025%) took more than 16 ms, all of them `kept`. Their cause was not
  investigated; scheduler or GC noise on the shared CPU is plausible.
- None took more than 100 ms.
- Actions changed: 1,562 of 762,078 decisions (0.20%):

  | Reason | Changed decisions |
  |---|---:|
  | `v2_rule` | 853 |
  | `v7_rerank` | 306 |
  | `landing_same_direction_normal` | 273 |
  | `head_veto` | 130 |

- The serving lane measured 0.65 ms per decision on the live path.

## Remaining failure modes and limits (ranked by estimated mass headroom)

**How headroom is estimated.** Headroom is the pooled mean mass-integral gain over all 180
episodes if the mode's deaths had not happened. There are two estimates:

- **hold:** the hero keeps its mass at death until H5000;
- **grow:** it also grows at its mix's median late survivor rate, about 0.15 per frame.

Both are upper bounds: they assume no later death. **Realistic** is my judgement of what a fix
could plausibly recover, given the mechanism evidence above.

| # | Failure mode / limit | Episodes | hold | grow | Realistic | Evidence |
|---|---|---:|---:|---:|---|---|
| 1 | **Self death, closed by the world** (`taken_escaped`: taken action still had an escape at the PNR) | 24 (13.3%) | 21.3 | 31.3 | medium | 16 of 27 count-PNR closures were caused by other snakes moving. v8's look-ahead called the move spacious in 27 of 29 cases. |
| 2 | **Encirclement by other snakes** (`enemy_body`) | 13 (7.2%) | 10.7 | 21.6 | medium | All 13 in `no_spacious` and hit pre-existing bodies, with no legal survivor on the fatal frame; enclosure began about 4 decisions before death |
| 3 | **Self death, unresolved** (unknown under the budget or CPU cap) | 12 (6.7%) | 12.4 | 17.9 | unknown | 11 of 12 have no `enclosed_early` verdict. They are probably long enclosures like rows 1 and 4. |
| 4 | **Self death, the hero's own fatal choice** (normal 12, boost 5) | 17 (9.4%) | 13.3 | 16.9 | **low (about 1–3)** | All in `no_spacious` at the PNR. 16 of 17 could escape only by surviving 40 more frames in a sub-`need` region, and 12 were enclosed early, so a shield mostly buys delay (about 5 mass per death). |
| 5 | **Growth rate of survivors** (non-death) | 111 (61.7%) | — | — | uncertain, potentially large | Food is abundant and near, with no starvation. The mass integral for survivors (432) is set by an intake of about 0.19 pellets per frame and a boost burn of about 74 per episode. Each +10% of intake is worth about +27 pooled mass. No measured ceiling exists. |
| 6 | Wall and head-on | 3 (1.7%) | 1.9 | 3.6 | negligible | Enclosure-type deaths with no surviving alternative |

All deaths together are worth at most +60 (hold) to +91 (grow) on a pooled mean of 337.4. That
is the ceiling for any safety-only lever. Rows 1–3 hold about 75% of it, and they share one
mechanism: **the space around the hero is closed by other snakes over the next 5–30 frames,
not by the hero's own move.**

## Recommendation: next lever

1. **Opponent-aware enclosure forecasting at serving time.** This is the recommended next
   lever and is grounded in rows 1–3.
   - **The idea:** extend v8's space check from "cells free now" to "cells that stay free while
     opponents move".
   - **A first version:** a pessimistic flood that treats as blocked every cell an opponent
     head can reach within *k* frames (v6's head reach, iterated). It ranks or vetoes moves by
     that forecast area, applied inside the existing v7 score and also in the `no_spacious`
     regime. A full opponent-modelling search over 10–30 frames is the heavier version.
   - **Why this and not alternatives:**
     - 16 of 27 exact closures were invisible to a static search at the PNR.
     - v8 called the fatal moves spacious in 27 of 29.
     - A bigger one-step cap fires in only 4 of 69 deaths.
     - The static fallback shield (row 4) buys about 40 frames per death.
   - **Upper bound:** about +45 (hold) to +70 (grow) pooled mass. A realistic first screen
     might recover a third of that.
   - **Cost:** a k-frame head-reach flood is cheap (a few ms with the 2×length v7 area cap).
     A real search needs a node budget.
   - **Next step:** a Tier-1 screen against v8(8), sized like the v8 screen.
2. **A growth or food-intake study** (row 5). This is the largest uncapped pool, but its
   ceiling is unmeasured.
   - Measure first, cheaply: hero intake vs a scripted greedy_food snake of the same length
     in the same worlds, and the boost burn per pellet gained.
   - Only then consider training-side work (intake reward or curriculum, distillation of a
     search policy).
3. **Do not prioritize:**
   - a deeper *static* trap search or a `no_spacious` fallback shield (row 4: mostly delay,
     about +1–3);
   - a larger one-step free-space cap;
   - more head-on work (head-on is solved: 1 death, unavoidable).

   The fallback shield remains cheap and harmless if someone wants it bundled with item 1,
   but it should not be screened alone.

## Caveats

- **Tier-0, a single namespace, and no confidence intervals.** The 69 deaths give
  rough shares only. The v5 comparisons cross engine, namespace and sample.
- **Headroom numbers are upper bounds.** They assume a rescued hero never dies later.
  "Realistic" is my judgement, not a measurement.
- **The static model.** The PNR walks freeze other snakes and food at each frame. That is
  exactly why world-driven closures show up as "taken action escaped".
  - `closure_check.py` attributes 16 of the 27 exact cases to the world changing between two
    frames. Food changes are included in "world"; given the growth rule they are unlikely to
    matter.
  - The other 11 were states exactly at `need` whose every continuation still failed.
- **The CPU cap.** 17 of 53 self deaths hit the 300 s cap, so 17 primary walks and 24
  count-only walks are unknown. These are never guessed.
- **Latency** is the v8 hook's wall time on SIMD row views. It is close to, but not the same
  as, the live serving cost (0.65 ms per decision in the serving lane). It was measured on a
  shared laptop CPU with 3 concurrent shards.
- **Kill credit** counts every opponent that hits the hero's body. Most of these are
  greedy_food snakes in the scripted mix, not hero aggression.
