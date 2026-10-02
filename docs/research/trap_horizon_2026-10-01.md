# Trap horizon: point of no return before Apex self-collision deaths (2026-10-01)

Development diagnostic (Tier-0/1). It is not a screen and not gate evidence, and it
produces no candidate. It measures how long before each self-collision death the
released configuration (champion + v2 free-space veto) last had a way out. The goal
is to choose the next intervention after v3 and v4 were retired.

## Setup

| Item | Value |
|---|---|
| Code | `research/trap_horizon_20261001/diagnose.py` (commit `31584a9`, clean tree) |
| Tests | `tests/test_trap_horizon.py`: 26 tests, about 3.5 s; episode runners are fatal |
| Output | `snake-dqn-artifacts/trap-horizon-20261001/run-v1` (not committed; `summary.json` sha256 `e458857a1eb7d12e…`) |
| Engine | SIMD, `run_simd_eval(vector61=True, hero_safety_veto=True, vector61_forward="rowwise")`, which is bit-exact with the live Watch hero at H5000 at record level |
| Hero | `champion_a5_freespace_20260621.pth` (`43d4e2c5…`) + v2 veto (`safety_veto.py` sha `1b62d15c…`, unmodified) |
| Worlds | H5000, `promotion-v2-watch-rect`, strict balanced rosters. 16 worlds per mix (frozen, scripted, mixed), shared across mixes. |
| Namespace | `trap-horizon-dev-v1/worlds`, using the `dev_screen.uint32_seed` recipe. It is disjoint from the first 1,000 seeds of every apex-safety, apex-veto strict, serving, v3 and v4 domain, from the smoke domains, from the task-aligned challenger namespaces, from the strict pilot's observed seeds and from seeds 0–999. The SIMD parity check reused `apex-safety-screen-v1`, which is covered. |
| Run | CPU slot 1 held, AC power checked, `OMP_NUM_THREADS=2`, torch threads 2/1. Analysis used 2 worker processes. |
| Wall time | **3,960 s total**: simulation 408 s (frozen 140, scripted 146, mixed 123), offline search 3,552 s |

**Capture.** A read-only wrapper around `Vector61SimdPolicy._apply_veto` calls the
original first and returns its result unchanged. At every hero decision it snapshots:

- the hero body (ordered), length, heading and boost counter;
- the cells of every other live snake and the food cells;
- v2's one-step counts, `need`, base action and final action.

The last 120 decisions per world are kept. Captured decisions equaled the veto's
decision counter in all 48 episodes.

**Escape search** (offline, one per frame and first action):

- **Hero moves.** The hero's own moves are simulated exactly, including boost (two
  cells, burn and trail pellet) and growth from that frame's food.
- **Model check.** The model reproduced **4,998 of 4,998** captured transitions, and
  all 42 fatal actions were predicted as `self`. A unit test also checks it against
  `BatchSim` directly, with random actions including boosts.
- **Other snakes** are static walls.
- **Primary criterion (as specified).** An action escapes if some continuation either
  reaches a state whose tail-aware reachable count (the v3/v4 release model, slack 1)
  is at least `need = min(length, cap)`, or survives 40 frames.
- **How the primary verdict is computed.** It is computed exactly as the OR of two
  one-sided searches, which are also reported as sensitivities:
  - `depth_only`: survive 40 frames.
  - `count_only`: reach a spacious state within 40 frames.
- **Search method.** Depth-first search with:
  - an exact failed-state memo;
  - a sound region prune, used only when length > 3·remaining + 2; tested to never
    change a verdict;
  - a budget of 50,000 nodes per (frame, action, criterion).

  When the budget runs out the result is **unknown**; it is never guessed. Results
  match a brute-force reference on small worlds.

**PNR.** The point of no return is the last frame at or before the fatal decision at
which at least one action escapes. `frames_before_death = fatal_index − pnr_index`,
so 0 would mean an escape still existed at the fatal decision itself. An unknown
frame after the candidate PNR makes the result a bounded `unknown`.

## Results

Episode outcomes (48): 42 self-collision deaths (frozen 13, scripted 14, mixed 15),
3 enemy-body deaths, 1 wall death, 2 survivals. All 42 self deaths were analyzed.
Fatal lengths ranged from 50 to 759 (median about 398); 38 of the 42 were at least 160.

### PNR distribution (primary criterion)

| | Count | Fraction |
|---|---:|---:|
| Exact | 37 | |
| Unknown (budget), bounded | 5 | |
| **≤ 8 frames** | **31** | **0.74** |
| 9–20 | 6 | 0.14 |
| 21–40 | 0 | 0 |
| > 40 | 0 | 0 |
| Unknown | 5 | 0.12 |

- Exact PNRs (n = 37): **median 4 frames, q25 1, q75 6**. Exact values:
  - 1 frame: 12 deaths
  - 2 frames: 1
  - 3 frames: 5
  - 4 frames: 6
  - 5 frames: 3
  - 6 frames: 3
  - 7, 9, 10 and 12 frames: 1 each
  - 15 frames: 1
  - 17 frames: 2
- The 5 unknowns are bounded:
  - [3, 4]: falls in the ≤ 8 bin;
  - [9, 16], [11, 15] and [16, 18]: all fall in the 9–20 bin;
  - [3, 15]: spans two bins.

  So no death has a PNR beyond 18 frames under this criterion.
- By mix (≤ 8 / 9–20 / unknown):

  | Mix | ≤ 8 | 9–20 | Unknown |
  |---|---:|---:|---:|
  | frozen | 9 | 3 | 1 |
  | scripted | 10 | 2 | 2 |
  | mixed | 12 | 1 | 2 |

### What the policy did at the PNR

- **Fatal choice in 31 of 42 deaths (0.74).** In these, the action taken at the PNR
  had no escape, but another action did. Every escaping action at every PNR was
  allowed by the action mask.
  - **24 of the 31 are boosts.** The same direction at normal speed had an escape.
    In 21 of these 24, v2 kept the action because the direction was spacious by its
    one-step count (160 of 160 in most cases).
  - Measured offline: the boost's two-cell landing state had a free-space count of
    **1 to 24 cells** (need 160, or 50 for the short hero). That count is the same
    with or without tail release. Moving in the same direction at normal speed
    landed in 160 cells in 21 of the 24.
  - This is v2's documented boost approximation, which scores a boost by its
    one-step direction, firing in practice. In 12 of the 24, the boost landed in a
    1-cell pocket and the hero died on the next frame (PNR = 1).
  - **7 of the 31 are normal-speed choices.** Another direction escaped, and v2 had
    no spacious direction (`no_spacious` fallback).
- **The taken action itself had an escape in 8 deaths** (6 of them exact), so these
  are not policy errors under this model. The trap closed afterwards, because other
  snakes or food changed (others are static in the model) or because of the
  survive-40 horizon shift: a 40-frame survival from t is only 39 frames from t+1.
  The escape at the PNR was survive-depth only in 17 of 42 deaths, and also a
  spacious-count escape in 25.
- **Unknown taken status:** 3 deaths.
- **v2 at the PNR:** `kept` in 23 deaths (every direction or the taken direction
  spacious), `no_spacious` in 19. At the fatal decision v2 was `no_spacious` in all
  42.

### Sensitivity criteria

| Criterion | Exact | Median (q25–q75) | ≤ 8 | 9–20 | 21–40 | > 40 | Unknown |
|---|---:|---|---:|---:|---:|---:|---:|
| count-or-depth (primary) | 37 | 4 (1–6) | 31 | 6 | 0 | 0 | 5 |
| depth-only (survive 40) | 41 | 4 (1–7) | 31 | 10 | 0 | 0 | 1 |
| count-only (reach spacious) | 30 | 3 (1–5) | 23 | 3 | 2 | 2 | 12 |

- **Count-only is the stricter "real way out" proxy for long heroes.** Surviving 40
  frames in a closed pocket does not save a snake of length 300 or more.
  - Its exact values are mostly identical to the primary. A few are much earlier:
    17, 22, 38, 59 and 60 frames.
  - Its 12 unknowns have lower bounds of 1 to 26 frames and upper bounds up to 70.
- So for a minority of deaths, roughly 5 to 10 of 42, the hero was already committed
  to a closed region 20 to 60 frames earlier and only survived inside it. The
  primary metric does not see this, because surviving 40 frames counts as an escape.

### Search cost (pure Python, 1 core; pre-death windows, so biased to hard states)

**Primary search on the taken action** (the check an always-on shield would run):
4,757 decisions outside the walked frames.

| Statistic | Time |
|---|---|
| Median | 0.09 ms |
| p90 | 6.8 ms |
| p99 | 12.7 ms |
| Max | 5.4 s (budget exhaustion) |

**All six actions at walked trap frames:**

| Search | Median | p90 | Max |
|---|---|---|---|
| depth-only | 0.07 ms | 60 ms | 15 s |
| count-only | 14 ms | 22 s | 34 s |

## Implications

1. **The trap is entered late, and mostly by a boost.**
   - Under the specified criterion, 74% of self deaths have their PNR at most 8
     frames before death, and none beyond 18 frames.
   - In 24 of 42 deaths (57%), the fatal step is a boost into a 1- to 24-cell pocket
     while the same direction at normal speed stays open.
   - The cheapest next intervention is therefore not a deep search. It is a
     **boost-aware veto**: score a boost action from its two-step landing cell, with
     the first cell blocked. This is v3's `boost_two_step` option, which was never
     screened. Its cost is one extra capped flood fill per boost decision.
   - On these 24 deaths the landing counts (1 to 24) are far below `need`, so such a
     check would have flagged every one of them.
   - Whether the hero then survives longer is **not shown**. After a switch to normal
     speed, the following frames may still trap it, and other snakes move. This
     needs a Tier-1 screen (v2 against v2 + two-step boost check) on a fresh
     namespace.
2. **A shallow always-on search shield is plausible for the remaining fatal
   choices.** These are 7 normal-speed choices with PNR at most 17 frames, and
   possibly the boost cases too.
   - A depth-40 count-or-depth check on the taken action costs a median of about
     0.1 ms in Python (p99 about 13 ms). Rare budget hits reach seconds, so a
     production shield would need a small node budget with a v2 fallback, as v4 had.
   - v4 (depth 8, budget 4,000) found "no escape when triggered". This is consistent
     with these numbers: v4 triggered on low one-step counts, and by then the PNR was
     usually past. v4 also kept the one-step boost approximation, so it never saw the
     two-cell landing.
3. **Policy and training remain the lever for the minority of committed deaths.** For
   about 5 to 10 of 42 deaths (count-only PNR 17 to 60 frames, plus count-only
   unknowns), the hero entered a closed region well before death. Short searches
   cannot fix these. The 8 "taken action had an escape" deaths are likewise outside
   what a static-world search can address.

## Caveats

- **Static model.** Other snakes are static walls and food is frozen at each frame;
  only the hero's own moves are exact. Real escapes can be closed by other snakes,
  which is one explanation for the 8 taken-escaped PNRs, and real traps can be opened
  by them.
- **The escape criterion is the one specified, and it is generous.** Area of at least
  `need` can be a dead-end corridor, and surviving 40 frames can be a slow death for
  a long snake. The count-only sensitivity shows the second effect is material for a
  minority of deaths.
- **Unknowns are budget-limited.** Primary 5 and count-only 12; they are not guessed.
  Their bounds are in the per-death JSON files.
- **Sample.** Development sample: 48 episodes, one namespace and 42 deaths, with no
  confidence intervals. The SIMD engine is record-level bit-exact with live at H5000
  but was not re-verified per decision here.
- **Cost figures** are pure Python on a shared laptop CPU, measured only on pre-death
  windows. They are not a serving benchmark.
