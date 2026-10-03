# Death census: what kills the Apex champion under the released v5 veto (2026-10-02)

Development diagnostic (Tier-0/1). It is not a screen and not gate evidence, and it
produces no candidate. It records every hero death of the released Watch configuration
(champion + `free-space-veto/v5-boost-aware`) and asks what the next intervention
should be. It is the follow-up to `docs/research/trap_horizon_2026-10-01.md`, the v2
study that motivated v5.

## Setup

| Item | Value |
|---|---|
| Code | `research/trap_horizon_20261001/diagnose_live_v5.py` (commit `e04da43`, clean tree; branch merged with `main` at `8fb257a`) |
| Tests | `tests/test_death_census_v5.py`: 15 tests, about 1.5 s. Episode runners are fatal in every test that calls `main`. One test calls `rollout` directly on a tiny synthetic world. |
| Output | `snake-dqn-artifacts/trap-horizon-v5-20261002/run-v1` (23 MB, not committed; `summary.json` sha256 `f15204cdd000b1b9…`, `intent.json` `bbb3a4851263f67c…`) |
| Engine | **Live** `tournament_eval.rollout`. v5 is not ported to SIMD. |
| Hero | `champion_a5_freespace_20260621.pth` (`43d4e2c5…`) + v5, installed through `dev_screen.hero_veto_installer(install_boost_aware_veto)`, which is the v5 screen's arm B. Unmodified sources are recorded in `intent.json`. |
| Worlds | H5000, `promotion-v2-watch-rect`, strict balanced rosters, 16 worlds per mix (frozen, scripted, mixed), shared across mixes |
| Namespace | `trap-horizon-v5-dev-v1/worlds` (`dev_screen.uint32_seed` recipe) |
| Run | CPU slot 1 held, AC power checked, `OMP_NUM_THREADS=2`, torch threads 2/1, `SNAKE_DQN_DEVICE=cpu`. Analysis used 2 worker processes. |
| Wall time | **6,304 s total**: simulation 1,436 s (frozen 604, scripted 324, mixed 507), offline analysis 4,868 s |

The namespace is disjoint from the following, each checked on its first 1,000 seeds per
purpose: diagnose.py's set (every apex-safety, apex-veto strict, serving, v3 and v4
domain, the smoke domains, the task-aligned challenger namespaces, the strict pilot's
observed seeds, and seeds 0–999); every `apex-veto-v5-*` domain (screen and its smoke,
strict dev/final/serving and smoke, web-serving watch/play/parity and smoke); and
`trap-horizon-dev-v1` with its smoke domain.

**Capture.** All capture is read-only.

- The installed v5 veto instance's `apply` is wrapped. The original runs first and its
  action is returned unchanged. At every hero decision the wrapper snapshots, in cell
  units:
  - the hero body, length, heading and boost counter;
  - the cells of the other live snakes and the food;
  - v2's one-step counts and `need`;
  - the masked Q row, the mask, the base and final actions, and v5's decision reason.

  The last 120 decisions per episode are kept.
- `GameLogic.check_collisions` is also wrapped. It keeps only the latest post-move,
  pre-collision world while the hero is alive, so at a death it holds the fatal frame.
- Captured decisions equaled the probe's `decisions` in all 48 episodes. The post-move
  frame equaled the last decision's frame in all 43 deaths.

**Analysis.**

1. **Fatal-frame counterfactual, all deaths.** This step is exact for one frame.
   Opponents choose their moves from pre-move snapshots, so their moves do not depend
   on the hero's action. Each of the 6 hero actions is replayed against the captured
   post-move world with the live collision order:
   - wall, then self;
   - head-on, including swapped head paths and v2's 1.15× size rule;
   - enemy body.

   An alternative that survives the frame is then searched for an escape: the
   trap-horizon count-or-depth criterion, depth 40, with other snakes frozen at their
   post-move cells.
   - A unit test checks the rule against the live `Snake.move` and `check_collisions`
     on 1,500 random worlds. All five outcomes occur and all match.
   - In the run, the rule reproduced the actual cause in **43 of 43** deaths.
2. **PNR walk, self deaths only.** `diagnose.analyze_death`, unchanged: static other
   snakes, count-or-depth primary criterion, depth-only and count-only sensitivities,
   a 50,000-node budget, and unknown results never guessed. The hero-move model
   reproduced **5,062 of 5,062** captured live transitions, and all 36 fatal actions
   were predicted as `self`.

## Results

### Death causes (48 episodes)

| Mix | self | head_on | enemy_body | wall | survived | Mean mass |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 15 | 0 | 0 | 0 | 1 | 205.5 |
| scripted | 9 | **5** | 1 | 1 | 0 | 144.4 |
| mixed | 12 | 0 | 0 | 0 | 4 | 265.4 |
| **total** | **36** | **5** | **1** | **1** | **5** | |

**A larger sample (descriptive only).** These are the same configuration's death causes
in the existing v5 strict-gate final records (`apex-veto-v5-strict-20261001/run-v1`,
candidate arm, N = 269 per mix). They were read only; no new episodes were played.

| | self | head_on | enemy_body | wall | survived |
|---|---:|---:|---:|---:|---:|
| frozen | 209 | 16 | 13 | 1 | 30 |
| scripted | 210 | **33** | 8 | 1 | 17 |
| mixed | 207 | 14 | 9 | 4 | 35 |
| pooled share | 77.6% | 7.8% | 3.7% | 0.7% | 10.2% |

Head-on counts are the same under the v2 incumbent arm (13 / 33 / 13). This is
expected, because v5 changes only boosts.

- **Early head-ons:** 30 of the 33 scripted head-ons, and 34 of 63 overall, happen with
  less than 10% survival, which means within roughly the first 500 frames.
- **Their mass:** those 30 scripted episodes average **0.6** mass, against **183.4**
  for the other 239 scripted episodes.

### Head-on, enemy-body and wall deaths (census, fatal frame)

| Cause | n | Cause reproduced | Legal alternative avoids death | …and has a static escape | Taken cell occupied before the move |
|---|---:|---:|---:|---:|---:|
| head_on | 5 | 5 | **5** | **5** | 0 (all moving heads) |
| enemy_body | 1 | 1 | 0 | 0 | 1 |
| wall | 1 | 1 | 0 | 0 | — |
| self | 36 | 36 | 0 | 0 | — |

- **All 5 head-ons look alike.** They are scripted mix, frames 65–283, hero length
  9–49, and the opponent is `scripted:greedy_food`.
  - The opponent was the same size or bigger, so the 1.15× rule gave the hero no win
    (opponent length 12–58).
  - Its head was diagonally adjacent to the hero's head, or two cells away for a boost.
  - v5 `kept` the base action every time, because the cell was empty before the move,
    so the mask and the free-space count saw no danger.
  - Each death had 2 to 5 masked-legal alternatives that survived the frame and had a
    static escape.
- **enemy_body (length 600) and wall (length 708).** In both, every action except one
  died; the one survivor was an illegal boost in the enemy-body case. These are
  enclosure deaths, like the self deaths below.
- **Self deaths at the fatal frame.** No legal alternative survived in any of them. v5
  was `no_spacious` at the fatal decision in all 36.

### Self deaths: point of no return (primary criterion)

| | v5 census (live, this run) | v2 study (SIMD, `trap-horizon-dev-v1`) |
|---|---:|---:|
| Self deaths / episodes | 36 / 48 | 42 / 48 |
| Fatal lengths | 262–807 | 50–759 |
| PNR exact | 30 (median **7**, q25 2, q75 13) | 37 (median 4, q25 1, q75 6) |
| PNR ≤ 8 / 9–20 / 21–40 / > 40 / unknown | 17 / 9 / 4 / 0 / 6 | 31 / 6 / 0 / 0 / 5 |
| Fatal choice at PNR, **boost** | **10** | 24 |
| Fatal choice at PNR, normal speed | **14** | 7 |
| Taken action itself escaped at PNR | 8 | 8 |
| Unresolved (taken status unknown) | 4 | 3 |
| v2/v5 one-step state at PNR | `no_spacious` 34, `kept` 2 | `no_spacious` 19, `kept` 23 |
| Escape at PNR only by surviving 40 frames (no spacious-count escape) | **33** | 17 |
| Count-only PNR > 20 frames before death ("enclosed early") | 13 yes, 9 no, 14 undetermined | 6 yes, 26 no, 10 undetermined |

By mix (≤ 8 / 9–20 / 21–40 / unknown): frozen 6/3/1/5, scripted 3/4/2/0, mixed
8/2/1/1. Every escaping action at every PNR was masked-legal.

**Count-only sensitivity**, which asks whether the hero can reach a spacious state
rather than merely survive:
- 19 exact, median **21** frames (q25 9, q75 34);
- bins ≤ 8 / 9–20 / 21–40 / > 40: 4 / 5 / 6 / 4;
- 17 unknown, with lower bounds of 1 to 53 frames and upper bounds up to 61.

**Depth-only sensitivity**: 31 exact, median 7.

**What changed from v2 (descriptive only).** The engine, namespace and sample all
differ.

- **The v2 failure mode is gone.** That mode was a boost from a state v2 called
  spacious into a 1- to 24-cell pocket (21 of 24 boost deaths in the v2 study). Here,
  0 of 10 PNR boosts were in a spacious state.
- **The 10 remaining boost fatal choices all happened under `no_spacious`.** In that
  regime v5's landing check is not applied: v5 tests a boost's landing only when its
  direction is v2-spacious, and otherwise leaves the base action unchanged.
  - Their landing counts were 1 to 44 cells (need 160); four of them were 1 to 4 cells.
  - Moving in the same direction at normal speed escaped in 7 of the 10 (no escape in
    2, unknown in 1).
- **The remaining self deaths begin earlier.** At the PNR the hero was already in a
  region below `need` in 34 of 36 deaths. The only escape left was usually surviving
  40 frames inside it (33 of 36), and that does not save a snake of length 262 to 807.
  The decisive event is entering the sub-`need` region, which happens before the PNR.
  The count-only horizon (median 21 frames, often unknown up to about 60) is the better
  measure of when that occurs.

### v5 counters (census)

| Mix | Decisions | Kept | Vetoes | `no_spacious` | Boost bases | Landing checks | Landing vetoes (all → same direction, normal speed) | Differs from v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 53,277 | 52,674 | 62 | 541 | 4,035 | 3,961 | 15 | 15 |
| scripted | 35,892 | 35,339 | 57 | 496 | 2,800 | 2,695 | 15 | 15 |
| mixed | 60,727 | 60,211 | 93 | 423 | 4,289 | 4,241 | 18 | 18 |

`no_spacious` covers 1,460 of 149,896 decisions (0.97%). The mean v5 cost per decision
was about 0.25 ms.

### Shield cost

A count-or-depth search on the taken action, outside the walked frames, covered 3,932
pre-death decisions (pure Python, 1 core):

| Statistic | Time |
|---|---|
| Median | 0.10 ms |
| p90 | 10.4 ms |
| p99 | 15.6 ms |
| Max | 7.5 s (budget exhaustion) |

## Recommendation (ranked)

Shares come from the strict-gate final records (N = 807 v5 episodes). Mechanism
fractions come from this 48-episode census. All of it is development evidence, and
each item needs its own Tier-1 screen.

1. **Opponent-head avoidance (normal and boost reach).**
   - **The rule:** when an alternative exists, veto a move into any cell that an
     opponent's head can reach next frame, at normal speed or by boosting, unless the
     hero is at least 1.15× that opponent's logical length.
   - **Addressable share:** 7.8% of episodes end in a head-on (12.3% in scripted). In
     the census all 5 head-ons had legal alternatives that survived the frame with a
     static escape, and the opponent's head was always within reach.
   - **Mass at stake:** the early head-ons are almost all scripted-mix, 30 of 269. If
     those episodes instead looked like the other scripted episodes, scripted mass would
     rise from about 163 to about 183 (+20, about +12%), and pooled mass by about +8.
     This is an upper-bound style estimate: the hero might still die early another way.
   - **Why first:** it is cheap (a few cell lookups per decision), and its per-death
     mass is by far the largest because these deaths happen near frame 0.
   - **Uncertainty:**
     - The one-frame counterfactual is exact, but whether the hero then survives longer
       is not shown.
     - The rule will sometimes give up food races against greedy opponents.
     - Head-on in the frozen and mixed mixes (16 and 14) mostly happens later and was
       not inspected here.
2. **A no-spacious fallback shield for self deaths.**
   - **The rule:** when v2/v5 finds no spacious direction (1% of decisions), stop
     leaving the base action unchanged and choose among the legal actions:
     - apply the landing check to boosts, preferring the same direction at normal
       speed when its landing is larger;
     - then rank actions with a budgeted count-or-depth search, falling back to v5's
       choice when the budget runs out.
   - **Addressable share:** under the static model it reaches the PNR fatal choice in
     24 of 36 self deaths (67%, about 52% of episodes once scaled by the 77.6% self
     share). The boost part alone is 10 of 36.
   - **But the realistic gain is mostly delay.** In 33 of 36 the only escape at the PNR
     was surviving 40 frames inside a region smaller than `need`, and at least 13 of 36
     (up to 27 including the undetermined) were enclosed more than 20 frames earlier
     under the count-only criterion. A delay of about 40 frames at length about 500 is
     worth only about 4 mass per death. Real rescues are possible only for the minority
     that the count-only criterion places inside a few frames: 9 deaths have a
     count-only PNR at or below 20 frames.
   - **Cost:** it runs only on the 1% `no_spacious` decisions. Median 0.1 ms, p99 about
     16 ms, with rare multi-second budget hits, so it needs a small node budget.
   - **Expected effect:** smaller and less certain than item 1. Worth screening because
     it is cheap and composes with v5.
3. **Policy or training for early enclosure.** This is the largest pool and the least
   tractable.
   - Self deaths are 78% of episodes. The decisive mistake in most of them is entering
     a region below `need`, typically about 20 to 60 frames before death (count-only
     criterion), at lengths of 260 to 800.
   - A shallow shield cannot see that far. v4's shallow search already found "no escape
     when triggered", and a deep count-only search costs seconds per decision.
   - The fix is likely training-side: a long-snake curriculum, a space-aware auxiliary
     target, or a reward for staying connected to the open area. It could also be a
     much deeper planner, which is not justified on CPU cost here.
   - Expected gain is potentially large (most remaining mass loss) but unquantified.
     It is a multi-day effort with an uncertain outcome.

## Caveats

- **Small sample.** 48 episodes in one namespace, 43 deaths and no confidence
  intervals. The strict-gate shares are larger but only descriptive here, and those
  worlds were already used for the v5 decision.
- **The v2 comparison is cross-engine** (SIMD vs live) and cross-namespace. The v2
  study saw no head-on in 48 episodes, while the strict v2 arm shows 33 of 269 in
  scripted. That absence is plausible as chance (about 12%) and was not investigated.
- **The static model.** In the PNR walk other snakes are frozen and food is fixed at
  each frame. The fatal-frame counterfactual is exact for one frame only. Its escape
  check freezes others at their post-move cells, and it adds every pellet seen before
  or after the move to the hero's food (growth only affects head-on size resolution).
  This reproduced the actual cause in 43 of 43 deaths.
- **The escape criterion is generous.** Surviving 40 frames, or a region of at least
  `need` cells, does not mean surviving long term. The count-only sensitivity and the
  survive-only escape counts show this matters for most remaining self deaths.
- **Unknowns are never guessed.** Primary: 6 (5 placed by bounds). Count-only: 17.
  Their bounds are in the per-death JSON files.
- **Cost figures** are pure Python on a shared laptop CPU, from pre-death windows only.
