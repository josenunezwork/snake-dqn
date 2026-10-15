# Librarian answers for the panel (2026-10-15)

Read-only. Sources on this machine: the report
(`scratchpad/mac_session/REPORT_mac_session_2026-10-12_to_14.md`), the digests
(`scratchpad/mac_session/digests_rendered.md`, cited as `digest L<line> (p:<page cursor>)`),
the raw census result (`scratchpad/p10.txt` line 10), and the repository at
`/home/user/snake-dqn` (Oct 8 state). `train_m3c.py` (branches m3c..m6) is NOT here; its
ancestor `research/redesign_m3_20261008/train_m3.py` is, and the digests quote the flags that
were added on top of it. Anything marked **derived** is my arithmetic on quoted numbers.

Status legend: answered / partial / needs_mac_artifacts / needs_new_measurement.

---

## A. The Mac trainer (`train_m3c.py` on branch m6) — questions 1, 7, 16, 29

These four questions overlap; one combined answer. **Status: partial** (code behaviour read
from `train_m3.py`, the M3 ancestor; the M4/M6 flags from the digests; nothing about
`train_m3c.py`'s own source can be checked here).

### TD loss and beta
- `train_m3.py:430`: `td = torch.nn.functional.smooth_l1_loss(qa, y)` — PyTorch Huber with the
  **default `beta = 1.0`**, mean over the 384 agent rows. Target `y = R + disc * q_next`,
  Double-DQN (online argmax over the resolved mask, target net value), `disc = gamma**k` (0 on
  terminal). Grad-norm clip 10, Adam 1e-4 (`train_m3.py:347,436-438`).
- The anchor's advantage-Huber uses `beta=0.1` (`train_m3.py:243`), but the M4/M6 arms run
  `--anchor-mode v8margin` = "DQfD margin only, no adv-Huber/KL" (digest L32 p:a2f189cd), so
  that beta is irrelevant to B8/M6.
- The M3 preregistration (`research/redesign_m3_20261008/PREREGISTRATION_M3.md:19-23`) and the
  research agent's read of `train_m3c.py` (digest L25 p:b1ce20c9: "Double DQN dueling, 5-step,
  Adam 1e-4, Huber, target sync 1000, replay ratio 4") agree. `--learner-fast` (T1) was
  verified "plain Adam bit-identical on CPU and MPS over 200 updates" (digest L438
  p:81f73729), so the loss form did not change on perf-learner. Whether `train_m3c.py` exposes
  a Huber-beta flag: not visible here (no such flag in any quoted argv).

### n-step
- Hard-coded `n_step = 5` (`train_m3.py:454`); the n-step window is built per world from a
  deque (`train_m3.py:519-525`), death flushes the pending window as terminal (`flush(e, -1)`),
  segment end bootstraps from the last observation (truncation, `train_m3.py:592-597`).
- perf-actor replaced the Python loops with a "vectorized n-step builder + eps sampling
  (identical RNG; old loops behind --actor-impl reference)" (digest L397 p:01cde733). No
  `--n-step` flag appears in any quoted argv; the literature track lists "longer n-step" as an
  untried lever (digest L1154). Treat n-step as **code-level, not flag-level** unless the m6
  source shows otherwise.

### Replay capacity and sampling
- `--replay 500_000` (flag; `train_m3.py:314`); `ObsRing(args.replay + 4*worlds*8)` holds
  uint8 observations, `Replay(args.replay)` holds n-step tuples (s, a, R, disc, next) as a
  ring (FIFO overwrite, `train_m3.py:134-153`).
- Sampling: **uniform** `rng.integers(0, size, 2n)` filtered to rows whose obs ids are still in
  the ring, truncated to n (`train_m3.py:155-159`). No PER, no stratification.
- Batch = 384 agent + 128 demonstration rows = 512 (`train_m3.py:405,420`); one update per 128
  new agent transitions (replay ratio 4) once `replay.size >= --warmup 20_000`
  (`train_m3.py:539-542`). Target sync every 1000 updates (`train_m3.py:443`) = 128k
  transitions, or 256k at `--batch-scale 2` (M6 ran 31,239 updates of 768 rows over 8M, digest
  L1220 p:NEWEST, i.e. one update per 256 transitions). **Derived:** each transition is drawn
  ~3 times over its lifetime (384/128), as the literature track said.
- Replay RAM: plan `ram_per_run_gb 19`, `ring 9.3 GB/run` (digest L31 p:a2f189cd);
  `--obs-pack` footprint 1.167 -> 0.699 GB at replay 20k + demo 20k rows, "same math"
  (digest L813 p:65231b02); report says 19 GB -> 6-7 GB per run. **Derived from code:** one
  unpacked ObsRing row = local 7x31x31 (6,727 B) + global 4x37x37 (5,476 B) + scalars 60x4 B
  (bv8) + mask 6 B = **12,449 B**; the Replay tuple is 32 B/transition.

### Segments, resets, staggering
- `build_segment` (`train_m3.py:259-301`) constructs a **fresh GridBatchSim of all E worlds**
  every `--segment-frames 5000` frames with new world seeds (offset `segment*E`); so yes, all
  worlds reset together at the segment boundary. Within a segment the hero **respawns**
  (`allow_respawn=True`, timer = frame_rate frames, `grid_sim.py:806,816-840`); its death is
  terminal for TD but the world continues. The sim's reward gamma is the learner's gamma at
  segment start (`train_m3.py:11,460-462`).
- B8/M6 production: `--worlds 128 --actors 2` on the Mac (digest L414 p:a49d089c; L1023
  p:19aa3a13); the pod argv used `--actors 3` (L921 p:0a2b5d99). Whether the actors' segments
  are staggered is **not stated anywhere**. The literature track's "qmax sawtooth ~1.28M"
  (digest L1145/L1154) equals 2 x 128 x 5000, which is only consistent with the two actors'
  segments being **in phase**; that is an inference, not a reading of the code.

### LabelBuffer and label_frac
- Digest L32 (p:a2f189cd): M4-A added "LabelBuffer (on-policy v8 labels); actor shield +
  labels (make_shield(carrier, runtime)); learner mixes label rows into anchor batch
  (label_frac) and tracks v8_override/label_rows". So the 128-row anchor batch is drawn from
  the on-policy label buffer with probability/fraction `label_frac`, the rest from the M2b
  demo shards. B8 and M6 ran `--onpolicy-labels --label-frac 1.0` (digest L249, L414, L921),
  i.e. **all** anchor rows are on-policy v8 labels on the student's own states. The arm-S
  smoke without `--label-frac` showed `label_rows_mean 57.76 .. 63.02` of 128 (digest L38),
  consistent with a default near 0.5. Sampling rule inside the LabelBuffer (uniform/FIFO):
  not visible here.
- v8 overrides the student on 0.1-0.2 % of decisions (`v8_override_rate 0.0021 / 0.0014 /
  0.0011`, digest L38, L41), so ~99.8 % of labels equal the student's own argmax.

### Margin constant
- Read from `<demo-data>/margin.json` (`train_m3.py:348`), fixed from the first M2b round
  as `MARGIN_FRACTION (0.1) x median per-row teacher Q range`
  (`research/redesign_m2_20261008/train_student.py:41,236-239`). The committed copy
  `research/redesign_m2_20261008/results_m2b/margin.json` = **0.06319992542266846**
  (`median_q_range 0.6319992542266846`). The Mac runs point at
  `redesign-m2b-20261008/data/margin.json`, which by construction holds the same first-round
  value (not verifiable here). Note the scale: the DQfD margin is 0.063 Q-units while the
  mass-reward Q is 50-150 (qmax 120-153 in the M6 smokes, digest L912).

### Anchor lambda schedule / the lambda column of M6's log.jsonl
- `Schedules.anchor(t)` (`train_m3.py:90-96`): **1.0 for t <= anchor_hold (default
  8,000,000)**, exponential decay to `anchor_floor` 0.1 at `anchor_floor_at` 15,000,000.
  `t` is the fine-tune's own transition counter (checkpoints save weights only,
  PREREGISTRATION_M3.md:79; `state["transitions"]` starts at 0).
- B8 recipe passes `--anchor-hold 8000000 --anchor-floor-at 15000000` explicitly (digest
  L414). The M6 argv (digest L921) passes `--anchor-mode v8margin --anchor-floor 0.1` and no
  hold/floor-at, so with the defaults above **lambda = 1.0 for the entire 8M fine-tune**. The
  actual `lambda` column of `redesign-m6-mac-20261014/*/log.jsonl` is on the Mac
  (needs_mac_artifacts to confirm). The smoke runs logged `lambda 1.0` at 100k and `0.1` at
  200k only because that smoke used shortened schedule flags (digest L38).
- log.jsonl rows (every 100k transitions) carry: transitions, updates, wall, rate, env_s,
  learn_s, eps, gamma, lambda, td, anchor, qmax, boost_rate, greedy_boost_rate, deaths_per_1k,
  demo_shards_loaded, demo_distinct, holdout_agreement every 500k (`train_m3.py:549-574`).

---

## B. The exact mass reward (questions 1, 7, 14)

**Status: partial** (base reward from code; the mass term from the digests' description).

- Base v2 reward (`src/simd_env/grid_sim.py:845-868`, `src/core/reward_events.py`):
  `r = gamma*Phi(L') - Phi(L) + death_value*[died] + kill_scale*sum(victim lengths)`, with
  `Phi(L) = L / 10` (`PHI_LENGTH_DIVISOR = 10.0`), **`Phi(death) = 0`** (so dying at length L
  loses `L/10` through the potential term), `death_value` default -3.0, `kill_scale` default
  0.3; both configurable via cfg (digest L32).
- `--reward mass --mass-kappa 1.0` adds `kappa*(1-gamma)*Phi(L)` per frame to that reward
  ("B/L add kappa(1-gamma)Phi(L), kappa=1", digest L25 p:b1ce20c9; the literature track's
  "death at L 1000 costs Phi(L)=100 Q units vs per-step 0.2", digest L1154, is exactly this
  form: 0.002 x 100 = 0.2/frame at L=1000). **The PBRS term is kept**; whether the mass term
  reads L or L' is not stated.
- `--kill-scale 0` -> kill term 0 (build_segment verified "kill_scale 0.0 / death -3.0",
  digest L35). Death value: -3.0 in C/B8/M6 CTRL; `--death-value -15` in M5 D20 (digest
  L436); DEATHW = `--death-mass-mult 2`, described as "death x 2 Phi(L)" (report table).
- Boost burn: no separate term; a burned segment lowers L by 1, which the potential term
  charges as ~-0.1 (and mechanics v2 drops a trail pellet). Corpse vs ambient intake: both
  enter only through L, so **one cell of body is worth the same (0.1 potential + 0.0002/frame
  mass term, which also sums to 0.1 over the 1/(1-gamma)=500-frame horizon) regardless of
  source** (derived). The gate world is `promotion-v2-watch-rect` (mechanics v2).
- Q scale check (derived): steady state at L=1000 -> Q ~ Phi(L) ~ 100; observed qmax 120-153
  (digest L912) and the "5 x demo_qmax = 57.3" tripwire fired 197 times in B8_s1's own run.

---

## C. The 30 bv8 features (questions 7, 23)

**Status: partial** (names from the numba port's feature planes; caps from the Python v7/v5/v6
reference in the repo; the bv8 featurizer itself is on branch m4b).

- `v8_nb.v8_feature_planes(res) -> (N, 6, len(FEATURE_NAMES)) float32` with
  **FEATURE_NAMES = (area_g, head_risky, landing_ok, v5_eligible, spacious)** (digest L207
  p:8a498ae4; L52 p:ca478168 "(N,6,5)"). 6 actions x 5 = **30 scalars**, appended to
  ego2s-b's 30 (obs spec (7, 60); digest L210, L265 "30 base + 30 v8 scalars"). Features are
  "Q-independent in [0,1]" (L210).
- Meaning and caps (Python reference, `src/evaluation/safety_veto_v7.py`, `_v5`, `_v6`,
  `safety_veto.py`):
  - `area_g` = v7's `area_score = min(area, cap)/cap`, `area` = tail-aware post-move landing
    flood (`landing_area`: v5 `simulate_action` body incl. boost burn, other snakes static,
    own body released by steps with slack 1), **cap = min(max(32, 2L), 4096, open cells)**
    (`safety_veto_v7.py:62-66,187-192`). So the tail-release landing area IS an input, but
    saturates at 2L (and at 4096 for L > 2048).
  - `head_risky` = v6 `head_risky_actions` (opponent head reach within 2 cells, size-resolved
    head-on rule).
  - `landing_ok` = v5 landing check for boosts (`landing_count >= need`, or n/a); the
    **boost landing COUNT itself is not an input**, only the boolean.
  - `v5_eligible` = masked-legal and v2-spacious (and landing_ok for boosts).
  - `spacious` = v2's `spacious_directions` (free-space threshold `free_space_threshold(L,
    logical L)`, the 160-cell-capped BFS of the vector61 teacher).
- ego2s-b's own 30 scalars include `region_frac_{left,straight,right} = min(R/L, 4)/4` and
  `region_log_* = log1p(R)/log1p(H*W)` with R the UNCAPPED 4-connected free component at the
  next cell, tail-aware by TTL > 1 (`src/simd_env/ego_raster_b.py:30-41`), plus a local
  `reach_time` K=24 time-aware flood plane (digest L25).
- **No boundary-TTL / time-to-wall-release signal and no multi-step tail-reachability**
  beyond v3's step-release model exists among the inputs. The M7 feature study's family A
  (time-aware flood) added +0.019 AUC at H100 (CI 0.005-0.033) and family C (global regions)
  +0.036 at H300 (CI 0.014-0.058) (digest L1152).

---

## D. Census: death frames, alive-frame survival, hazard by L x frame (questions 2, 3, 9, 13, 15, 17)

**Status: needs_mac_artifacts** for everything per-episode or per-decision; what exists in the
digests is below.

Artifacts on the Mac: `snake-dqn-artifacts/b8-death-census-20261014/results.json`,
`runs/{b8_s0..b8_s4,incumbent,champion}.json` (per-death rows) + `_series.npz` (per-decision
series), `identity.json`, README (p10.txt line 10, `artifacts` list). The census kept "a deep
copy of the prepared world" at every step and re-stepped the fatal frame for all 6 actions,
so death frame, length and region size at trap onset are recoverable from those files
without re-running.

What the digests give (all 32 scripted worlds of round 52, H5000, no veto for B8):
- Episode-level survival (k/32, deaths): b8_s0 0.656 (11), b8_s1 0.375 (20), b8_s2 0.625
  (12), b8_s3 0.531 (15), b8_s4 0.469 (17); pooled 0.531 (75 of 160). incumbent frp3-s12+v8
  0.656 (11), champion_a5+v8 0.688 (10). Mass 570.7 / 454.6 / 368.7. Median length at death:
  1106 / 793 / 1218 / 972 / 934; incumbent 643; champion 520 (p10.txt L10; digest L902).
- **Frame of death is not in the digests** (only the length at death and the trap length:
  median trap 31 decisions, 39 of 75 traps >= 30 decisions; champion traps ~4 decisions).
  So alive-frame survival on the census bank cannot be computed here.
- Hazard: "above 800 B8 dies at about 0.17 to 0.18 per 1000 decisions"; below 800 "26 deaths
  observed vs about 29 expected at the champion's per-length rates"; B8 spends 42 % of its
  659,595 decisions at L >= 800 (134 of 160 episodes reach it) with 49 of 75 deaths; champion
  5.6 % of decisions at >= 800, 0 deaths (reaches 800 in 20 of 32 episodes, median max mass
  824); incumbent 25 % at >= 800 with 2 deaths ("about 3x lower than B8's, but 2 deaths
  against 6 expected") (digest L903, L905; p10.txt). **Derived:** B8 at L >= 800: 49 /
  (0.42 x 659,595) = 0.177 per 1k decisions; below 800: 26 / (0.58 x 659,595) = 0.068 per
  1k. Champion decision counts are not given, so its per-1k hazard cannot be derived.
- No L-bucket x frame-bucket crosstab exists anywhere; the `_series.npz` files on the Mac
  are the input for it.
- Other census facts the questions touch: 85.3 % enclosures; boost involved in 8 % of deaths
  (last 5 decisions), 17 % within 20 decisions, 0 % on the fatal frame; opponent head within
  2 cells at the fatal decision 17 %, strict head involvement 1.3 %; 7 % hit an opponent body
  within 8 cells of its head; B8 boosts 0.10-0.24 % of frames vs champion 7.7 %; B8 died
  where champion survived in 51 of 160 pairs (46 self, 5 enemy_body), survived where it died
  in 26; "tight" decisions (best 1-step landing area < L) ~7 % at L >= 1000 for B8 vs
  0.02-0.16 % incumbent (digest L906).
- Sealer identity per death (own body / opponent / wall) at the FINAL cell is in the
  per-death rows ("hero's collision partners (slot, kind, hit offset)"): 67 self, 7 enemy
  body, 1 head-on; but who sealed the POCKET (vs the final cell) was not computed.

---

## E. Phase R look-2 per-world data (questions 2, 8, 15, 18)

**Status: needs_mac_artifacts.** Only clause-level numbers are in the digests.

- The clause: `c6_guard_scripted_survival` = mean over seed 0's bank, scripted mix, of
  `surv5(hero) - surv5(champion_a5+v8)`, threshold >= -0.05, protective
  (`research/sequential_phase_r/example_rules.py:85-92,122`). `surv5` is read from each
  record's `prefix_h5000.survival_fraction` (`hooks.py:140-147`), and `survival_fraction =
  alive_frames / scored_horizon` (`src/evaluation/metrics.py:133`): **alive-frame, not
  episode-level**. Guard cell = `{"hero": hero, "baseline": "champion", "seeds": [0]}`.
- Look 2 (n_per_seed_mix 32): c6_guard_scripted_survival estimate **-0.156** (adjusted
  -0.156, threshold -0.05, FAIL); c6_guard_point +136.819 pass; c2_scripted_survival (vs
  incumbent, 5 seeds pooled) -0.003 pass (thr -0.03); c3_frozen +0.011; c3_mixed +0.038;
  c5_mi10 +74.326; efficacy 128.72, SE 16.41, LB 101.48, seed effects [76.394, 162.197,
  142.832, 155.180, 105.027], tau2 541.64 (digest L412 p:2f7837eb, L516). Look 1 (n 22):
  guard -0.0730 adj -0.14198; c2_scripted -0.015 adj -0.052; c3_frozen -0.009 adj -0.034
  (digest L409). Look 0 (n 11): scripted survival vs incumbent +0.027 (SE 0.054) (L407).
  **The SE of the guard clause at look 2 is not quoted**; for scale, the M3 look-1 JSON in
  the repo has guard SE 0.0726 at n=22 (`research/redesign_m3_20261008/results/phase_r/look-1.json`).
- Per-world `surv5`/`mi5` live in the shard records (`<root>/shards/look-2/records/*.json`,
  one per (hero, seed, mix, world), each with `record` and `prefix_h5000` blocks); the
  look-2.json itself lists record paths and sha256s plus clause estimates (structure of the
  M3 `look-1.json`: keys action, decision, records[726]{key, path, sha256}, ...). The M4 root
  is under `snake-dqn-artifacts/` (name not quoted; M3's was
  `redesign-m3-phase-r-20261008/shards/look-1/records/`).
- Incumbent surv5 per bank and per-seed candidate survival: same records (incumbent is the
  decision baseline on every seed's bank, champion on seed 0's only). Not in the digests.
- The report's "needs about 0.64" subtracts 0.05 from the census's episode-level 0.688 on the
  round-52 dev bank; the clause is alive-frame on a different bank (`redesign-m4-phase-r-nv/v1`
  namespace, digest page_1792016251885.txt:249). The two are not the same quantity.
- Context: on the census bank B8 trails the incumbent by 0.125 episode-level (0.531 vs
  0.656) while Phase R's c2 found -0.003 alive-frame vs the incumbent on its own banks
  (verifier note, p10.txt).

---

## F. New measurements nobody has made (questions 4, 5, 6, 10, 11, 12, 19-22, 24, 27, 28)

All **needs_new_measurement**. What exists that each would build on:

- **Exact-replay counterfactuals / commitment horizon / sealer identity / Q-gap (4, 10, 11, 12):**
  the census only did one-step counterfactuals: at the fatal frame all 6 actions (96 % no
  action survives; 98.7 % no advisory-safe action); at trap onset (64 traps) the previous
  decision re-stepped per action: 41 unavoidable one step later, 23 avoidable, v8's action
  never among them, 1 of 23 needed a boost (p10.txt key findings 1, 4). Nothing beyond one
  step; no Q-gap at commitment; no pocket-sealer attribution; no tail-in-pocket flag. The
  census code (`research/b8_death_census_20261014/census.py`, branch census-b8) already
  deep-copies the prepared GridBatchSim each step and re-steps it, which is the mechanism a
  backward exact replay needs. Shadow-v8 per-action diagnostics were captured at every
  decision.
- **GridBatchSim deepcopy timing (4, 29):** `observation_probe.clone_sim` =
  `copy.deepcopy(sim)` + full-digest check (`src/evaluation/observation_probe.py:187-193`),
  written for BatchSim (GridBatchSim subclasses it; no GridBatchSim use in the repo). Oct 3
  bench (`docs/research/serving_time_search_feasibility_2026-10-03.md:149-154`): BatchSim
  `deepcopy(sim)` E=1 0.49 ms; E=1 clone + 8/32 greedy steps 5.7/24 ms; E=16 step 2.29 ms,
  E=64 step 8.55 ms; vector61 featurize 0.04 ms/row batched. GridBatchSim deepcopy was not
  timed; the census ran 32 worlds x 5000 frames with a deep copy every step in ~200 s per
  arm (upper bound ~1.25 ms per 32-world step incl. the copy, derived).
- **Linear probe on the 256-d fuse (5, 19):** fuse = `Linear(256+128+n_scalars, 256) ->
  LayerNorm -> ReLU` (`src/model/ego2s_network.py:57-59`), value/adv heads read it. The M7
  rows exist on the Mac: L >= 800, stride 3, 166,187 / 163,552 / 153,167 rows for H50/100/
  300 with 1,425 / 2,805 / 8,113 positives, 85 death episodes, 9 arms (5 B8 + 4 CTRL), 32
  worlds, 5 folds by world, 2,000 bootstraps; GBM AUC existing 0.91 / 0.811 / 0.695, LR 0.888
  / 0.825 / 0.726 (digest L1152). The capture is on branch m7-feature-auc (commits b0f26f2,
  bd0e750) in `.claude/worktrees/wf_8af9a1a9-edb-1`. No fuse activations were captured.
- **Training death records (6, 26):** `train_m3.py` logs only `deaths_per_1k` per 100k
  (`train_m3.py:567`); M6's `--tight-telemetry` adds `telemetry.jsonl` with tight_frac,
  deaths/1k, greedy_boost_rate, max qmax and `death_len_mean` per interval (digest L912). No
  per-death row (length, segment frame, mix, eps flag) is logged; that needs a code change.
  Existing readings: CTRL_s10 totals "tight 93355/3358146 deaths 980" (digest L1035); 1M
  smokes from B8_s1: CTRL deaths/1k 0.133 (0.14 final interval), TIGHT 0.157 (0.21), DEATHW
  0.143 (0.20), DEATHW `death_len_mean 628` (L912); literature track "death_len_mean ~200-235
  near the end of CTRL_s10 (unverified)" (L1154, L1199). Note the DEATHW smoke's 628 sits
  against the 200-235 claim. Eps: `--eps-end-at 1` in M6 makes eps = 0.01 from the first
  transition (`Schedules.eps`, `train_m3.py:98-100`); derived P(>= 1 eps action in 5
  decisions) = 1 - 0.99^5 = 4.9 %. The random action is uniform over mask-legal actions
  including boosts (`train_m3.py:497-499`).
- **Checkpoint-phase probe (20):** no 160k-granularity checkpoints exist; B8 has
  ckpt_5000k/10000k/15000k + final, L60 every 5M (`--checkpoint-every 5000000`, digest
  L435). 1.28M cycle = 2 actors x 128 worlds x 5000 frames (inferred, see A).
- **Seed ensemble (21, 28):** launched as track 1 of wf_2ee9a0af-657 (branch
  research-ensemble from census-b8; mean-Q of 5 B8 finals, mean-Q of CTRL finals, majority
  vote; GO if scripted >= 0.64 with mass >= incumbent+60) and returned null on the rate limit
  (digest L1052, L1145). Per-world seed correlation 0.035 (L1145). B8 per-seed scripted
  survival 0.66/0.38/0.62/0.53/0.47 (L1065).
- **champion_a5+v8 and frp3-s12+v8 on all five banks (22, 27):** champion ran only on seed
  0's bank (guard) and the census bank (0.688, 10 deaths); the incumbent ran on all five
  Phase R banks (records on the Mac, per-world not digested) and the census bank (0.656).
  Dev-probe incumbent per mix (n32, round 52): frozen 413/0.62, scripted 455/0.66, mixed
  437/0.75 (L261).
- **Corpse share / intrusion mechanics (24):** nothing measured; the census notes "an
  opponent head within 2 cells at the fatal decision in 17 %", "7 % hit an opponent's body
  within 8 cells of its head", 5 greedy_food opponents per scripted world.

---

## G. Flags vs code, Mac RAM, --obs-pack (question 29)

**Status: partial.**
- Flag-level in the quoted argv: `--replay` (capacity), `--anchor-hold / --anchor-floor-at /
  --anchor-floor`, `--anchor-mode {.., v8margin}`, `--onpolicy-labels`, `--label-frac`,
  `--shield v8 --shield-full-until --shield-off-at`, `--reward mass --mass-kappa`,
  `--kill-scale`, `--death-value`, `--death-mass-mult`, `--tight-cost --tight-min-len
  --tight-telemetry`, `--gamma-start/-end/-hold/-anneal-end`, `--eps-end-at`, `--worlds`,
  `--actors`, `--mixed-frac`, `--obs-version {b,bv8}`, `--v8-backend`, `--v61-backend`,
  `--learner-fast`, `--learner-precision`, `--batch-scale K --lr-scale`, `--obs-pack`,
  `--demo-mmap`, `--compile-learner`, `--compile-actor`, `--actor-impl reference`,
  `--segment-frames` (M3). **Not seen as flags:** n-step, Huber beta, replay sampling scheme
  (uniform only), label source other than the on-policy v8 buffer, actor stagger.
- Mac: "18 CPUs" (digest L267), M5 Pro cores (L795), MPS learner; RAM not stated. It ran two
  concurrent 8M/20M runs plus probes (M6 lanes; batch-scale A/B with `--obs-pack --worlds 64`).
- `--obs-pack`: see A (12,449 B/row unpacked for bv8; 1.167 -> 0.699 GB at 20k+20k rows;
  19 GB -> 6-7 GB per run). Packed bytes per row are not quoted.
- deepcopy: see F.

---

## H. One-line table

| # | Question (short) | Status |
|---|---|---|
| 1,7,16 | train_m3c loss/n-step/replay/segments/LabelBuffer/margin/lambda | partial |
| 14 | exact mass reward | partial |
| 23 | 30 bv8 feature names and caps | partial |
| 2,15 | census death frames / alive-frame survival | needs_mac_artifacts |
| 8,18 | look-2 per-world surv5/mi5 | needs_mac_artifacts |
| 3,9,13,17 | hazard by L x frame bucket | needs_mac_artifacts (series exist; crosstab not computed) |
| 26 | CTRL_s10 log/telemetry | needs_mac_artifacts (per-death x eps needs new code) |
| 4,10,11,12 | exact-replay counterfactuals, sealer, Q-gap | needs_new_measurement |
| 5,19 | fuse-feature probe | needs_new_measurement |
| 6 | training death records with eps flag | needs_new_measurement |
| 20 | checkpoint-phase probe | needs_new_measurement |
| 21,28 | seed ensemble | needs_new_measurement (track launched, result null) |
| 22,27 | champion/incumbent on all banks | needs_new_measurement |
| 24 | corpse share / intrusion | needs_new_measurement |
| 29 | flags vs code, RAM, obs-pack, deepcopy | partial |
