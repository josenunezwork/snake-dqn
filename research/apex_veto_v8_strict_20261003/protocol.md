# Apex + v8 veto (lambda=8) vs Apex + released v7 veto (lambda=4): Tier-2 group-sequential strict challenge (pre-registration)

Method `strict-sequential-obf-bonferroni-v1`, runner `sequential-strict-template/v1` (first
production use of the template). Governance: Tier 2 of
[`governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md) as amended by
[`governance_amendment_sequential_gates_2026-10-02.md`](../../docs/research/governance_amendment_sequential_gates_2026-10-02.md)
(including its ratified implementation notes). Written 2026-10-03 (UTC) after the v8 Tier-1
screen closed `ADVANCE` and **before any calibration or final episode of this study and before
any intent**. A `STRICT_PASS` produces a receipt only; it changes no champion file, default,
config, released veto or deployment. Release is a separate, explicit action (and would need its
own web serving qualification, as v7's did).

Files: `spec.py` (the `StudySpec`), `preregistration.py` (generates the three inputs below),
`screen_deltas.json` (skew-check input), `operating_characteristics.json` (OC report),
`look1_band_cost.json` (band-cost report), `dry_run.py` (plumbing dry run), tests
`tests/test_apex_veto_v8_strict.py`.

## Question and decision it informs

Does replacing the released v7 space-preference veto (lambda 4) with the v8 space-and-head veto
at lambda 8 on the served Apex hero improve the H5000 mass integral, strictly (OBF-sequential
Bonferroni superiority in >= 2 of 3 mixes, scripted non-inferiority, survival bands), on fresh
worlds? `STRICT_PASS` allows a later, separate release action to consider v8 (after a web
serving qualification). `STRICT_FAIL` keeps v7; it does not show that v8 is worse.

## Arms and identities

| Role | Checkpoint | Wrapper (hero only; opponents unwrapped) |
|---|---|---|
| Incumbent (released) | `champion_a5_freespace_20260621.pth` `43d4e2c5…d747ac93` | `free-space-veto/v7-space-preference(lambda=4.0)`: `safety_veto_v7.py` `56ff7009…7e2980`, `safety_veto.py` `1b62d15c…c428`, `safety_veto_v3.py` `ed3a6d86…b5be1`, `safety_veto_v5.py` `d86d084e…0ec86c` |
| Candidate | the same bytes | `free-space-veto/v8-space-and-head(lambda=8.0)`: `safety_veto_v8.py` `faf3695f…e4ac05` plus `safety_veto_v7.py`, `_v6.py` `a6117cb9…1757d5`, `_v5.py`, `_v4.py` `3f0881af…e44578`, `_v3.py`, `safety_veto.py` |

`spec.arm_identities()` computes every sha256 from the files and the descriptors from the
classes, and fails closed unless: the incumbent's method, descriptor, checkpoint and four
source sha256s equal the **v7 STRICT_PASS receipt**'s candidate
(`apex-veto-v7-strict-20261002/run-v1/output/receipt.json`, sha256 `86ee3679…f750422`, the
receipt `web/backend/safety_veto_serving.py` pins for the served default since `da951d2`); the
candidate's seven veto modules equal the pinned sha256s, which are the bytes at the v8 screen
commit `e2606178` (a test checks `git show`); the candidate descriptor has
`space_preference_lambda == 8.0` and `head_avoidance == true`. The identities are frozen in the
intent, recomputed before every segment by the template (any drift stops the run), and each
record envelope binds its arm identity's sha256. Install: `dev_screen.hero_veto_installer`
around `tournament_eval.rollout(hero_safety_veto=True)` (the built-in vector61 guard runs
first), exactly as the v7 strict package and the v8 screen. The candidate is installed
**without** the screen's diagnostic-only `reference_lambda` (not in the descriptor; it never
changes a decision; the served form would not carry it). Non-veto imports are bound by the
source closure (`src/`, the template, the validation scripts, `research/apex_safety_20260926`,
this package).

Every episode re-hashes the four pool checkpoints before `rollout` loads them (fail closed),
validates the record with `strict_promotion.validate_strict_world_record(...,
candidate_wrapper=<arm descriptor>)` (profile digest `d396d3ed…0e8b`, world identity from the
roster row, the v2 probe counters), and attaches the arm's `veto_diagnostics` (v7 or v8
counters; reported, not gated). `validate_record` rechecks the probe method, the diagnostics
kind and the world identity on the envelope.

## Profile, mixes, rosters, worlds

- Profile `promotion-v2-watch-rect`, H5000; config `research/apex_safety_20260926/deployment.yaml`
  (sha256 `4146baa3…715aa5`, checked by every worker).
- Mixes frozen, scripted, mixed; rosters from `dev_screen._design_rows` (strict balanced
  construction, slot members rotating by world index; a bank prefix has the prefix rows).
- Seeds `uint32_be(sha256("<domain>|worlds|<i>")[:4])`:

| Namespace | Count | Use |
|---|---|---|
| `apex-veto-v8-strict-dev-v1` | 16 | incumbent-only calibration, 3 mixes |
| `apex-veto-v8-strict-final-v1` | ordered bank of N_max = 249 | final paired worlds, world-major round robin |

`excluded_seeds()` (checked fail closed by `prepare`, recorded in the intent) holds: the first
1000 seeds of every earlier domain/purpose (a frozen copy of the v8 screen's table, which
covers the v8 screen `apex-veto-v8-screen-v1` and DEV sweep `apex-veto-v8-dev-v1` and their
smokes, every v7 bank, the v7 web serving lanes and every earlier veto/strict/serving/trap
domain; a test asserts the copy is a superset of the screen's); the five task-aligned challenger
namespaces; the strict pilot's observed seeds; seeds 0..999; and the seeds saved in the v7
strict rosters and intent, the v7 sweep/screen/serving intents, the v8 DEV sweep intents and
summary, and the v8 screen intents and summary (missing file = refusal). No smoke namespace:
the template has no smoke mode.

## Calibration (incumbent only, before any final world)

16 worlds x 3 mixes (48 v7 episodes). `delta_NI = 0.03 x calibration incumbent scripted mean
mass_integral` (the v7 strict rule; planning value 9.42 from the screen's v7 arm). Bands (the
v7 strict rule): per mix, candidate mean `survival_fraction` in `[ref - 0.02, ref + 1]`, `ref`
= calibration incumbent mean of that mix. Written create-only to `calibration.json`.

## Plan (frozen in the intent)

- N_max = **249** worlds per mix; looks at fractions 0.25/0.5/0.75/1.0 -> sizes **63, 125, 187,
  249** (actual fractions 0.253/0.502/0.751/1).
- Efficacy per mix: one-sided H0 delta <= 0, Lan-DeMets OBF at 0.05/3. Boundaries (z) 4.6175,
  3.1836, 2.5427, 2.1734; nominal p 1.94e-6, 7.27e-4, 5.50e-3, 1.487e-2. Required successes 2
  of 3 (a crossed mix stays crossed).
- Scripted NI: own OBF at 0.05, boundaries 3.7252, 2.5341, 2.0148, 1.7205 (nominal p 9.76e-5,
  5.64e-3, 2.196e-2, 4.267e-2); NI when `mean - t * se > -delta_NI`.
- Bands: `block_at_stop`, judged once at the qualifying look with margin `1.645 x sd x
  (1/sqrt(n_k) - 1/sqrt(249))`.
- Futility: non-binding CP < 0.10 under MDE 30, `futility_policy = "followed"`,
  `futility_action = "stop"`.
- MDE **30** mass-integral units per mix (sizing below). Interleaving: unit `u = world x 3 +
  mix`, worker `u % 2` runs both arms (incumbent first). 2 workers (strict policy), template
  default caps (calibration 1800 s, skew check 900 s, final 45000 s, audit 900 s, handoff 120
  s), no retry, no resume, global slot lock root and global ledger, `dry_run = false`.

## Sizing (rule stated before any strict data; development variance only)

Rule (`preregistration.py`, `SIZING_RULE`): per-mix SD = sample SD of the screen's 60 paired
deltas; `n_fixed(MDE)` = smallest n with `n >= ((t_{1-0.05/3,n-1} + t_{0.90,n-1}) x sd_max /
MDE)^2` (90% per-mix power at the MDE in every mix); `N_max = ceil(1.02 x n_fixed)` (the
amendment's ~2% for four OBF looks); MDE = the smallest multiple of 5 with N_max <= 300 **and**
the projected final-stage time per worker <= 70% of the worker budget. The screen's observed
means are not used. Screen SDs: frozen 136.2, scripted 136.8 (binding), mixed 81.6. Result:
MDE 25 -> N_max 359 (infeasible), **MDE 30 -> n_fixed 244, N_max 249**, projected 30,029 s per
worker = 66.8% of 44,970 s (basis: the screen's mean episode times, v7 41.3/24.7/33.2 s and v8
46.4/28.5/35.6 s for frozen/scripted/mixed, 3 concurrent shards, the v8 arm also running a
reference v7; x1.15 overhead). The MC confirms 90.4% / 90.2% per-mix sequential power at the
MDE (frozen / scripted SD).

## Operating characteristics (`operating_characteristics.json`)

Normal deltas with the per-mix screen SDs (simulate.py's `Replica`, reused unchanged; bands not
simulated here), 200k replicates per null row, 50k per power row:

| Scenario (frozen, scripted, mixed) | P(PASS) | familywise false rejection | E[worlds/mix] |
|---|---|---|---|
| global null (0,0,0) | 0.0006 | 0.033 | 195 |
| one effect, two nulls (130,0,0) | 0.021 | 0.033 | 222 |
| scripted at NI margin (130,-9.42,130) | 0.050 | - | 247 |
| MDE in every mix (30,30,30) | 0.989 | - | 178 |
| **screen means (38.6, 19.6, 1.4)** | **0.543** (frozen crosses 0.99, scripted 0.54, mixed 0.03) | - | 229 |
| frozen at its 90% LB (15.8, 19.6, 1.4) | 0.218 | - | 236 |

Boundary precision (2M replicates): efficacy any-look at zero 0.01677 (target 0.01667), NI at
its null 0.0502 (target 0.05).

**Joint bootstrap of the screen's world records** (world indices resampled jointly across mixes;
candidate survival from the same worlds; a resampled 16-world calibration reference and
delta_NI; bands at the qualifying look; futility followed; replica cross-checked against
`run_sequential_gate` decision by decision, 300/300 agree):

| Scenario | P(PASS) | STOP_FAIL_BANDS | FINAL_FAIL | STOP_FUTILE | stop look 1/2/3/4 | E[worlds/mix] |
|---|---|---|---|---|---|---|
| **observed screen effects** | **0.278** | 0.134 | 0.531 | 0.057 | 0.00/0.02/0.30/0.68 | 228 |
| screen shape shifted to the MDE in every mix | 0.472 | 0.469 | 0.059 | 0.000 | 0.00/0.28/0.60/0.12 | 177 |
| screen shape at the global null | 0.0004 | 0.000 | 0.220 | 0.779 | - | 197 |

Honest expectation: at the screen's effects the gate needs frozen **and** scripted (mixed is
about 0: 3% crossing), scripted crosses about 54% of the time, and the efficacy+NI gate passes
about 55%. **The survival bands then roughly halve that, to about 28%**: the band reference is a
16-world calibration mean whose own SD (about 0.05-0.08) exceeds the 0.02 band, and v8's
survival gain over v7 is small (screen +0.068 / +0.047 / +0.007), so a candidate with equal
survival fails a band about 36-40% of the time even at the last look (`look1_band_cost.json`).
This is a property of the inherited v7 band rule, kept unchanged as required; it is disclosed,
not tuned. A `STRICT_FAIL` here is therefore weak evidence about v8's mass effect; the look
receipts record which component failed.

## Early-stop band cost at look 1 (`look1_band_cost.json`)

Amendment definition (candidate equal to the reference, reference known, pilot v8 SD of
survival_fraction 0.213 / 0.260 / 0.204): fail probability at a look-1 stop (n = 63, margin
0.022 / 0.027 / 0.021) = 0.53 frozen, 0.58 scripted, 0.52 mixed (final look: 0.07 / 0.11 /
0.06). With the 16-world calibration noise: 0.51 / 0.53 / 0.51 at look 1 and 0.38 / 0.40 / 0.36
at the final look. At the screen's survival differences, with calibration noise: 0.17 / 0.31 /
0.46 at look 1, 0.09 / 0.20 / 0.31 at the final look. Look-1 stops are rare here (< 0.1%).

## Skew check

Input `research/apex_veto_v8_strict_20261003/screen_deltas.json` (the screen's 60 paired deltas
per mix, recomputed from the raw A/B records and checked against `summary.json`; sha256 frozen
in the intent). Thresholds 0.020 efficacy any-look (every mix), 0.06 NI any-look (scripted
only); 300k resamples; remedy on failure: stop and escalate (no final world). Planning-only
pre-run at the planning delta_NI (not binding; the binding run uses the calibrated delta_NI):
efficacy 0.0129 / 0.0137 / 0.0184 (frozen / scripted / mixed; mixed skewness -0.43, close to
the 0.020 limit), scripted NI 0.0444: would pass.

## Run, stopping, outcomes, audit

As in the template README (no deviation): calibration -> skew check -> looks with barrier and
create-only look receipts -> independent stdlib audit (`sequential_audit.py`) -> closeout
(`receipt.json` only on `STRICT_PASS`). Outcomes `STRICT_PASS`, `STRICT_FAIL`,
`SKEW_CHECK_FAILED`, `INCOMPLETE`, `INVALID_STOP`; never relabelled, never resumed or rerun,
never stopped early outside these rules. After an early stop per-mix means are naive and
descriptive only.

Expected duration: calibration about 0.25 h; skew check under 1 min; each look about 2.1 h;
final about 7.6 h at the screen's effects (8.3 h at N_max); audit minutes. The deadline is set
at `prepare` to at least 13 h 32 min after it (the template's caps-plus-handoff floor).

## Operator commands

```
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
$PY research/sequential_strict_template/sequential_runner.py prepare \
  --spec research.apex_veto_v8_strict_20261003.spec:SPEC \
  --out-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-strict-20261003/run-v1 \
  --n-max 249 --mde 30 --n-calibration 16 \
  --skew-input research/apex_veto_v8_strict_20261003/screen_deltas.json \
  --deadline-utc <now + 16 h, UTC> --authorization-quote "<the user's words>"
caffeinate -dimsu $PY research/sequential_strict_template/sequential_runner.py run \
  --intent <root>/intent.json        # inside tmux session v8strict, on AC power
$PY research/sequential_strict_template/sequential_runner.py status --intent <root>/intent.json
```

Before `run`: AC power (`pmset -g batt`), clean committed tree, no other job holding the CPU
slot locks (`cpu-slot-{1,2,3}.lock`), no other heavy CPU load started for the duration.

## Non-claims

No champion file, default, config, released veto or deployment changes on any outcome. A
`STRICT_PASS` does not qualify the web serving path. Screen numbers are planning inputs only.
The result covers lambda = 8 only, one wrapped hero against unwrapped opponents, under
`promotion-v2-watch-rect` H5000 only.
