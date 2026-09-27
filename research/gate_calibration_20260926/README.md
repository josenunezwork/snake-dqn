# Gate calibration from saved H5000 records (2026-09-26)

**Status: every number here is UNAUDITED and descriptive.** They come from 144 saved
per-episode records of the `task-aligned-strict-challenge-20260926` pilot-v1 run. Nothing was
replayed, and n = 16 worlds per mix per stage. This is Tier-0 saved-data analysis. It steers the
design of a future gate and makes no promotion decision.

It answers review §7 item 1 ("calibrate the promotion instrument before training anything
else") as far as saved data allows. It closes with a gate design and an A/A protocol that can
be pre-registered.

## Run

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/gate_calibration_20260926/analyze.py \
  --out /private/tmp/claude-501/-Users-josenunez-Projects-ml-snake-dqn/8b38327f-2897-4918-b079-74962b22441c/scratchpad/gate_calibration/gate_calibration.json
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gate_calibration.py
```

About `analyze.py`:

- It uses only the standard library and numpy, plus the repo's stdlib-only
  `src.scripts.eval_stats` for the pilot's own sizing rule.
- It reads `<source>/{development,pilot}/producer/records/*.json` and each stage's
  `physical-report.json`.
- It writes exactly one JSON file, at `--out`, and refuses any path inside the research
  artifact tree.
- Its output is byte-identical across runs. This was checked with two runs and `cmp`.
- The run behind this README produced `records_sha256_digest`
  `672a630aa52b5300af8e4a7bf2689ad23e3d02e9cb263b4682648a6d8a460396`. That digest covers 144
  records:
  - development: 48 Apex;
  - pilot: 48 candidate (6102) and 48 paired Apex.

**Cross-check.** The script reproduces the saved pilot `sizing.json` exactly:

- paired SDs 56.32, 118.26 and 55.81;
- MDEs 5.26, 3.88 and 7.26;
- N = 1072, 8676 and 552.

`test_real_records_reproduce_saved_pilot_sizing` pins this.

**Inputs.**

- Development and pilot worlds are disjoint (overlap 0).
- Stage wall time was 17.74 s per Apex episode in development (851.7 s / 48). The pilot ran
  at 30.96 s per episode (2972 s / 96, a mix of candidate and Apex episodes).
- Every record has `deaths == 1`, because `hero_terminal` ends the episode at the first death.
  So `mass_integral = survival_fraction × mean_mass_alive` exactly, with a maximum identity
  error of 0.0.

## Results

### Apex per mix (mean / SD / CV)

| stage | mix | mass_integral | survival_fraction | log1p(mass) | mean_mass_alive |
|---|---|---|---|---|---|
| dev | frozen | 52.6 / 61.6 / 1.17 | 0.288 / 0.199 / 0.69 | 3.15 / 1.57 / 0.50 | 125.7 / 89.4 / 0.71 |
| dev | scripted | 38.8 / 40.4 / 1.04 | 0.238 / 0.169 / 0.71 | 2.81 / 1.63 / 0.58 | 111.6 / 79.2 / 0.71 |
| dev | mixed | 72.6 / 65.7 / 0.90 | 0.372 / 0.189 / 0.51 | 3.79 / 1.27 / 0.34 | 157.0 / 81.2 / 0.52 |
| pilot | frozen | 65.9 / 50.9 / 0.77 | 0.339 / 0.184 / 0.54 | 3.59 / 1.46 / 0.41 | 153.1 / 81.7 / 0.53 |
| pilot | scripted | 92.2 / 114.0 / 1.24 | 0.343 / 0.264 / 0.77 | 3.38 / 1.96 / 0.58 | 169.5 / 139.0 / 0.82 |
| pilot | mixed | 71.4 / 53.3 / 0.75 | 0.368 / 0.183 / 0.50 | 3.76 / 1.38 / 0.37 | 158.3 / 78.2 / 0.49 |

- Per-world CV of mass is 0.75–1.24.
- Apex survives only 24–37% of the horizon on average.

### A/A across banks: same policy, disjoint worlds

The records support only one A/A contrast: Apex on the development bank vs Apex on the pilot
bank. The table gives the pilot minus development difference and its Welch t (df ≈ 19–30).

| mix | mass Δ (t) | survival Δ (t) | log1p Δ (t) |
|---|---|---|---|
| frozen | +13.3 (0.66) | +0.052 (0.76) | +0.45 (0.83) |
| scripted | +53.4 (1.77) | +0.105 (1.35) | +0.57 (0.90) |
| mixed | −1.2 (−0.06) | −0.004 (−0.06) | −0.03 (−0.07) |

This answers the review's open question in §5. The scripted bank means of 38.8 and 92.2 differ
by a 2.4× ratio, yet that gap sits within ordinary world-to-world noise (t = 1.77, driven by a
heavy right tail). It is sampling noise between 16-world banks, not a bank-construction
effect.

This matters for sizing. The pilot fixed its MDE as 10% of a single 16-world development mean,
so the MDE itself was noisy. The scripted MDE of 3.88 came from the low draw of 38.8, and that
low draw alone inflated N to 8,676.

### Paired 6102 vs Apex on the 16 pilot worlds per mix

- **Δ** is candidate minus Apex.
- **r** is the Pearson correlation of the candidate's and Apex's per-world values, which
  measures how much pairing helps.
- **ratio** is (Var c + Var a) / Var Δ. Values above 1 mean pairing helps.
- **d** is the observed mean Δ divided by SD Δ.
- **N** is the pilot rule's required count: paired t, 80% power, α = 0.05/3, with MDE equal to
  10% of the development Apex mean.

| mix | estimand | mean Δ | SD Δ | r | ratio | d | N |
|---|---|---|---|---|---|---|---|
| frozen | mass_integral | −30.2 | 56.3 | −0.01 | 0.99 | −0.54 | 1072 |
| frozen | log1p mass | −0.27 | 1.49 | 0.30 | 1.39 | −0.18 | 209 (2278 at ×1.10) |
| frozen | survival_fraction | +0.051 | 0.238 | 0.06 | 1.06 | +0.21 | 643 |
| frozen | ridit(mass) | −0.133 | 0.406 | 0.03 | 1.02 | −0.33 | 480 |
| frozen | sign(Δmass) | −0.375 | 0.957 | – | – | −0.39 | 857 |
| frozen | mean_mass_alive | −72.1 | 81.5 | 0.18 | 1.13 | −0.88 | 393 |
| mixed | mass_integral | −34.9 | 55.8 | 0.11 | 1.09 | −0.63 | 552 |
| mixed | log1p mass | −0.33 | 1.28 | 0.38 | 1.42 | −0.25 | 107 (1686 at ×1.10) |
| mixed | survival_fraction | +0.026 | 0.205 | 0.24 | 1.31 | +0.13 | 284 |
| mixed | ridit(mass) | −0.191 | 0.339 | 0.31 | 1.38 | −0.56 | 303 |
| mixed | sign(Δmass) | −0.375 | 0.957 | – | – | −0.39 | 857 |
| mixed | mean_mass_alive | −74.3 | 74.8 | 0.29 | 1.21 | −0.99 | 213 |
| scripted | mass_integral | −66.4 | 118.3 | −0.22 | 0.95 | −0.56 | 8676 |
| scripted | log1p mass | −0.35 | 2.22 | −0.05 | 0.96 | −0.16 | 583 (5064 at ×1.10) |
| scripted | survival_fraction | −0.045 | 0.303 | −0.14 | 0.91 | −0.15 | 1513 |
| scripted | ridit(mass) | −0.133 | 0.405 | 0.03 | 1.03 | −0.33 | 478 |
| scripted | sign(Δmass) | −0.375 | 0.957 | – | – | −0.39 | 857 |
| scripted | mean_mass_alive | −93.0 | 143.3 | −0.05 | 0.98 | −0.65 | 1540 |

- **Mass.** 6102 won 5 of 16 worlds in every mix (sign test p = 0.21 in each).
- **Survival.** Win/loss counts were 6/10 (frozen), 11/5 (mixed) and 8/8 (scripted).
- **Ridit.** Ridit is the pooled rank within a mix, scaled to (0, 1). A mean ridit Δ of δ
  corresponds to a Mann-Whitney P(candidate > Apex) of about 0.5 + δ. The 10% MDE, 0.057,
  therefore means P ≈ 0.557.
- **Sign.** The sign MDE of 0.1 means P(win) of 0.55.
- **log1p at ×1.10.** The value in brackets uses a multiplicative MDE of log 1.1.

### How much of the variance is death timing

Every record ends at the hero's first death, so the log of mass splits exactly:

log m = log s + log a,

where s is survival_fraction and a is mean_mass_alive.

| group | corr(mass, survival) | R² | Spearman | Var(log m) share from log s / log a / 2·cov |
|---|---|---|---|---|
| Apex dev, all 48 | 0.957 | 0.915 | 0.994 | 0.24 / 0.26 / 0.50 |
| Apex pilot, all 48 | 0.929 | 0.862 | 0.991 | 0.24 / 0.26 / 0.50 |
| 6102 pilot, all 48 | 0.969 | 0.939 | 0.979 | 0.31 / 0.20 / 0.49 |
| per mix, Apex (6 cells) | 0.957–0.980 | 0.916–0.959 | 0.965–0.997 | about 0.24 / 0.26 / 0.50 |

For the paired deltas, the fit of Δmass on Δsurvival has R² 0.85–0.90 per mix, and the fit of
Δlog mass on Δlog survival has R² 0.994.

**Reading.** Most of the outcome variance is when the hero dies. A snake that lives longer
also grows larger, which is the large 2·cov term. The 6102 effect, however, sits in growth
while alive: mean_mass_alive has the largest standardized effect, d = −0.65 to −0.99, while
the survival effect is small and changes sign across mixes. **Death timing is the noise; for
6102, growth was the signal.** A safety veto is the opposite case: its effect is designed to
act on survival.

### Death causes

| group | self | head_on | enemy_body | wall |
|---|---|---|---|---|
| Apex dev (48) | 40 | 6 | 1 | 1 |
| Apex pilot (48) | 45 | 3 | 0 | 0 |
| **Apex total (96)** | **85 (89%)** | **9** | **1** | **1** |
| 6102 pilot (48) | 36 | 2 | 8 | 2 |

No episode reached the horizon alive.

- **Apex.** Self-collision is the dominant cause in every mix. The records name the cause but
  not whether the safe set was already empty, so the pre-trap share (review §3.1) cannot be
  computed from these records.
- **6102.** It died on an enemy body in 8 of 48 episodes. Apex did so in 1 of 96.

### MDE ladders

The ladders below use the pilot's paired SDs. The table shows required worlds per mix, with the
binding mix in bold.

**Raw mass, with the MDE set as a fraction of the pooled Apex mean.** That mean pools 32 worlds
per mix: frozen 59.2, scripted 65.5, mixed 72.0.

| MDE | frozen | mixed | scripted |
|---|---|---|---|
| 10% | 845 | 561 | **3047** |
| 20% | 212 | 141 | **762** |
| 30% | 94 | 63 | **339** |
| 50% | 40 (floor) | 40 (floor) | **122** |

**log1p mass, with a multiplicative MDE.**

| MDE | frozen | mixed | scripted |
|---|---|---|---|
| ×1.10 | 2278 | 1686 | **5064** |
| ×1.20 | 623 | 461 | **1384** |
| ×1.30 | 301 | 223 | **669** |
| ×1.50 | 126 | 94 | **280** |

At the same MDE, the log scale needs more worlds than raw mass. The paired table's "N = 107–583
for log1p" uses 10% of the mean of log1p, which is a 28–38% multiplicative effect. Those
numbers are not comparable to raw-mass Ns and should not be quoted as a saving.

### Non-binding futility rule, evaluated on the pilot

**Rule.** At each look, stop if in any mix the one-sided (1 − 0.10/3) upper t bound on mean
paired Δ is below 0.

The table shows the upper bound for each mix, in the order frozen / mixed / scripted.

| look (worlds per mix) | raw mass | stop | log1p | stop |
|---|---|---|---|---|
| 8 (first 8 by seed) | −28.7 / +10.2 / +38.2 | yes (frozen) | −0.36 / +0.39 / +2.23 | yes (frozen) |
| 16 | −2.3 / −7.3 / −8.0 | yes (all three) | +0.47 / +0.31 / +0.74 | no |

- The rule on raw mass flags 6102 as a clear loser within 16 worlds per mix, or within 8 if
  the first look comes that early. On log1p it does not reach a stop at 16.
- This matches the d column: 6102's deficit concentrates in high-mass worlds, which the log
  scale compresses. For the one real contrast on record, raw mass was the more sensitive
  estimand.
- This table is a single-look illustration. Repeating the same fixed per-look bound at several
  looks does not control the cumulative false-stop rate (see "Sequential futility stop"
  below): at four looks it falsely stops about 27% of truly equal candidates.

### Throughput

The incumbent arm is deterministic given the world seed (see the A/A section), so it runs once
per world bank and is cached. Each candidate then costs 3·N candidate episodes. Figures below
use 17.74 s per live H5000 episode on CPU. The batch-engine column uses the **3–4× projection**
of `docs/research/simd_vector61_plan_2026-09-26.md` ("Measured cost"), not the ~100× the
engine achieves for non-vector61 policies: for vector61 the featurizer (0.09–0.11 ms/row on
natural states, about 0.2 ms/row with long bodies, vs 0.84–0.90 ms/row for live `get_state`)
dominates, giving roughly 1–1.3 ms against 3.55 ms per live env-frame when all six slots are
vector61. That projection is not measured end to end, and action parity is not yet shown.

| design (binding N per mix) | candidate episodes | live CPU hours | batch engine at 3–4× (projected) |
|---|---|---|---|
| pilot rule, raw mass 10% of dev mean (8676) | 26,028 | 128 | 32–43 h |
| raw mass 10% of pooled mean (3047) | 9,141 | 45 | 11–15 h |
| **raw mass 20% of pooled mean (762)** | **2,286** | **11.3** | **2.8–3.8 h** |
| log1p ×1.20 (1384) | 4,152 | 20.5 | 5.1–6.8 h |
| ridit P≈0.557 (480) | 1,440 | 7.1 | 1.8–2.4 h |

The batch-engine figures are hypothetical today: `src/simd_env/eval_engine.py` does not
featurize vector61 checkpoints. It raises an error, as of the brief for this lane. The gate
needs:

- vector61 featurization in the batch sim;
- a parity check against the live H5000 promotion profile, including frozen Apex opponents and
  scripted anchors. `docs/simd_env_spec.md` is the parity contract.

Live episodes average about 0.3 × 5,000 hero frames. A 1-hour gate at 762 worlds per mix needs
about 0.64 candidate episodes per second, which is at most 3,200 scored frames per second.

## What this says about gate design

1. **Pairing buys nothing for mass.**
   - Candidate and Apex per-world mass are uncorrelated: r = −0.22 to 0.11, and the pairing
     variance ratio is 0.95–1.09.
   - Pairing helps a little on log1p and ridit in frozen and mixed (ratio about 1.4), and not
     at all in scripted.
   - The mechanism: once the two policies take one different action, the episode diverges, so
     a world acts almost as an independent draw for each arm.
   - Pairing is still worth keeping for bias control and because it lets the incumbent arm be
     cached, but N must be planned as if the arms were nearly unpaired.
   - Covariate adjustment on Apex's own per-world score (CUPED-style) would remove about
     r² ≈ 0–5% of variance. It is not worth the complexity.
2. **Changing the estimand is not a free lunch.**
   - At an equal multiplicative effect, log1p needs more worlds than raw mass: 5,064 vs 3,047
     at 10%.
   - It was also less sensitive to the one real deficit on record: d = −0.16 to −0.25 vs −0.54
     to −0.63.
   - Rank and sign estimands are robust to the heavy tail, but they answer a different
     question, P(win), not mean mass.
3. **The pilot's 8,676 was inflated by a noisy MDE reference.**
   - With the same SDs, anchoring the 10% MDE on a pooled 32-world Apex mean gives 3,047.
   - The reference mean should come from a large cached Apex bank, not from 16 worlds.
4. **Death timing is the dominant noise source** (R² 0.86–0.95 within arm). Self-trap is 89%
   of Apex deaths, and no Apex episode reached the horizon alive. Interventions that change
   survival therefore also change the noise.
5. **A futility stop is cheap and would have caught 6102.** It stops by 16 worlds per mix on
   raw mass, and by 8 in the frozen mix. This is a single-look check; a repeated schedule must
   spend α across looks (see "Sequential futility stop" below).

## Recommended gate: draft for pre-registration

Every value below is a proposal to be frozen before any candidate episode runs, using
development-only data.

- **Unit and pairing.**
  - A world is a (mix, seed) pair drawn from a fresh, registered seed namespace that
    overlaps no earlier bank.
  - Every arm plays the same worlds, with equal N in all three mixes.
  - The Apex arm runs once per bank and is cached. It is valid only after the A/A-identical
    check below passes on that code revision.
- **Primary estimand.** The per-mix mean paired Δ of `mass_integral` (`logical-mass/v1`,
  profile `promotion-v2-watch-rect`, H5000), which is unchanged from the deployment metric.
- **Secondary estimands.** These are reported, with no gating role:
  - survival_fraction and mean_mass_alive, the exact decomposition;
  - ridit and win-rate, as robustness checks;
  - a death-cause census.
  
  A candidate whose mechanism is survival, such as the flood-fill veto, pre-registers
  survival_fraction as its mechanism check.
- **MDE.** 20% of the cached Apex bank mean per mix. That bank has at least 256 worlds per
  mix, measured before the gate. A 10% MDE costs about 45 live CPU hours per candidate, or a
  projected 11–15 hours on the batch engine once it featurizes vector61, so it is not a
  routine gate.
- **N.**
  - Set N from a development-only paired SD, with the largest mix binding.
  - With today's only paired SDs (6102), 20% gives **N = 762, rounded up to 768 worlds per
    mix**. At 10%, N ≈ 3,050.
  - Because a new candidate's Δ SD can differ, pre-register an upward-only re-estimate of SD
    at the second look, with a hard cap. The recommended cap is 3,072 per mix on the batch
    engine (a projected 11–15 hours per candidate at 3–4×) and 768 live.
- **Test.**
  - A one-sided paired t per mix.
  - Holm across the three mixes at a family α of 0.05.
  - Keep the existing scripted non-inferiority margin from `src/evaluation/strict_promotion.py`.
  - Efficacy is tested only at the final look.
- **Sequential futility stop.** This stop is non-binding: it can only stop a candidate, never
  promote one. It spends a futility α of 0.10 across looks and mixes with
  `src.evaluation.screen_stats.futility_plan` (Lan–DeMets O'Brien–Fleming-type spending),
  evaluated per look with `evaluate_futility_look`.
  - Looks come at 32, 128, 384 and 768 worlds per mix (information fractions 1/24, 1/6, 1/2,
    1).
  - Each mix gets `futility_plan([32, 128, 384, 768], alpha=0.10/3)`. Its z boundaries are
    10.36, 5.08, 2.79 and 1.85, and its cumulative α spent is 0, 0, 0.0026 and 0.0333.
  - At each look, stop as a "clear loser" if any mix's tail-matched paired statistic is below
    −boundary.
  - For a truly equal candidate, the cumulative chance of a false stop over all four looks is
    at most 0.10 (Bonferroni over mixes). A Monte Carlo with 3 independent mixes and Brownian
    looks gives 0.096.
  - Alternative considered and rejected: repeating a fixed one-sided (1 − 0.10/3) t bound per
    mix at every look. It is about 10% per look (0.096 at the first look), but the same
    Monte Carlo gives a **cumulative false stop of about 27%** over the four looks, so it
    would kill about a quarter of truly equal candidates and cut power for small real gains.
  - Being non-binding, neither rule raises the promotion false-positive rate.
  - O'Brien–Fleming spending puts almost no α at the early looks: the 32-world boundary of
    10.36 effectively never stops. Catching a clear loser early is the job of the Tier-1
    check, not of this in-gate stop (next item).
  - The single-look pilot table above, where a fixed bound stops 6102 at 16 worlds, remains
    an illustration of effect size, not the recommended schedule.
- **Relation to the Tier-1 check.** The mandatory Tier-1 "not a clear loss" check is defined
  normatively in `docs/research/governance_tiers_2026-09-26.md` ("Rule: a cheap H5000 check
  against Apex"): at least 16 worlds per mix, the pooled mean of per-mix means
  (`screen_stats.stratified_mean_of_means`), 90% CI, stop if the pooled upper bound is below 0.
  This README does not define a second version. The futility looks above belong to the
  Tier-2 gate only.
- **Throughput.**
  - Live, 768 worlds per mix is about 11.3 CPU hours per candidate. That is acceptable for a
    final overnight gate but not for screens.
  - On the batch engine the projection is 3–4× cheaper (about 2.8–3.8 hours for a full gate),
    not 100×. See `docs/research/simd_vector61_plan_2026-09-26.md`. It needs vector61
    featurization wired into `src/simd_env/eval_engine.py`, action parity, and parity CI
    against this live profile.
- **Variance reduction to test on development data before freezing.** Adopt none of these
  without a development measurement:
  1. A respawn-continuation estimand, with `hero_terminal` off: `mass_integral` over all 5,000
     frames, averaging several lives per world. Deaths currently come at 24–37% of the
     horizon, so about 3 lives per world could cut per-world variance by up to about 3×, if
     lives are roughly independent. This changes the estimand and must be justified against
     deployment.
  2. A decomposition gate: separate pre-registered tests on survival and on growth while alive.
     It adds power when the mechanism is known in advance.
  
  Several episodes per world are not a lever. Rollouts are deterministic given the seed, so
  they amount to more worlds.

## A/A protocol (Apex vs Apex)

**Why the classic A/A is degenerate here.** `tournament_eval.rollout()` calls
`set_seed(seed)` and derives each world and each slot's policy seed from the world seed. CPU
inference is deterministic at a fixed thread count. Apex vs Apex on the same world therefore
produces identical records: Δ ≡ 0 and SD Δ = 0. **An A/A on shared worlds measures
determinism, not noise.** Its "null variance" of zero must never feed a power calculation. The
noise a gate faces comes from two sources:

- between-world variance, which the disjoint-bank A/A measures;
- chaotic divergence once policies differ, which the paired candidate SD measures.

**Protocol.** Run under `SNAKE_DQN_DEVICE=cpu` with fixed torch and OMP threads:

1. **A/A-identical (determinism).**
   - Run Apex twice on the same 16 worlds per mix, in separate processes. One run should use a
     different process order, and one a different thread count, to probe sensitivity.
   - Pass only if every per-world record is identical: `mass_integral`, `survival_fraction`,
     `mean_mass_alive`, `max_mass`, `death_cause`, `denominators` and `kills`.
   - A pass authorizes caching the incumbent arm. Any difference is a defect to fix before any
     gate, not noise to average.
2. **A/A-cross-bank (world variance).**
   - Run Apex on two disjoint banks and compare them with Welch t per estimand. Expect
     |t| < 2 in most cells.
   - The pooled SD sizes the MDE reference bank.
   - The saved records already give one instance: the table above, with t between −0.07 and
     1.77.
3. **A/A-perturbed (optional; how fast pairing decays).**
   - Force one different legal action at frame k for k in {50, 500, 1500}, with Apex
     unchanged otherwise.
   - Report the per-world correlation and the SD of Δ against Apex.
   - If one forced action already gives r ≈ 0, pairing can never cut variance for any real
     candidate. That would confirm point 1 above as a property of the simulator, not of 6102.
4. **Ordering check** (from the blueprint): Apex > scripted greedy_food anchor > random_safe on
   the same worlds. This confirms the instrument ranks known-different policies.

**What the dev_screen C arm will verify.** `research/apex_safety_20260926/dev_screen.py`
belongs to another lane. Integration confirmed its C arm is Apex re-run on arm A's worlds
after all A/B episodes, through the same `rollout` call with `hero_safety_veto=False` (the
opt-in veto code present but disabled), and that `summarize` requires every C record to equal
its A record under canonical JSON (`INVALID_NONDETERMINISTIC` otherwise).

- C must reproduce A's per-world records exactly, which is step 1 above at dev-screen scale.
- That match shows:
  - the rollout is deterministic, so the Apex arm can be cached;
  - the default-off wiring of the new veto path does not change behavior;
  - paired Δ = B − A is attributable to the veto alone.
- Any nonzero C − A Δ fails the screen as a defect. Its SD is not an estimate of noise.

## Caveats

- All numbers are unaudited and descriptive, from saved JSON records with n = 16 per mix per
  stage. Nothing was replayed, and the per-record SHA-256 values are listed in the output JSON
  only as a digest.
- All paired statistics come from one candidate, 6102. The Δ SD of a candidate closer to Apex
  may be smaller, and the pairing correlation may be higher, if it diverges later. Treat N as a
  planning figure, not a guarantee.
- SDs from 16 heavy-tailed worlds are themselves imprecise. The scripted SD (118) rests on a
  few large-mass worlds.
- The sizing rule is the pilot's per-mix 80% marginal power at α = 0.05/3. It is not joint
  power for Holm together with non-inferiority.
- The ridit and sign Ns come from paired-t approximations on rank-transformed data. They
  indicate magnitude only.
- Batch-engine throughput assumes the 3–4× projection from
  `docs/research/simd_vector61_plan_2026-09-26.md` (not measured end to end) and vector61
  support in `eval_engine`, which does not yet exist. Live time uses the development stage's 17.74 s per Apex episode. Candidate episodes through a
  serving bridge were slower in the pilot, at 30.96 s averaged over both arms.
- The records carry no fallback or safe-set data, so the self-trap share here counts
  self-collisions, not pre-trap decisions.
