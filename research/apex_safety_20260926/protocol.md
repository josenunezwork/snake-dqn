# Apex serving-time free-space veto: development screen (pre-registration)

Status: **non-authoritative development screen.** It has no bridge governance,
no independent audit and no promotion authority. Whatever it finds, the deployed
agent does not change. Its only possible outputs are "worth a strict gate" or
"not advanced". Its data may also serve as an independent pilot for sizing a
later strict run on fresh worlds.

Written 2026-09-26, before any screen episode was run. The only live runs so far
are a 2-episode × 500-frame plumbing smoke (see "Verification so far").

## Why

At the deployment profile, the incumbent Apex/vector61 champion
(`champion_a5_freespace_20260621.pth`, sha256 `43d4e2c5…7ac93`) mostly dies by
trapping itself. This comes from the strict pilot's saved incumbent records
(`task-aligned-strict-challenge-20260926/pilot-v1/output/{development,pilot}/producer/records/incumbent-*.json`,
96 H5000 episodes):

| | Count or value |
|---|---|
| Death cause | self 85, head_on 9, enemy_body 1, wall 1 |
| Alive fraction (frozen / scripted / mixed) | 0.31 / 0.29 / 0.37 |
| Per-world SD of the mass integral | 56 / 88 / 59 |

Apex already observes per-action flood-fill free space. These are the last three
61-D features, from `SnakeStateMixin._get_free_space_features`. Nothing makes it
act on them. Scripted snakes already use `ScriptedSnake._apply_free_space_veto`.
Portfolio review §7 recommendation 2 asks for exactly this comparison: Apex
against Apex plus a capped flood-fill veto, at H5000, across all three mixes,
with no training.

## Intervention (arm B)

`src/evaluation/safety_veto.py`, installed on the hero only, through the opt-in
`rollout(..., hero_safety_veto=True)` hook in `src/scripts/tournament_eval.py`.
Its default is off, and when off the record is byte-identical to before.

1. The policy computes its Q-values, the existing hard action mask and
   `base = argmax(masked Q)`, all unchanged.
2. At the decision it calls `snake._get_free_space_features(other_snakes)`,
   the same call the state builder and the scripted veto make. Then
   `cap = min(160, max(32, 2·length))` and `need = min(logical_length, cap)`.
   Direction `d` is spacious when `round(f[d]·cap) >= need`. These are the
   scripted formulas.
3. Action `a` (0–5; `a % 3` is its direction, `a >= 3` boosts) is eligible when
   it is mask-allowed and its direction is spacious.
   - No eligible action: the choice is unchanged (`fallback_no_spacious`).
   - Base action eligible: it is kept (`kept_base`).
   - Otherwise: the highest-Q eligible action is played, with ties going to
     the lowest index (`vetoes_applied`).

**One-step approximation for boost moves.** A boost moves the head two cells,
but eligibility uses the one-step feature of its direction. The mask already
excludes boosted paths with an immediate collision. The veto does not
flood-fill again from the second cell, and it ignores the length that boosting
burns.

**Scope.** The veto only runs in the greedy Q-value branch. That is always the
branch at evaluation, where epsilon is 0 and the policy exposes `dqn`. The veto
is not trained, and it does not affect opponents.

Each arm-B record carries `probes.safety_veto`:

| Field | What it holds |
|---|---|
| `method` | Veto identity: `free-space-veto/v1` |
| Caps | The BFS caps the veto used |
| `counters` | `decisions`, `kept_base`, `vetoes_applied`, `fallback_no_spacious`, `vetoes_to_boost`, `vetoed_base_boost` |

`decisions` must equal `denominators.decision_frames`. The summary reports how
many episodes satisfy this.

## Arms

| Arm | Hero | Worlds |
|---|---|---|
| A | Apex champion (incumbent) | 40 per mix |
| B | Apex champion + veto | same 40 per mix, paired with A |
| C | Apex champion (determinism control) | first 8 worlds per mix, run **after** all A/B episodes |

Opponents, rosters, profile and seeds are identical between paired A and B
episodes. The two differ only in the veto.

## Profile, config and rosters

- **Profile:** `promotion-v2-watch-rect`, H5000.
  - The digest must equal `d396d3ed…0e8b`, the strict pilot's recorded profile.
  - It is resolved from `research/apex_safety_20260926/deployment.yaml`, a
    byte copy (sha256 `4146baa3…15aa5`) of
    `research/task_aligned_challenger_20260924/deployment.yaml`. The copy is
    needed because that research tree is not on this branch.
- **Rosters:** `src.evaluation.strict_promotion.materialize_rosters` with
  roster width 5.
  - Checkpoint pool in the strict pilot's order: champion, `best_apex_fs`,
    `best_apex_stage1_fs`, `best_apex_pre_fs`.
  - Canonical `greedy_food` / `random_safe` anchors, with the canonical mixed
    rule: odd slots take the checkpoint pool, even slots take `random_safe`.
- **Roster preflight:** before running, the screen rebuilds the strict pilot's
  16 development worlds × 3 mixes and requires every world identity to equal
  the one recorded in the pilot's incumbent records. At preflight today, 48/48
  matched.
- **Checkpoints:** each is sha256-verified against pinned hashes, copied into
  `OUT/checkpoints/`, and verified again. Rollouts load only the copies, whose
  hashes are checked a final time at the end.

## Worlds (new namespace) and disjointness check

- **Recipe:** `seed_i = uint32_be(sha256("apex-safety-screen-v1|worlds|i")[:4])`
  for i = 0..39. The same seeds are used for all three mixes, as in the strict
  design. The rosters differ by mix. A collision inside the namespace stops the
  run with no repair.
- **Disjointness:** checked in code before anything runs (`disjointness_report`)
  and saved in `intent.json`.

| Checked against | Source | Result |
|---|---|---|
| Task-aligned challenger namespaces: development 16, shakedown 4, pilot 16, final 120, serving 50 | Recomputed from the frozen recipe in `research/task_aligned_challenger_20260924/namespaces.py` (domain `task-aligned-challenger-20260924/original6102/v1`) | 0 overlaps |
| World seeds the strict pilot actually played in development and pilot | Read-only from the saved records | 0 overlaps; they equal the recomputed development and pilot namespaces exactly, which proves the recipe |
| Integers 0..999 | `tournament_eval` CLI and test defaults | 0 overlaps |
| Optional JSON int lists | `--exclude-seeds-json` | Checked when given |

**Not checked:** the challenger's `training` and selection world ids. They live
in four admission metadata records outside this lane's read scope. They are
historical ids, not SHA prefixes of this domain. A chance uint32 collision is
about 40·M/2³² for M such ids. Pass them with `--exclude-seeds-json` if exact
certainty is needed.

## Pre-registered analysis

- **Primary:** the paired mass-integral delta (B − A) per mix.
  - Test: one-sided paired t (`eval_stats.paired_delta_test`, H1: mean > 0).
  - Multiplicity: Holm over the 3 mixes at family α = 0.05
    (`eval_stats.holm_three_mix_superiority`, `required_successes=2`).
- **Secondary (descriptive, no multiplicity claim):**
  - paired survival-fraction delta per mix, with an unadjusted one-sided test;
  - death-cause tables per arm and mix;
  - veto counters: totals, veto rate per decision, episodes with any veto, and
    the `decisions == decision_frames` consistency check;
  - win/loss/tie counts.
- **Determinism control:** every C record must equal its A record under
  canonical JSON.
- **Decision (`summary.json` → `decision`):**
  - `INCOMPLETE`: any planned A/B pair or C episode is missing, for example
    after a deadline stop. No scientific decision is made.
  - `INVALID_NONDETERMINISTIC`: complete, but some C ≠ A. The paired design is
    not trustworthy, so no decision is made.
  - `RECOMMEND_STRICT_GATE`: complete, deterministic, and Holm rejects in
    ≥ 2 of 3 mixes.
  - `NOT_ADVANCED`: otherwise. This means "not established", not "no effect".
- **Informational pooled summaries (`summary.json` → `pooled_informational`,
  not part of the decision):** computed with `src/evaluation/screen_stats.py` on
  the B − A mass-integral deltas at 90% confidence, the level of the governance
  clear-loser filter (`docs/research/governance_tiers_2026-09-26.md`):
  - `stratified_mean_of_means`: equal-weight mean of the three per-mix means,
    Welch-Satterthwaite t CI, and whether the pooled or any per-mix upper
    bound is below 0;
  - `random_effects_across_mixes`: DerSimonian–Laird with mixes as strata
    (Hartung–Knapp CI with q floored at 1, τ², I²);
  - `crossed_mixes_by_worlds`: the crossed mixes × worlds grand mean over
    worlds paired in every mix. The mixes share world seeds, so this is the
    only one of the three that models the between-mix covariance.
  If any of the three cannot be computed (too few worlds, zero variance), it
  is reported as `{"available": false, "reason": ...}`. A pooled upper bound
  below 0 would be reported as "harm not excluded at the clear-loser level";
  it does not change the decision above.
- **Informational sizing:** `eval_stats.paired_delta_pilot_size` on the B − A
  deltas, with MDE = 10% of the arm-A mean per mix, the strict calibration's
  fraction. It is labelled as sizing a later strict run on **fresh** worlds.
  None of these 40 worlds may be reused as final worlds.

**Power caveat (stated in advance).** Per-world SDs are about 56–88 and the
paired SD of B − A is unknown. With n = 40 per mix, a one-sided α ≈ 0.017 first
Holm step and 80% power resolve only effects of roughly 3·SD_Δ/√40 ≈ 0.47·SD_Δ.
For SD_Δ ≈ 60 that is about 28 mass, or 40–50% of the arm-A mean. This screen
can flag a large benefit, which is plausible if self-trap deaths are prevented.
It cannot detect a modest one.

## Runtime and resources

- **CPU only:** `SNAKE_DQN_DEVICE=cpu`, torch 2 intra-op / 1 inter-op threads,
  `OMP_NUM_THREADS=2`.
- **Episodes:** 2 arms × 3 mixes × 40 worlds + 3 × 8 controls = **264 H5000
  episodes**.
- **Estimated time:**
  - The strict pilot's development stage measured about 18 s per H5000 Apex
    episode (48 in 853 s). At that rate the screen takes **about 80 min**.
  - Arm B adds one bounded flood fill per decision: 3 BFS of at most 160 cells.
    The 500-frame smoke was too short to measure this, because the snake stays
    short and the fills stay small.
  - Budget 80–100 min and set `--deadline-utc` at least 2 h out.
- **Deadline behaviour:** an episode starts only if the remaining time is at
  least max(45 s, 2 × the mean episode time so far). A stop leaves every
  finished record, writes the summary, and yields `INCOMPLETE`. The run cannot
  be resumed; run again into a new `--out`.

## Outputs (`--out DIR`, must not exist)

The runner refuses any `--out` inside `ongoing-research-20260913`, which it
only reads.

| File | Contents |
|---|---|
| `intent.json` | argv, git state, config and profile pins, pool hashes, snapshots, seeds, disjointness and roster-parity reports, planned episode count |
| `checkpoints/` | Verified checkpoint copies |
| `records/{A,B,C}-{mix}-{seed}.json` | One per episode: arm, mix, world index and seed, roster member hashes, wall seconds, and the full `rollout` record (arm B includes `probes.safety_veto`) |
| `events.jsonl` | One line per finished episode |
| `summary.json` | Everything under "Pre-registered analysis" (including `pooled_informational`), plus wall-clock per arm |

## Command

```
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=2 ./venv/bin/python \
  research/apex_safety_20260926/dev_screen.py \
  --out <new dir outside ongoing-research-20260913> \
  --worlds-per-mix 40 --determinism-worlds 8 \
  --deadline-utc <ISO-8601 with offset, >= now + 2h>
```

The `--smoke-frames N` mode (N ≤ 500, ≤ 2 episodes) is plumbing-only:

- it uses the legacy, profile-free rollout path at a truncated horizon;
- its decision is always `SMOKE_NO_DECISION`.

## Verification so far

- `tests/test_safety_veto.py` covers:
  - the pure veto (it changes the choice only when needed, never picks a masked
    action, and matches the base when all directions are spacious);
  - cap and need parity with `ScriptedSnake`;
  - counters;
  - the AISnake hook, both off and on;
  - rollout default-off byte identity and veto-on counters;
  - the dev-screen decision rule and smoke guard.
- Existing tests pass: `test_tournament_eval`, `test_ai_snake`,
  `test_ai_snake_greedy_rng`, `test_scripted_snake`, `test_free_space`.
- **Preflight:** disjointness passes, and roster parity is 48/48.
- **Smoke:** 1 world × arms A, B × 500 frames on the scripted mix. No veto
  fired: 500/500 `kept_base`, because the snake stayed short. A and B were
  therefore identical, as expected. The profiled H5000 path has not been run
  live by this lane.
