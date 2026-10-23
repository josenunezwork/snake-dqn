# E8_LATE_BODY_AMENDMENT_04 — stable solve and fresh collection

Status: **RATIFY-WITH-CHANGES / CONDITIONAL IMPLEMENTATION**, 2026-10-22 PT.
Authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-18/JUDGMENT.md`,
sections 3–7. Effective only when independent Codex-V ticks every section 4
acceptance line and Claude verifies the receipt. This sibling supersedes only
amendment 03 section 2 step 2 and section 4, clarifies its section 2 step 3
actor-phase moments, and adds the collection allocation below. The FIT-only
replay acceptance below replaces amendment 03 section 5 reproduction criteria
for this amendment's replay; no historical receipt or gate is changed.
Amendments 01/02/03, erratum 03, resume amendment 05 and all prior evidence
remain byte-identical. No amendment-04 Colab data exist at freezing.

## 1. C1: deterministic zero-sum solve

Use all natural completed FIT reservoir rows in their existing order, never
held-out, supplement, fixture or unfinished-life rows. Let z be the frozen
256-dimensional LayerNorm core and h the existing clock. Use the explicit
float64 Helmert basis Q (256 by 255): for zero-based column j, entries 0..j
are `1/sqrt((j+1)*(j+2))`, entry j+1 is `-(j+1)/sqrt((j+1)*(j+2))`, and later
entries are zero. Thus `Q.T Q=I` and `Q.T 1=0`, deterministically without QR
sign ambiguity. Basis identifier: `helmert-zero-sum-256-v1`.

Before solving, require the core LayerNorm affine weight exactly 1 and bias
exactly 0; otherwise raise ValueError through the finite refusal path. Require
`max(abs(mean(z, axis=1))) <= 1e-6` on FIT features, otherwise refuse. Existing
nonfinite tripwire precedence remains. Solve once
`lstsq([1,z@Q,h], y, rcond=None)` in CPU float64, one numerical thread. Map
`c_core=Q@c_q` to the existing head. No ridge, rcond tuning, row reweighting,
feature or gate change. Rank deficiency uses the same SVD minimum-norm rule.
Moments from the same FIT targets, stored-fp32 coordinate mapping, installation
in the existing fp32 head and reset of only the two head Adam entries remain
unchanged. Report rank, condition number, core coefficient norm and sum,
`common_mode_sensitivity_per_1e-4_bias=1e-4*sum(c_core)` (absolute magnitude
must be <=1 return unit), basis, FIT count and fit errors.

Exactly one amended replay of final code uses the verified source-05 exports.
Acceptance is FIT-side only: rank=257; raw FIT MSE within 1e-4 relative of
896454.557, or <= old `[1,z,h]` FIT MSE plus 1e-4 relative; maximum absolute
installed prediction difference from the old installed head on FIT rows <=20
return units; absolute common-mode sensitivity <=1.0. Tests cover synthetic
near-zero-sum features, fp32 installation identity, rank deficiency,
repeat-install and restore-then-install identity. The D1 held-out rows and all
derived numbers are consumed development evidence. Held-out numbers may be
recorded for the ledger but never select a solver, tolerance, threshold or
feature, and never enter this replay's acceptance.

## 2. C2: zero-unit Mac first-step fixture and moments clarification

Install the C1 head on a small synthetic fixture, enable the actor, and run
three real PPO `update()` calls on synthetic rollouts with core, encoder and
residual trainable, global gradient clip 1.0 and core learning rate 1e-4.
All values, losses and gradients must be finite. On a fixed held feature
batch require, for each k, `max(abs(V_after_step_k - V_before))` <= the
installed PopArt scale. A uniform 1e-4 shift of `core.norm.bias` must change
V by <=1.0 return unit, including the analytic coefficient-sum check.
This is light Mac CPU software evidence at 0 Colab units, not a policy assay.

Correction to amendment 03 section 2 step 3: moments are frozen at the FIT
moments for the actor phase because `update_targets` runs only while
`actor_enabled` is false. No adaptive actor-phase moments are introduced.
The existing output-preserving PopArt semantics outside that phase remain.

## 3. C3: fresh natural collection allocation

Control flag `e8_critic_convergence_amendment_04=true` and its `_sha256`
bind this amendment. Enabling both 03 and 04 is refused. Required integer
`natural_collection_world_id_offset` is a positive multiple of 5, disjoint
from supplement/fixed-holdout world ranges. Natural CollectorConfig receives
that offset. Run/snapshot identity binds flag, SHA and offset; initializer
seed 20261021 and model/torch RNG remain unchanged. The raw offset world ID
is the FIT/held-out split key (`world_id % 5`); modulo classes are preserved.

Exclude every accessible burned natural root of sources 02/03/04/05 from
snapshot `refit_seen_seeds` / training-status and D1 chunk world IDs re-derived
through `_world_seed`, all 384 evaluation seeds, fixed-holdout and supplement
roots, and ordinary-bank seeds. Never open the protected study. Publish
`collection-allocation-audit.json` plus SHA and completion sidecars before
dispatch, recording source/hash coverage and any inaccessible history per
REGISTRATION_P1 section "Fresh paired evaluation and one-look decision".
Audit both new root sets for 128 worlds over episodes 0..E, E>=the count
covering the 4M prefix at observed approximately 3–5 lives/world; freeze E
and offsets in that artifact. Canary and source offsets must differ; audit
cross-lane collision freedom as well as zero burned-set intersections.
Any collision rejects the whole allocation and advances the offset by a
fixed step; individual worlds are never replaced. The collector's excluded
seed attempt loop also guards later roots beyond the finite audit range.

## 4. Frozen pricing and operational disposition

| Item | Estimate minutes | Base / effective timeout minutes | Base / effective cap units | Base / effective included reserve units |
| --- | ---: | ---: | ---: | ---: |
| Exact-pin family canary | 15 | 15 / 30 | 1.00 / 2.00 | .20 / .40 |
| Fresh Y3 source | 75 | 150 / 300 | 4.81 / 9.62 | .96 / 1.92 |

L4 only, `cuda:0`, rate ceiling 1.54u/h, tolerance 0.0. First wall/unit/
transition cap wins. The owner's 2x rule is already applied once; estimates,
rate, N and cadence are not doubled. Legacy conservative debit is max(own,
gross), never a refund. Frozen envelope 25.50u, charged 7.26u, new liability
11.62u gives 18.88u; reserve the unadmitted 3.86u replacement, leaving 2.76u.
Replacement admission requires actual residual reconciliation. The 1M
qualification needs separate judge pricing after a genuine Y3 PASS.

Canary must support the registered source estimate/caps and match proof ==
current == manifest executable identity; otherwise no source. Rate >1.54u/h
or any other assignment at admission means HOLD. Timeout/cap is INCOMPLETE,
no completion attempt. Nonfinite retains KILL precedence. Scientific support
or accuracy refusal is final NOT_READY for this recipe, without refit,
alternate head, checkpoint hunting or extension. One outcome-blind mechanical
SUPERVISOR/transport retry is preauthorized only after stop_verified=true,
no gate outcome read, and actual charged-to-date +9.62+3.86 <=25.50u;
a second mechanical failure means STOP and judge review. The declaration's
two-retry maximum is not authority to dispatch a second completion attempt.

## 5. Unchanged scientific scope and preconditions

**A critic PASS under this amendment establishes initialization adequacy of
the value head only; it funds nothing beyond the registered Y3 replacement
and the separately priced 1M qualification, and establishes no late-body
policy improvement.**

The gate remains EV>=.10 and MSE strictly below the float64-SVD `(1,h,L)`
comparator on natural completed held-out L>=800 rows; whole-world modulo-5
split, support minima, 4M fit-world cap, one support continuation, one look,
refusal/nonfinite precedence stay unchanged. Actor/env/reward/features,
opponents/PPO parameters, initializer seed, 100M endpoint and timing windows
stay unchanged. B8 s3 remains served and release requires the owner.

Judgment section 6 remains binding in full: merge code and obtain the
line-by-line independent Codex-V receipt and Claude campaign-log check;
publish replay and allocation audit; verify qualified Colab CLI tree
`553af3f692515b2bf66173356fe4d099acc96942`; fresh source-latebody-06 identity,
empty or verified uploaded ledger and independent pack review; immediately
before dispatch verify live usage/sessions with zero other assignments and
rate<=1.54u/h, current debit and carried WARNING_UNEXPLAINED_DEBIT. A literal
empty mapping is not an inventory. Stay outside X5 Thu20:00–22:00 PT and K-1
Mac windows, never reserve an idle Colab lane. Claude alone dispatches through
`tools/ops/dispatch.sh`; Codex prepares/tests, never dispatches. Protected
`t34-round0-study-01/` stays closed and `../snake-dqn-l2mac` untouched.

Any C1–C3 acceptance failure, held-out-driven variant selection, change beyond
these items or edit to amendment03/its receipts means no spend and return to
a judge with the receipt. Preparation costs 0u, light Mac CPU only.
