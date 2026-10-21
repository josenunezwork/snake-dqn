# E3 information-value development experiment

Frozen before E3 reconstruction, targets, fits or outcome inspection. Authorized by
the owner's E3 implementation request; authority: big-lever round-01/JUDGMENT.md
section (f) and CHARTER.md. This is development evidence, excluded from EVERY
registered T34 verdict, strict gate, serving qualification and release decision.

## Question and data

Does full-resolution geometry, then history, improve action-ranking projection
of D300 mass returns compared with B8 inputs at equal trainable parameter count?
Only the unsealed T3 rollout-label bank may supply worlds. Never access the
sealed t34-round0-study-01 artifact directory. Preserve all existing source.

Use at most 256 worlds and 1,536 unique source-world/decision roots, half saved
decisive and half saved counterweight. Select worlds and roots deterministically
from source identities, before producing mass targets; sort world identities by
SHA256('bl-e3/v1|' + source key), then source key. Select the first 256 eligible
worlds, each with both row classes and reconstructible 128-decision history.
World-disjoint split: first 192 train, next 32 validation, last 32 held-out test.
Within each split, select 3 roots per class per world in decision order; if a
world lacks 3 of a class, fill that class quota from remaining roots in the same
split in world-order/decision-order. Never replace a world after replay failure.
Insufficient worlds/roots or unreconstructible history => INCOMPLETE. Tiny
preflight uses the same pipeline with explicit smoke-only sizes and cannot emit
a scientific GO/KILL. It is excluded from every experiment estimate.

Replay the recorded hero checkpoint and roster with the T3 prepared selector
and runtime hooks. Verify saved local/global/scalar/mask/action/frame at every
selected root. Record 128 preceding decisions (not future or current action),
all six seats' head position, heading, logical mass, alive state and executed
action, plus a history-valid flag; zero-pad only before world start. Verify
recorded death endpoint when traversed. T3's original per-root decisive/counter
classification is retained; never select by new mass returns.

At each root branch every resolved legal first action, then follow the original
B8 checkpoint and opponent policies reactively, as in the recorded T3
continuation operator (not an open-loop tape after intervention). D300 return
is sum of post-transition living hero logical mass / 300; dead remainder = 0,
no discount, no terminal bootstrap. Target action regret is max legal return
minus action return. Recompute mass returns: survival labels/final mass are not
substitutes. Verify the unchanged B8 branch against the actual replay mass
prefix, including zero-dead remainder. Illegal actions never train or select.

## Models and fitting

A receives only original B8 local/global rasters and 60 scalars. B adds the full
145x83 board, separate owner-relative occupancy planes (hero and each opponent)
and unclipped segment release order (tail-relative steps; no 255 clipping).
C receives B plus the 128-decision hero/opponent history through a 64-unit GRU.
All three have exactly equal trainable parameter counts; matching capacity must
participate in forward computation, never unused padding parameters. Report counts.
New ranking students, same seed 20261021, AdamW lr=0.0003, weight decay=0.0001,
batch 32, 60 epochs, pairwise softplus ranking loss weighted by absolute D300
return difference / 1000, plus 0.1 smooth-L1 to negative regret / 1000.
Use validation decisive mean regret to select epoch (earliest tie); evaluate
held-out test once, after all three fits. Normalization is fixed by physical
units, never fitted on validation/test. No sweep, replacement seed or extension.
Choose lowest-index action on tied student scores; mask resolved illegal actions.

## Frozen decision rule (precedence)

1. INCOMPLETE: unavailable histories, identity/control/parity failure, nonfinite
   training/metrics, missing rows/classes/splits, cap expiry or incomplete exports.
2. Otherwise GO only if B or C reduces held-out decisive mean regret >=20%
   relative to A (A regret must be positive), paired whole-world one-sided
   Student-t LB90 of A-minus-candidate decisive mean regret >0, and unnecessary
   switches <=1% of held-out counterweight roots. An unnecessary switch changes
   recorded B8 action to an action with strictly greater D300 regret (>1e-9).
   Average roots within world, then worlds equally for decisive comparisons;
   compute LB90 with df = number of paired test worlds minus one. Report each arm.
3. Every other complete valid result is KILL. Keep history only if C also meets
   GO and reduces B's world-equal decisive regret >=10% (B regret >0).
   If B and C qualify but history fails this rule, select B; if only C qualifies
   but history fails, report GO for representation, history NOT retained and no
   deployable architecture selected. No outcome authorizes release.

## Execution and limits

Mac Phase 0: real entrypoint on tiny source input, nice -n 10, <=2 total math
threads, CPU; time reconstruction, branches and fits; report extrapolation and
its limits on the contended Mac. Deterministic process pool, fixed-order merge,
one math thread per process; prove serial/parallel arrays equal. On Colab use
L4 CPU processes for reconstruction and L4 CUDA for fits, qualified
research/colab_cli only; no source edits there. Overall cap 16 units including
setup/export/failures, 48h calendar, expected ~3h reconstruction +4h fits.
Reserve 3.2 units (20%); first wall/unit cap wins. At conservative 2 u/h only
6.4 allocated hours are usable, so an over-budget preflight forecast MUST block
full dispatch rather than silently shrink N, D300, history or fit recipe.
All exports: immutable payload, '<payload>.sha256' and '<payload>.complete'.
Manifest commit and input hashes are locally validated; Claude dispatches.
No Colab calls, retries, purchases or automatic dispatch in this task.
