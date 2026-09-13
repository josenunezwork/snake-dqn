# Solo food-signal diagnostic — 2026-09-13

The three ambient-reward endpoints showed a working immediate food target on the
controlled Watch transitions, but none met the full frozen teacher screen on
its held-out placement set. Seed 1903 passed its teacher-train split only. This
is a completed diagnostic, **not** evidence of reliable solo learning and not a
promotion result. The independent raw-artifact audit passed after checking the target arithmetic,
teacher scores, optimizer receipts, and input/output identities.

The diagnostic followed the [three-seed budget study](../solo_budget_curve_2026-09-13/README.md), whose ambient recipe produced uneven food outcomes. It asks a
narrower question: do those final checkpoints receive a positive native TD
signal for controlled adjacent food, and can a restored local optimizer fit a
small, direction-specific target? It does not measure behavior in a new world.

## Frozen inputs and method

The three inputs were the ambient 200k endpoint checkpoints for training seeds
2026091901, 2026091902, and 2026091903, frozen under source revision
`56e0e92434ae510a84ebaf6f58cf41c3e4421b04`. The complete input identity is
[diagnosis-inputs-v2.json](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/diagnosis-inputs-v2.json), SHA-256
`ac702b1c74b6c8141f84a6e2a1165c66cf305813dad74317eec38e7b831de363`.
The diagnostic driver is [diagnose.py](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/diagnose.py), SHA-256
`d389be8f7dd792fc6b5af35f6cdc469638a5a3df41b84cf1231ed1e26fb219c6`.

The fixed bank used 24 native Watch food transitions: three normal-action food
placements (left, straight, right) from each of eight physical base states.
Each had an action-matched no-food control. Four bases formed the teacher-train
split and four were held out, yielding 12 placements in each split. The bank
was selected without checkpoint-dependent filtering. Two candidate states were
retained as exclusions because food was already adjacent; they were not
substituted silently.

For each endpoint, the native one-step target routine evaluated all 24 food
transitions. The teacher then restored two identical network-and-Adam clones:
an anchor and a directional clone. Both were anchored to the initial Q values;
only the directional clone received a 0.5 increment for the normal action that
pointed to food. Each clone took 32 full-batch updates on 72 selected-action rows expanded from
the 12 training observations. Forward computation padded these to 256 rows;
padding was discarded before loss.
Across three endpoints this was 192 clone-only optimizer updates; no clone
weights were saved or used as a policy.

`D = Q_directional − Q_anchor` was used to decode the intended direction and
calculate its margin over the best other normal action.
The frozen response-loss criterion was separately calculated against
`Q0 + bump` (0.5 only on the food-matching normal action), not against `D`: `0.5² / 18 − 1e-6 = 0.013887888888888889`.
It is a placement-independent additive-response benchmark relative to `Q0`,
not an estimate of game return. The threshold follows SmoothL1's quadratic
region; the two-clone setup preserves Adam's restored moment state.
[PyTorch Adam documentation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html)
and [PyTorch SmoothL1Loss documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html)
provide the implementation context.

## Results

All 72 native TD residuals (24 per endpoint) were positive and had absolute
value below 1, within the quadratic region of SmoothL1 with beta 1. That verifies reward-to-target plumbing for these controlled
Watch transitions. It does **not** establish the sign, distribution, or use of
TD residuals in the actual training stream.

Every split had a positive mean directional margin and a response loss below
the frozen bound. The full screen additionally required all 12 placements to
decode correctly. Only seed 1903's teacher-train split passed; none of the
three held-out splits did.

| Endpoint | Native TD positive | Teacher train: decoded / 12 | Train mean `D` margin | Train loss | Held out: decoded / 12 | Held-out mean `D` margin | Held-out loss | Frozen split result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1901 | 24 / 24 | 10 | 0.01314 | 0.013036 | 7 | 0.00691 | 0.013209 | fail / fail |
| 1902 | 24 / 24 | 10 | 0.01392 | 0.012625 | 9 | 0.00970 | 0.012801 | fail / fail |
| 1903 | 24 / 24 | 12 | 0.02956 | 0.012031 | 11 | 0.01903 | 0.012632 | pass / fail |

The [result files](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal) retain per-row margins, targets, losses, clone hashes, and 64 optimizer-step receipts per endpoint. The
[plot PNG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/plot/solo-food-signal.png)
and [SVG](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/plot/solo-food-signal.svg)
passed visual review: all nine panels, labels, legends, threshold lines, and
final directional markers were legible. Native-TD vertical scales differ by
endpoint, so their heights are not directly comparable.

## Qualification and execution evidence

The final diagnostic qualification ran 18 tests in 0.57 seconds with the
project venv and an import witness for the frozen PQN trainer. Its
[result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/qualification-diagnose-v2/result.json)
records the exact interpreter, imported paths, and thread counts.

Two earlier failures remain preserved. The original qualification produced
`2 failed, 16 passed` because its native-target fixtures omitted required
`num_envs` and `num_snakes`; the routine rejected the malformed fixture during the unit tests. The first freeze was also rejected by schema validation before any
Torch work. The versioned v2 fixture repair supplied the required `E=1` and
`S=1` fields; it changed neither the diagnostic driver nor the experimental
inputs.

The three diagnosis jobs took 42.379 seconds in total. Their observed peak RSS
was 725,991,424 bytes and the lowest available host memory was
27,801,468,928 bytes. Each used the local resource limits recorded in its
[supervisor receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/supervisor-runs): two CPU threads, one Torch interop thread, a 4 GiB process RSS ceiling, and a
12 GiB available-memory floor. The MPS limit was unused. These jobs performed
clone-only optimization; they added no reinforcement-learning steps or full
evaluation episodes.

The [independent final audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/audit-diagnoses-v2-result.json)
passed with status `PASS_INTERNAL_CONSISTENCY`. It independently recomputed
native targets, TD residuals, SmoothL1 values, teacher losses, direction decoding,
controlled/raw margins and drift from the saved NPZ arrays. It also checked all
192 optimizer receipts, expanded-row relationships, source/input/output hashes,
and supervisor completion. An external before/after comparison confirmed all
235 frozen audit inputs unchanged.

The original auditor incorrectly compared the raw world-bank seed column with
the checkpoint's training seed. That failed assertion and its receipt are
retained. Version 2 removed that invalid comparison while preserving the exact
raw-to-frozen-bank seed checks; no study output changed.

## Interpretation and boundary

The controlled food reward reached the native target calculation in all three
endpoints, and each endpoint could produce a positive average local directional
response. That is useful fault isolation: it rules out a completely absent
controlled food reward path for these inputs. The missing held-out 12/12 decode
screens prevent a claim that the learned representation reliably generalizes
even across this small placement bank.

This was a post-training, fixed-bank probe of existing checkpoints. It used fresh controlled reset states, but did not run full behavioral
evaluation episodes, estimate population uncertainty, select a checkpoint, or
run the shared tournament promotion gate. Its outcome therefore does not
justify replacing the incumbent. The independent audit confirms internal consistency of the saved evidence;
it does not independently rerun checkpoint inference or optimization.

## Evidence

- [Input freeze and source/driver hashes](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/diagnosis-inputs-v2.json)
- [Endpoint 1901 result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/seed1901/result.json), [1902 result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/seed1902/result.json), and [1903 result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/seed1903/result.json)
- [Plot summary](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/plot/summary.json), [plot closure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/plot/closure.json), and [visual review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/plot-visual-review.json)
- [Qualification result](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/qualification-diagnose-v2/result.json), [retained original failure](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/qualification-diagnose-failure.json), and [supervisor receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-signal/supervisor-runs)

The next diagnostic uses all six historical training telemetry streams to
check whether ambient-food exposure declined or persisted as the policies
diverged. It requires no new environment or optimizer steps.
