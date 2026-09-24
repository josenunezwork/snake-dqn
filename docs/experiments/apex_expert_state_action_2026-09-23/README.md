# Apex expert-state action agreement

This closed diagnostic asked whether the saved 10,000-update H64 growth checkpoints
agree with their frozen greedy teacher on the same saved teacher states. It did not
train a model or create new games. The independent saved-record audit passed, and the
frozen decision is `TEACHER_PATH_ACTION_GAP`.

The prospective primary interval is between the first and second food events. Each
of the six seed-by-bank cells had to meet both 95% pooled-frame agreement and 95%
macro-case agreement. None passed either threshold:

| Seed | Known adaptive: pooled / macro | Formerly fresh adaptive: pooled / macro |
|---:|---:|---:|
| 2026099201 | 76.2397% / 78.0897% | 73.3149% / 74.0037% |
| 2026099202 | 68.3884% / 72.9765% | 73.8689% / 75.4424% |
| 2026099203 | 72.0041% / 74.8079% | 73.2225% / 73.1789% |

Before the first food, agreement was 100% in all six cells across 96 frames per cell.
That narrow result does not establish broader competence. The primary interval had
no label conflicts, missing cases, or boost-mask violations. Post-second-food values
are descriptive because the boost/action mask can differ there; see the full
[per-phase table](table.md).

The compared states came from the frozen teacher's own trajectories on the existing
known-adaptive and formerly fresh/adaptive banks. “Fresh” is the historical bank
name, not a new-world confirmation. These measurements describe teacher-label
action agreement, not training-set fit or policy quality. They do not show that a
disagreeing action was harmful, identify a TD-learning or representation cause, or
establish why agreement changes after food. No training, reward change, or policy
promotion follows from this diagnostic.

The query reused saved teacher games and prior Q-cache values. It made 13,541 new
batch-one online forward evaluations (4,189 / 4,922 / 4,430 by seed), reused 4,831
prior cache slots and 60 duplicate query slots, loaded three checkpoints, and ran
zero games and zero updates. The two natural receipts totaled 3.528631 seconds;
peak child RSS was 538,853,376 bytes and minimum available memory was 34,683,027,456
bytes. The independent audit passed after checking 18,432 saved Q slots. The analysis and
audit are complete, and this diagnostic line is closed; any next step requires a
separate design review.

The [earlier second-food continuation learning curve](../apex_second_food_continuation_2026-09-22/td-q-learning-curve.png)
and [representative paths](../apex_second_food_continuation_2026-09-22/representative-fresh-paths.png)
are historical context from that earlier greedy-policy continuation, not results
from this action-agreement diagnostic. No learning curve is applicable here because
this diagnostic performed no updates.

Frozen identities: design SHA-256 `325ca03df0128bbfeed4fa04fbb477ecca9321b8e0f43613035adcc3a0342865`;
intent `e4fd0ea910ef18eb14b69e142735f0bd8447faa2dd5e1a81e61c52858a815984`;
query report `16282cfee55cfd13f245ab27ed1a76892843f4fb626f04ebba42e22c67f27018`;
audit report `d7d5814f6ac08105428ba12ef286778816995b880c602e84343922dca3637388`;
query source `30b8b5b040796eecd824722f51f940329f4987b75436c1bc6fca4113b8316941` and
audit source `054c7c94510b5e18e552131bae21d86a1dddff91759cd14f3dd6ac0302bf6a00`;
input freeze `e227a917f8d7e0bd224eb9757b58a40cecb0f58b848d50f9a57bc9d1bf523da7`;
closeout `43caab3fef89e4b6f0c4bbf83945ed056b02449645a88c115e2acd720506904f`.
