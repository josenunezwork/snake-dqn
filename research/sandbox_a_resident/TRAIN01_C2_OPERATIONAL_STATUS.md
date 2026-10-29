# TRAIN-01 c2 operational status, 2026-10-29

**BLOCKED FOR LIVE DISPATCH.** This status supersedes any operational readiness
inference from TRAIN01_C2_ACCOUNTING.md or a successful candidate dry-run.
The accounting-stop classification and operational dispatch qualification are
separate decisions. No c2 attempt has run or been spent.

The exact r6 host accounting guard was independently **ACCEPTED** by Argon at
`0a0f5faf`. It is an outcome-blind mechanical stop under the existing manual
completion rule. The genuine receipt verifies stop and clean cleanup; all 2072
polls, 2501 log lines and retained exports pass the complete fault scan. The
latest verified snapshot is 0336, not the unexported later iteration 350.
Completion index 2 of 2 would retain endpoint 0384, 48 remaining updates and all
frozen science. Provider-side causation of the rate mismatch remains unknown:
the failed usage/sessions/active-record sample was never persisted.

The first candidate seal (`packet-train01-15-c2-i0336`, runtime `0615e055`, broker
`42b05397`) failed offline validation at the host's 240-second test deadline.
A retained direct reproduction killed `training_preflight` at 239.999986 s,
while its orphan pytest child subsequently finished 1605 tests in 262.61 s.
All those owned validation descendants subsequently exited.

Proposed correction `dacd9191` increased only host windows (420 s inner, 900 s
outer, 960 s builder). The fresh `packet-train01-15-c2-i0336-r1` (runtime
`9591b2ef`, broker `c2105288`) passed committed packet validation, CUDA contract
mocks and `PASS_DRY_RUN ... allocations=0 live_accounting=NOT_CHECKED`.
Its two complete resident test passes were retained. **That offline pass is not
live admission:** Argon independently **REJECTED** the host-window correction.

The live wrapper still limits `OPS_ALLOC_LOCK_TIMEOUT_SECONDS` to 600 s and
shares that deadline across `live.validate`, launch and calibration wait. One
validation subprocess runs two full admission passes; the detached native `run`
repeats both before allocation. The new observed test cost cannot safely fit
those phases inside the existing lock window. If a live validation barely fits,
the detached run can outlive the remaining wait and allocate after the parent
has released the serialization lock. Increasing only validation timeouts does
not solve that defect. Validation subprocess timeouts also fail to reap their
whole descendant tree; the reproduction demonstrates that leak.

The rejected timeout increase was **reverted** in `16aab753`. The remaining
implementation is byte-identical to the accepted `0a0f5faf` code, with this
operational status added. Qualified `research/colab_cli`, training kernels,
source recipe, account guards, live allocation lock, caps and serving state are
unchanged. Candidate seals and their unrun DISPATCH.txt files remain immutable
as **unqualified evidence**. Do not execute either candidate's live command.

The correct next step is a separately bounded ops repair that accounts for
host admission before allocation, preserves lock ownership through verified
rate calibration, propagates finite deadlines through every validation layer,
and reaps only owned validation process trees on timeout. Test the actual live
wrapper path with the offline provider, including slow admission, lock hold,
late-allocation refusal, native-run admission and cleanup; obtain independent
Argon/Astra acceptance. Then build a fresh c2 seal from the same 0336 snapshot,
with completion index 2 and endpoint 0384 unchanged. This is not permission for
a c3, an extra training run, a different seed, an intermediate selection or a
scientific look. Main's H100 fallback deadline repair alone does not qualify
this separate ordinary TRAIN-01 path.

Keep A100 for that future seal: frozen sandbox preflight requires A100 and the
runtime fixes TORCH_CUDA_ARCH_LIST=8.0. H100 preference support does not replace
the required runtime/admission qualification. No RunPod, provider/account call,
allocation, stop, dispatch, push, released model or serving change was performed.

Evidence directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-c2/`.
Read SUMMARY.md, ARGON_REVIEW_3.txt (admission ACCEPT),
ARGON_HOST_BUDGET_REVIEW.txt (host correction REJECT),
FINAL_DRY_RUN_VALIDATOR.log, BUILD_RESULT_FINAL.json and
SOURCE_UNCHANGED_AND_CLEANUP.json. The prepared candidate and the final branch
have different source pins because the rejected host correction was reverted.
