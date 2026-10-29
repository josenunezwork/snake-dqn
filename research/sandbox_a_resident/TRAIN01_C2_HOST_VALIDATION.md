# TRAIN-01 c2 offline validation deadline correction

The first c2 seal (implementation `0a0f5faf`, runtime `0615e055`, broker
`42b05397`, packet `packet-train01-15-c2-i0336`) was never dispatched. Its
offline dry-run failed. A retained direct reproduction shows the exact cause:
the host `training_preflight` command is killed at 239.99998633295763 seconds.
Its already-started pytest child subsequently completes **1605 passed in
262.61 seconds**. The prior packet identity and bytes remain preserved as an
unqualified offline build.

The accepted readiness-6-era resident suite has grown beyond the host admission
window. Full snapshot/receipt validation also precedes that suite. Increase only
the bounded offline sandbox-admission subprocess window from 240 to 420 seconds.
The existing outer formula still covers both independent admission passes plus
60 seconds of ordinary host overhead: 900 seconds. The builder waits at most
960 seconds for that outer dry-run. Ordinary subprocesses remain at 60 seconds;
an earlier caller/account-lock deadline still clips either validation bound.
No test, validation step, identity check, lease, Colab job timeout, budget,
registered scientific endpoint or retry limit is removed or changed.

Regression tests simulate a 300-second inner validation, a 650-second outer
validation and an 850-second builder wait without sleeping. They also exercise
an overlong/hung validation, the ordinary 60-second guard and a shorter lock
deadline. The final fresh seal uses suffix `-r1`, still c2 index 2 of 2, with
the same iteration-0336 source and frozen A100 recipe. No code or fallback change
from main's GPU-preference repair is imported; qualified `research/colab_cli`
remains byte-identical. Evidence and the exact unrun live command are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-c2/`.
