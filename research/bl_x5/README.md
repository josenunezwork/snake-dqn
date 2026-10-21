# X5 bare Mac timing

This development adapter reuses T7's `run_simd_eval` call geometry: GridBatchSim,
full H5000, eight worlds per unit, CPU, Python vector61 backend, rowwise forwards,
advisory ego2s mask, and bare B8 s3. It does not call the registered T7 coordinator,
analyze efficacy, open a strict family, or alter serving.

The original X5 specification is in the big-lever round-07 judgment, lines 174–177;
PLAN_v2 §2 schedules it and §8 requires the idle receipt and retained stopped
attempts. The owner explicitly requested immediate execution instead of the night
window. Forty-eight globally distinct development worlds (16 per mix) are evaluated
twice with B8 because no F parent was supplied. The two repeats share worlds;
they are 96 episodes, not 96 independent worlds.

Each of three single-thread workers holds one shared development CPU slot through
the coordinator. The coordinator keeps all locks until its children exit. AC,
open lid, thermal state, load, foreign process CPU, and nice level are retained.
A foreign Python CPU lane at ≥25% blocks admission; samples during evaluation
repeat at 30-second intervals. This is an operational guard, not proof that no
short background activity occurred. A stop file or cap raises at a frame boundary;
the code sends no signals. Helpers have no timeout that could kill a process.
Pathological hung native code or OS helpers therefore remain a deadline risk.

Evaluation has a 7,000-second wall cutoff and each worker a 2,300-second process
CPU cutoff, leaving cleanup reserves under the two-hour envelope. There is no
scientific kill rule or automatic retry. Every distinct attempt needs its own
root, allocation key, freshness audit, idle receipt, and manifest.

JSON evidence is create-only with `<name>.json.sha256` and
`<name>.json.complete`. Unit hashes, timings, frozen world coverage and record
identities are re-read before `MEASURED` is emitted. Forecasts separately label
measured process CPU costs, measured wall rates, startup overhead and idealized
worker scaling. The plan sizes are 192 scripted / 96 frozen / 96 mixed and
384 / 128 / 128, each as a single bare arm and as two paired arms. The E13 row
prices 768 episodes (128 per mix × two arms). This does not set strict N, price
an unknown replacement model, controls, audits, product work, or serving.

There is no upload or Colab manifest. Z-9 is `NOT_APPLICABLE_MAC_ONLY`, never a
transport PASS. Any future Colab work needs separate supervisor transport proof.

## Execution and retained interruption, 2026-10-21

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/week1-20261021/mac`.

The first attempt used commit `b68c805940da1aa3e8971d33c33a28800b489aa4`.
It ran from 14:50:44 PT for 55.1244 seconds, then cooperatively stopped after the
independent reviewer identified overlap with the separately dispatched K-1 audit.
All three workers exited 2; no complete unit survived, so no rate is available.
The canonical `x5.json` remains `INCOMPLETE`; it must never be overwritten.

The explicit foreign-CPU guard was added at
`709a673609bec1ce1e8b06d5bf2f4f7e57f5fc84`. A separately prepared
`x5-recovery-01` allocation passed freshness but was refused at launch at
14:52:35 PT due to another CPU lane. It started no children and no episodes.
Its terminal coordinator and admission-refusal receipts are retained. Preparation
of `x5-recovery-02` was also refused before worlds were allocated when tests
resumed. That directory contains no frozen attempt or gameplay.

The run is `NOT_READY(exclusive Mac window)`. The active K-1 full identity audit
holds a development CPU slot; its forecast is 5,400–14,400 seconds. These are K-1
planning inputs, not X5 measurements. No bank CPU-hour forecast can be supplied
until X5 completes in the required exclusive window. A later successful receipt
must use a fresh root and a sibling disposition linking these retained attempts.

After the coordinator has reserved the Mac, use the commands in the artifact
root's `DISPATCH.txt`. `check` is read-only, `prepare` requires a clean committed
checkout and a passing idle/freshness check, and `run` refuses an existing start.
Wrap `run` in `caffeinate -dims` in a unique detached session with create-only logs.
To stop an owned attempt without a signal, create its `STOP` file. Do not run
targeted tests alongside a timing attempt.
