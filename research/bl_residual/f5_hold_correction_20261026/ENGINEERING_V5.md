# F5 HOLD engineering V5 — supervisor output layout

Status: ENGINEERING ONLY; HOLD REMAINS; NO DISPATCH PERFORMED.
Owner request: publish failed preliminary qualification attempt 02, repair the
output-layout mismatch, and prepare a create-only V5 attempt-03 qualification
package without dispatch/Colab. This is an engineering successor, not a claim
that the attempt-01 disposition authorized attempt 03 or that V4 independent
acceptance automatically covers changed source.

The qualified runtime creates `COLAB_CLI_ROOT/output` before launching its child
(`research/colab_cli/runtime.py`, `_supervise`). The step-3 shell passes that exact
root to the driver. V4 rejected it solely because it existed. V5 permits an
absent path or an empty real directory, and refuses any file, symlink (including
a dangling link), hidden entry, child directory or other pre-populated output.
`mkdir(exist_ok=True)` accommodates the supervisor; immutable export APIs and
fresh attempt/ledger identities continue to protect all evidence writes.

Regression tests enter the actual non-dry-run `qualify` path with a supervisor
layout and only the CUDA workload stubbed. Both absent and pre-created empty
output succeed and publish synthetic exports; rerunning against those exports
refuses without changing bytes. Populated directories, files and symlinks refuse
before qualification. Local synthetic PASS records are test fixtures, never
qualification evidence for a production source.

Read-only sweep: `f5_qualification.run` already treats the output root as
supervisor-owned and reserves fresh `qualification`/`paired` producer children.
`f5_colab.sh` passes `output/prepare` and `output/paired` to `f5_job.prepare` and
`fit_pair`; their absent-child checks are correct. Local package builders
(`f5_step3_package`, `f5_timing_recovery_package`, `f5_package`) retain absent
absolute destination requirements. No other F5 runtime driver was found to
reject the supervisor-owned empty root. Supervisor/runtime are unchanged.

V4 seal/package/evidence remain immutable ancestry. V5 binds the V4 engineering
seal, the attempt-02 publication, this correction and the current full source
closure. Package identity, paths and qualification attempt number advance to
V5/03; source hashes/control/empty-ledger bindings follow the new source.
Scientific recipe, fixture identities/allocation, environment contract, input
payloads, thresholds, look discipline, historical maxima, thirteen retained
completions, resume/retry maximum, worker count, archive/shard limits, finite
envelopes, transfer/staging limits and required exports retain V4 meaning.
The preliminary qualification still executes no timing fixture. Measurement
remains HOLD and requires the existing full subsequent admission order.

No CLI transport source, sandbox source, checkpoint or serving pin is changed.
No allocation, dispatch, numerical look or operative fit acceptance is performed
or manufactured. Changed-source genuine CUDA/native-v6 qualification, independent
acceptance and normal account/path admission remain future execution evidence.
