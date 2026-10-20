# T34 v2 teacher

Implementation of `docs/research/specs/2026-10-19_t34_v2_teacher.md`, bound to
`DESIGN_V2_SHADOW.md` (including Amendment 1) and `SUCCESS_CONTRACT.md`.
The original v1 engine remains available for historical replay and tests.

## Implementation and verification intent

Written before this implementation's smoke exports. This task prepares tooling;
its tiny development fixtures are excluded from all campaign estimates. It does
not collect a label round, run efficacy, fit a student, or authorize serving.

- Extend the existing ExIt export/replay and shadow/LAD packages. Use D=300 and
  H=300 by default, fp32, teacher stride-ten probes and stride-one refinement of
  every sampled survive-to-die gap. Bisection is not the teacher label rule.
- Student greedy continuation supplies dense KL-anchor labels. Decisive labels
  and deterministic sparse pre-LAD labels carry explicit roles. Apply beta in
  {1, 0.5, 0} only at scheduled teacher decisions; invalidate after intervention.
- Preserve six-snake actions, RNG stream identities, initial/every-250/final
  hashes, immutable exports, and whole-world replay rejection.
- All implementation verification runs on the Mac. At most two numerical CPU
  threads total; no live Colab, pods, tmux, or other worker operations.
- A1: compare repeated exports in separate spawned processes. A2: run the
  existing replay harness on the local smoke. These are tooling checks, not
  cross-runtime A2 acceptance or candidate efficacy evidence.
- Run each new real benchmark/label entrypoint with two worlds on Mac CPU.
  Record exact commands, elapsed time, work and replay result. L4 and optional
  CPU-runtime manifests are prepared/locally validated only, never dispatched.
- Commit/push coherent steps on the assigned branch; publish to main at the end
  after integration checks. Preserve campaign contracts and existing artifacts.

Acceptance table and exact operator commands will be completed with the final
implementation and measured preflight evidence.
