# perf-sim tooling (2026-10-07)

Profiling, byte-identity and throughput tooling for the `perf-sim` branch (CPU-throughput
optimizations of the live simulator; plan and results summary in
`docs/research/perf_plan_2026-10-07.md`). Nothing here runs a study or touches RunPod.

| File | What it does |
|---|---|
| `episode_harness.py` | Plays strict-gate episodes (frp3-s12 + v8 hero, deployment config, `promotion-v2-watch-rect` H5000, `dev_screen._design_rows` rosters, own seed namespace `perf-sim-identity/v1`) from any source tree (`--root`). `identity` mode writes per-frame world digests (all snakes, food, frame, Python + NumPy RNG state), a digest of every `ApexNetwork.forward` input/output (float32 bytes), the canonical record, the veto diagnostics and a non-int-coordinate counter; `time` mode times (optionally under cProfile). |
| `actor_harness.py` | The real `ApexActor` in-process with a digesting buffer client: `--world gate` (FRP-v3 gate-world actor; needs a tree with `src/training/gate_world.py`, i.e. frp-v3/frp-v4) or `--world train` (the default actor arena, any tree). Digests every row sent to the buffer. |
| `compare_identity.py` | Reference vs candidate comparison of the above (timing fields stripped). Exit 0 only if identical. |
| `analyze_profile.py` | cProfile dump -> % self time by component. |
| `ab_bench.py` | Paired A/B throughput: both trees play the same episodes at the same time (one slot each). |
| `with_slots.py` | Runs a command while holding shared CPU slot locks; refuses on battery / lid closed. |
| `pod_bench.py` | PREPARED, NOT LAUNCHED: one-pod sweep of concurrent single-thread episodes (1 / cores / vCPUs / 1.5x) for the SMT and cpu3c-vs-cpu5c questions. |
| `results/` | Identity comparisons and A/B summaries committed as evidence. |

## Identity evidence recorded on 2026-10-06/07 (Mac, AC)

References: `git archive db2ef7a` (main) for eval and train-world actors, `git archive 41a1608`
(frp-v3, the FRP-v3 gate-world actor code) for the gate actor. Candidates: the perf-sim
commits, and for the gate actor the frp-v3 tree with the perf-sim `src/` diff applied
(`patch -p1`; the only conflict-free overlap is `src/game/ai_snake.py`, which applied with an
offset). See `results/identity_*.json`.

Re-run (two slots):

```bash
S=/tmp/perf; mkdir -p $S/ref_main && git archive db2ef7a | tar -x -C $S/ref_main
H=research/perf_sim_20261007
./venv/bin/python $H/with_slots.py --slots 1 -- ./venv/bin/python $H/episode_harness.py \
    --root $S/ref_main --mode identity --out $S/eval-main.jsonl --worlds 10
./venv/bin/python $H/with_slots.py --slots 1 -- ./venv/bin/python $H/episode_harness.py \
    --root . --mode identity --out $S/eval-perf.jsonl --worlds 10
./venv/bin/python $H/compare_identity.py $S/eval-main.jsonl=$S/eval-perf.jsonl
```
