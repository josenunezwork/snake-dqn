# Prospective served-tick host-load measurement amendment

Owner authorized in the 2026-10-27 task: distinguish routine macOS/desktop
background load from competing agent/repository work, retain CPU covariates,
report overall and low-background timing, keep behind flags, nice 15/threads 1,
and do not run the long panel in this implementation task.

This engineering diagnostic amendment applies only to **new attempts explicitly
launched with `--classified-host-load`**. The default remains the original host
guard. Existing attempts, exports, manifest, panel order, seeds, exposure, timing
bars, parity checks and release authority are unchanged. Prior aborted attempts
remain contaminated; this amendment does not rescore or recover their cells.
It supersedes only the host-load measurement paragraphs of
`SERVED_TICK_README.md` for opted-in attempts. No serving code imports these tools.

## Classification and stopping rules

1. Exact probe owner and its descendants are the intended workload, excluded
   from foreign load. Sampled transitive ancestry handles short-lived helpers;
   an unrelated `ps` or shell receives no blanket exemption.
2. Foreign Python (including system Python), Codex CLI/exec, agy, Claude CLI,
   node, pytest, uv and executables in project/research paths are workloads.
   Python training/tests and Colab supervisors therefore remain foreign work.
3. Executables under `/System/`, `/usr/libexec/`, `/usr/sbin/`, `/sbin/` are
   macOS services, after the workload rule. Installed `.app` Chromium/Electron
   executables ending `(Renderer)` or `(GPU)` (including `Codex (Renderer)` and
   Chromium's `Helper (Renderer)`) under `/Applications/` or
   a user's `Applications/` are desktop background. A whole desktop app or
   arbitrary argv text never establishes an exemption. Other executables are
   unknown. Traversal paths receive no background classification.
4. Workload or unknown processes at **>=20% of one core per process** abort the
   entire attempt, both in historical point screens and interval CPU deltas.
   No process is stopped, reniced, or disabled to manufacture quiet conditions.
5. Sum macOS service and renderer CPU across all PIDs. Background load alone
   aborts only at **strictly >150% of one core for strictly >60 seconds** in
   consecutive observed intervals. Exactly 150% or exactly 60 seconds does not
   cross this ceiling. A sub-ceiling interval resets the duration. A missing
   interval or interval longer than 5 seconds cannot establish continuity.
   Churn cannot erase measured high background load or reset the ceiling.

Power, open-lid, thermal, lease, heartbeat, source and lifetime guards still
apply. The policy is recorded in `raw.json` and `host.json`; the reducer refuses
a mismatch. CPU and report source plus this amendment are included in the source
binding. Reduction verifies the original host export triplet.

## CPU evidence and low-background windows

The independent monitor takes cumulative user+system CPU counters from macOS
`ps TIME` about every **1 second**, separately from full host snapshots about
every 10 seconds. TIME resolves to 10ms; `ps %cpu` is retained only as historical
screen evidence and a conservative foreign-work guard, not as aligned CPU.
CPU deltas use PID + `lstart` birth identity, shared `perf_counter_ns` capture
midpoints, explicit interval bounds, per-process classification/CPU, category
totals and capture brackets. A capture is not atomic; its duration, counter
rounding, short-lived unseen processes and bursts below the sampling scale limit
precision. These observations do not prove causal effects or absolute idleness.

New, exited or exec-changed foreign processes make an interval ineligible for
low-background selection; so do missing/gapped intervals longer than 5 seconds.
The full tick window includes scheduled wake through step/snapshot and client
delivery where measured. A tick is low-background only with full interval
coverage and **every overlapping interval's aggregate background <=20% of one
core**, with workload and unknown totals each below 20%. Missing coverage never
means zero CPU. The report retains original tick identities, aligned sample
indexes, weighted mean/max CPU, coverage and selection counts.

Each cell reports timing distributions for **all measured ticks** and for the
**low-background subset**, including empty-subset null statistics. Production
client gaps require consecutive original receipts and complete low coverage
between them; sparse retained ticks are never joined into artificial cadence.
The existing all-tick timing verdict remains authoritative. The subset is
descriptive sensitivity analysis, cannot turn FAIL into PASS, and does not
establish unconfounded causation. Unknown/no-low coverage must be disclosed.

## Operator commands (prospective; not executed here)

Read-only admission check, no lease or gameplay:

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --classified-host-load --check
```

Fixed panel, only in an available Mac window, with a fresh output directory:

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --classified-host-load --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-argon-classified-04
```

The launcher already enforces CPU, nice 15, five math thread caps at 1 and
caffeinate. The full panel still takes the original 45–60 minute forecast and
120-minute lifetime cap. Do not run alongside foreign heavy agent/repo jobs.
Choose another new directory if this one exists; never overwrite an attempt.
To recompute a completed classified report, use the existing `--reduce <raw.json>
--output <fresh-report-directory>` command; the recorded policy selects its
covariate reporting automatically.
