# T34 Path B prospective timing clarification — 2026-10-19

Written before any efficacy estimates or bank runs. Supersedes only the nominal
deadline sentence in PROTOCOL.md. The server performs step, snapshot, broadcast,
then a full sleep of 1/fps. Submitting at the end of step and expiring one sleep
interval later would reject even an instantaneous response at the next action.

Nominal target-frame deadline = submission start + **2/fps**: one cadence sleep
interval plus one processing budget for the target frame. Serialization,
transport, previous snapshot/broadcast and target pre-action preparation all
consume this budget. The actual consumption cutoff remains the **first next
frame's hero action boundary**. No tick waits for a result, a response completing
after consumption is lost, and no response can be reused in a later frame.
Misses, busy requests, startup costs and every fallback remain in estimates.
Both completion and consumption must be strictly before the nominal deadline.

Controls changing speed, food, epsilon, pause, reset, mode or checkpoint fence
requests. The worker checks its deadline between speculative transitions and
returns an expired fallback; no retry or replacement request is queued.

This changes no action rule, trigger, horizon, input access, bank, comparator,
metric, statistical threshold, resource cap, timing bar or serving scope.
Pre-bank implementation-only checks used one unallocated seed (3519284711) to
inspect serialization and compare the next transition. They produced no
efficacy estimate and are excluded from every development bank. Unit tests use
synthetic release-shaped checkpoint fixtures, also excluded from efficacy.
