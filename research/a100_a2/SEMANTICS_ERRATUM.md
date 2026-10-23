# A2 pre-fit source-semantics clarification

2026-10-22, before any A2 fit or predictions. Independent Codex-V review found
that the two historical tables encode time differently. The frozen registration's
common `alive_steps + 1` event-time convention requires normalization on import.

T5 `horizon` is the number of transitions including the fatal transition
(`research/t5_separability_20261016/restore.py`, replay construction; the T10
horizon probe uses `T <= horizon` for a death). Therefore T5 dead branches become
`alive_steps = horizon - 1`; survivors at H300 stay `alive_steps = 300` and are
right-censored. A5 already records completed alive steps before death and needs
no subtraction. A death on transition 30/60/100/200/300 is positive at that
horizon, never shifted into the following interval. Earlier right-censoring
contributes only intervals whose endpoint was observed.

This clarifies the source adapter and preserves the registered labels, horizon
endpoints, split, factorial, fit endpoint, kill rules and caps. No label is
regenerated or selected based on a prediction. The registration remains immutable.
