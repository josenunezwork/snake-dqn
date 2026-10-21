# P1 pre-data execution note — 2026-10-21

The registration committed at `464b21f8` remains byte-identical. This note
clarifies the 1M qualification execution boundary before P1 data exist:
1,000,000 is not divisible by 128 worlds. `requested_pause_at=1000000`,
`pause_at=1000064`, overshoot 64 <=127 decisions. Preserve whole-world collection,
do not reset or discard trajectory state, and report requested/actual counts.
The fixed scientific target is exactly 100,000,000 actor transitions. All five
tranche boundaries are divisible by 128. This note changes no recipe, seed,
endpoint, bank, guard, deadline, one-look rule or budget.

At each boundary retain the full transitive snapshot set and its sidecars.
Resume from the qualification prefix to 20M (18,999,936 additional decisions),
then 40M, 60M, 80M, 100M. Only 100M exports the scientific endpoint. Y3's source
continuation is an excluded durability branch; qualification starts from Y3's
qualified origin, not from the durability branch's end. Replacement L4 state
comparison excludes elapsed-time diagnostics only; clocks cannot influence
policy, optimizer, simulator, RNG or update counts.

For the 120-unit tranche invoice, reserve is 24 units (20% of envelope),
not 20% of nominal cost. Nominal remaining tranche work <=24 L4-hours at 2u/h
is therefore required for 2x liability + reserve <=120u. The independent
complete-trajectory <=46h gate still includes the measured prefix and probes.
Preparation templates have no manufactured admission, URL, endpoint or parity
receipt; missing prerequisites are named and block dispatch.
