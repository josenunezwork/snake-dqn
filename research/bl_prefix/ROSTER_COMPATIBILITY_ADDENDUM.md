# Pre-gameplay roster-label compatibility repair

The first real-entrypoint preflight refused BEFORE world construction/gameplay:
`strict_promotion.materialize_rosters` accepts only uint32 labels. Its roster
member selection depends solely on ordered world index and slot phase, not the
seed's value (`strict_promotion.py:715` onward). The simulator/EnvRng accepts the
full high-range seeds, but this legacy artifact materializer does not.

Freeze this package-only mechanical adapter before gameplay: construct roster
rotation with ordinal labels `0..N-1`, using the unchanged existing materializer,
then replace each plan's label with its actual audited high-range seed before
world construction. Ordinals are NEVER used to initialize or play a world.
They preserve exactly the original allocation-order opponent rotation; there is
no modulo reduction, seed truncation, protected-source edit or new mix.

Keep the failed preflight log. The next fresh output must prove full-width
seed propagation and the real independent 300-frame restoration controls in
all three mixes. A further failure is INCOMPLETE. The collision audit and seed
receipt remain valid because no allocated world/RNG identity changes.
