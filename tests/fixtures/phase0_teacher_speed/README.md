# Phase 0 search-teacher speed oracle

`teacher_pre_45069720.py` is a verbatim copy of
`research/phase0_search_20261019/teacher.py` from commit `45069720` before the
teacher-speed optimization.  Its SHA-256 is
`d15960f0dab45b3e08d49549990620e341a0e6b2c665654ebf669b43d9d14f17`.

The parity suite imports this file under a private module name.  It builds a
separate strict world for each arm, so the oracle never imports or invokes the
candidate scorer.  The registered timing fixtures cover the scripted, frozen,
and mixed rosters at synthetic valid-serpentine hero lengths 200, 800, and
1500.  Each arm uses the registered search configuration: N=8, H=128,
epsilon=0.02.

Batching changes floating-point forward grouping, so score and initial-Q
comparisons permit an absolute tolerance of `1e-3`; chosen actions must agree.
Within one CPU implementation, repeated candidate results are required to be
exactly equal.  The suite also asserts that a search leaves its source
simulator and Vector61 runtime arrays untouched, and that an exact score tie
selects the first legal action.
