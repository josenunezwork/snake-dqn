"""T34 Claim A production pipeline: student labels -> replay/cache -> fit -> probe.

This package only imports, wraps and subclasses the ratified T34 modules
(``research.t34_exit_b8`` and ``research.t34_exit_b8_fit``).  It never edits
them, so their study-bound bytes and the frozen distillation loss are unchanged.
"""
