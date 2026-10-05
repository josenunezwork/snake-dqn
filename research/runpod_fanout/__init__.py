"""RunPod fan-out runner for Tier-1 experiments (screens, sweeps, censuses).

Not for strict gates or serving qualification (refused by :mod:`.jobspec`); a strict gate
that opts into RunPod (sequential strict template v3, governance amendment 2026-10-05) uses
``research/sequential_strict_template/remote_backend.py``, which reuses the serverless pieces
here with its own handler and runtime. Usage is in
the :mod:`.runner` module docstring; policy in ``fanout_policy.json`` and the upload
allow-list in ``checkpoint_allowlist.json``.
"""
