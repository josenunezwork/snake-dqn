# Frozen T34 serial shadow oracle

Verbatim modules from commit `a832e281` (the parent of this implementation).
Do not modernize these files or let their imports resolve to candidate policy or
engine helpers. `research.t34_exit_b8.profile_shadow_batching.serial_module`
loads the three modules with private, dependency-pinned import bindings and
checks these SHA-256 identities:

- `teacher_v2_serial.py`: `db5f58add590dcf56146c1057e154231b96eac377a40d45bc4b5d8d7de0a9d7d`
- `policy_serial.py`: `ae0be54e70957fa3936b0b950cdc20fcf1b16fdfce3852c9618b110103c08635`
- `engine_serial.py`: `ad1324eeffc0b82c6b8b8eca6fc1ffd328e8a0f8213d0d24952506cf161feb05`

Simulator, raster, runtime, and shared shadow helpers are unchanged by this
optimization. Each oracle/candidate run owns independent worlds and policy
state. World JSON and receipt sidecars are compared byte for byte.
