# P packet-08 construction and VM verification

Mechanical successor to packet-07; science source inventory and recipe unchanged.
Prepared only. Opus reviews this Astra build; no independent review is claimed here.

- Real Mac lookup before any adapter binding returned exactly
  `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`
  (FRP3). The regression asserts the entire ordered list; a future registry change
  refuses this proof. B8 choice has requested=None; FRP3 requested=frp3-s12. Names
  are the actual registry names, confirmed against the resolver, not stems.
- Adapter side effects accept only GameSession's exact hero champion_path keyword,
  no-argument opponent lookup, and no-argument v8 lookup. Other roles/arguments
  raise. Real-constructor binding counts and scope restoration are tested.
- Sparse tests load a test-only sitecustomize guard through PYTHONPATH in every
  Python interpreter. It blocks /Users except isolated test files and installed
  venv dependencies; ancestor existence probes allow walking to that venv. It
  checks open/listdir/scandir/chdir/stat/lstat/access; refuses native subprocesses,
  os.system, and child guard-environment removal. Negative probes prove refusal.
- The VM fixture invokes real main(run), execute, spawn ProcessPoolExecutor,
  worker, no-argument uploaded_checkpoints (environment inherited), run_arm,
  publication and empty-ledger validation. Only SEEDS (one excluded fixture),
  WORKERS=2 and run_arm frames=60 are substituted by the external test driver,
  also imported by spawn children. No production engineering mode was added.
  It additionally runs main(smoke), natural L600, P0/P48 H300, death and respawn.
  Worker PID/startup records prove each working interpreter installed the guard.
- Mac construction starts with SNAKE_MECHANICS_V2 removed. Exact checkpoint
  choices are compared separately, apart from upload path. After real checkpoint
  loading, the test maps only checkpoint path metadata to the uploaded paths.
  It then compares complete 60-tick arm records (including roster and action
  trace), L600 state digest, every semantic certificate and detail field for
  P0/P48, plus 200 death/respawn ticks and final digest. Pickle transport hashes
  and sizes are checked independently: tensor storage identities and diagnostic
  timing counters are not deterministic byte serialization. Each real submit
  verifies its restored source digest, and both root sizes must fit the cap.
- Qualified transport remains byte-identical (tree
  5833e54e9c5396121cbbf99a5cd4e52006e564fd). supervisor.py creates a fresh absolute
  /content/snake-cli/<generation> root (1453,1512), sets COLAB_INPUT_DIR to its
  inputs child (1477), mkdirs parents (1633-1640), and uploads declared dests.
  transfers._rpc(UPLOAD_ASSEMBLE) creates a regular temporary file with xb,
  checks size/hash and os.replace publishes it (transfers.py 206-230). The
  regression executes that exact RPC code on both real checkpoint payloads,
  then admits the resulting regular absolute layout through uploaded_checkpoints.
  No cache links, Drive mounts or symlink staging are used on this route.
- Ledger v2 carries a separate construction field (uploaded.py sha256, mechanics
  flag). Each payload also records it; a hash-bound outcome-free metadata export
  binds the construction to payload unit_id/size/sha256. Metadata is complete
  before payload publication; the payload .complete marker is last. Resume and
  ledger creation validate metadata without parsing outcomes; changed adapter,
  mechanics, missing or foreign metadata refuse. Admission binds all six unit
  components, and manifests require payload plus metadata exports. Old unbound
  units refuse; 06b exported none and packet-07 was not executed locally.

RETRY_08.json retains execution ordinal 2 / manual completion ordinal 1 of 2.
Packet numbering does not consume attempts. No science outcomes or protected
payloads are opened by these engineering checks; all real gameplay uses FIXTURE_SEED.
