# E7 — frontier-existence test (frozen development registration)

Authorized by the owner's E6/E7 request; authority round-02 JUDGMENT.md §(f).
Commit BEFORE any gameplay including preflight. Development only, excluded from
EVERY registered campaign verdict. No serving/release or training authorization.
Read REGISTRATION_E6.md for shared bank, history, matching and audit requirements.

Use the SAME 128 E6 late-death source worlds, each restored at 100, 150 and 200
decisions before its recorded death. Save full simulator/opponent/RNG state and
actual preceding 128-decision observation/action history for every ancestor.
No strict bank or sealed study access. Missing roots/history or failed audit =>
INCOMPLETE, never count unavailable worlds as failed rescues.

At each root sample EIGHT stochastic B8 prefixes of 64 decisions, using a
positive softmax temperature calibrated by bisection on the actual preceding
128 observations to expected 5% non-greedy probability (all legal six actions,
boost included; tied maximal actions count as greedy). Record temperature,
expected and realised non-greedy rates. No safety veto added. Then B8 greedy to
original H5000. All eight share one training future-environment RNG stream;
policy sampling has eight independent seeds. Opponents react at every step.
Select largest remaining mass-sum/5000 on TRAINING RNG, ties lowest prefix
index. Selection is per ancestor root, never on confirmation outcomes.

Confirm the selected 64-action prefix under TWO independent future RNG streams,
paired with greedy B8 from the exact same root/stream. Replay selected actions
when legal; if a selected action becomes illegal, use greedy B8 for that step
and report fallback count (never silently resample or select again). A confirmed
root requires candidate minus paired B8 >=+20 remaining mass-sum/5000 AND no
loss of terminal survival in EACH confirmation stream. One source world counts
once if ANY of its three roots confirms; separately report the fraction at each
100/150/200 ancestor. Confirmations are not selection/training RNG.

One complete fixed look. Identity/audit/parity/missing row/cap failure =>
INCOMPLETE. Otherwise <20% of the 128 source worlds => FRONTIER_SCARCE: the
1%-exploration pilot is unlikely to discover rescues unaided; register raised
exploration or an archive arm before any 100M tranche and prioritise lever 4.
>=20% => TRANSFER_ADMISSIBLE: a separately named lever-7 transfer arm with
80/10/10 occupancy may be registered. These labels never trigger an automatic
training run. Report 100/150/200 strata and the effective-reach collapse.

Cap 20 units AND 10 L4 allocation-hours (first wins), charged at ceiling 2 u/h,
4 u reserve (16 usable), inclusive of setup/export/failure. 12 single-thread CPU
processes, fixed-order merge. Shared E6 collection charged once to E6; if bank
absent E7 refuses. Forecast includes all 3 ancestors, 8 training prefixes,
paired B8 and 2 paired confirmations; judgment's 20M steps is a conjecture,
not permission to omit work. No retry or post-data extension. Qualified
research/colab_cli only; .sha256 + .complete exports. No Colab calls in this build.
Tiny Mac preflight: nice 10, <=2 total single-thread workers, separate burned
fixture namespace; operational timing only, never an E7 scientific outcome.
