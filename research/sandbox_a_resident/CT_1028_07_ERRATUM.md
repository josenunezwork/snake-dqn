# Packet 07 appended reseal: structured exit code takes precedence

Use packet-ct-1028-07/reseal-01 after independent Opus review. The first packet-07
seal and all evidence remain immutable. The runtime source pin changes; science,
registered allocation, envelopes, qualified transport and serving do not.

A final structured-facts audit extended the signal-code guard from
CHILD_NONZERO_EXIT receipts to every retained receipt class and execution record:
a mechanical terminal reason cannot exempt a concrete non-signal exit code.
Exit 7 with reason timeout now refuses even if failure_class is SUPERVISOR or
RPC_EXEC_POLL_TIMEOUT and no error text is present. Tests cover all three receipt
classes and the nested poll/execution scan. This closes a structured counterexample
without interpreting provider stderr or any error-message literal.

The end-to-end fixture now uses canonical output/exports/ct directories. Its
completed c1 dispatch must refuse the already-used output identity; fresh c2
passes. Earlier synthetic fixtures had exported outside the declared output and
then attempted to dispatch a logically completed c1. Corrected cleanup removes
only the injected fault file, preserving the real canonical output and stages.
These are fixture corrections; numerical continuation remains identical.
