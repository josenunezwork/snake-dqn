"""Draft-only transition classifier for the R32 fixed-D4096 contingency.

This module deliberately does not read files, issue commands, admit a study, or
release a candidate. It translates only the complete future packet state into a
draft status so a future independently reviewed registration can use the same
ordered truth table as ``DELIVERY_DRAFT.md``.
"""


def classify(
    registered_flag: bool,
    top_guards_pass: bool,
    funded_registration: bool,
    checkpoint_matches: bool,
    product_complete: bool,
    strict_pass: bool,
    serving_pass: bool,
    owner_release: bool,
) -> str:
    """Classify the fixed-D4096 delivery contingency without authorizing release.

    Args:
        registered_flag: Whether the published registered F4 disposition reports
            ``TOP_PASSED_DOSE_FAILED``.
        top_guards_pass: Whether every required top-controller bar and guard
            relevant to that qualifying disposition passed.
        funded_registration: Whether a new Astra-judged, pre-data registration
            funds the fresh route.
        checkpoint_matches: Whether the exact immutable fused D4096 bytes and
            provenance bind to F4.
        product_complete: Whether all separately required fresh product evidence
            is complete and passes its registered bars.
        strict_pass: Whether the separately registered fresh Mac strict gate
            passes.
        serving_pass: Whether Mac serving qualification passes.
        owner_release: Whether an owner release decision is recorded externally.

    Returns:
        A draft-only status. No value returned by this function is a dispatch,
        admission, serving, pin, or release authorization.

    Raises:
        TypeError: If any control is not a built-in ``bool``. In particular,
            truthy text such as ``"false"`` is not a control value.
    """
    controls = (
        registered_flag,
        top_guards_pass,
        funded_registration,
        checkpoint_matches,
        product_complete,
        strict_pass,
        serving_pass,
        owner_release,
    )
    if any(type(control) is not bool for control in controls):
        raise TypeError("all contingency controls must be built-in bool values")
    if not registered_flag:
        return "CONTINGENCY_CLOSED_NO_QUALIFYING_FLAG"
    if not top_guards_pass:
        return "CONTINGENCY_CLOSED_TOP_GUARD_FAILURE"
    if not checkpoint_matches:
        return "CONTINGENCY_CLOSED_IDENTITY_MISMATCH"
    if not funded_registration:
        return "CONTINGENCY_BLOCKED_UNREGISTERED"
    if not product_complete:
        return "CONTINGENCY_BLOCKED_PRODUCT_INCOMPLETE"
    if not strict_pass:
        return "CONTINGENCY_BLOCKED_STRICT_NOT_PASSED"
    if not serving_pass:
        return "CONTINGENCY_BLOCKED_SERVING_NOT_PASSED"
    if not owner_release:
        return "ELIGIBLE_FOR_OWNER_RELEASE_DECISION"
    return "DRAFT_OWNER_RELEASE_RECORDED_EXTERNAL_EXECUTION_REQUIRED"
