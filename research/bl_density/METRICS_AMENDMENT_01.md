# Pre-data clarification 01

This amendment is committed before any density gameplay measurement. METRICS.md remains
immutable. Include this file's SHA-256 alongside the original metric digest in every run.

For the classifier, drop columns constant on the TRAINING rows and report their feature names.
Return INSUFFICIENT only when no varying training feature remains (or the other original
insufficiency rules apply). Binary v8 indicators can legitimately be constant; one such column
must not suppress the entire 35-feature diagnostic. Retain every column in the KS report.
Use sklearn LogisticRegression(random_state=0,max_iter=1000,class_weight='balanced'), default
L2 regularization C=1; StandardScaler fitted only to training rows. Sort episode IDs within each
class and hold out the last ceil(25%) episodes of each class, retaining all rows. Report actual
train/test IDs and counts. Same episode identity across classes is invalid. No fallback model.

Tiny smoke labels may be retrospective-censored by their deliberately shortened horizons:
when fewer than 300 original transitions remain and the original hero has not died, true doom
is unknown. Report null and exclude these roots from recall/continuation agreement checks.
The full-H300 branch still executes. This exception applies only to preflight; production
sampling t<=4700 on H5000 remains fully observed and exact.
