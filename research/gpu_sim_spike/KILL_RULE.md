CONTINUE (recommend a full port) only if **both**:
- end-to-end GPU cost per dense-search decision ≤ **1/10** of the L4 CPU baseline on the long-body mixes, **and**
- first-action agreement ≥ **99 %** of searched decisions on the S0 fixtures, with 0 unexplained (bug) disagreements.
Otherwise STOP: write `RESULT.md` with the numbers and the reason; no second attempt without a new brief.
