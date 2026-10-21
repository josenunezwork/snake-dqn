# E10 two-phase sampling clarification (pre-data)

Authorized by the same owner request and round-03/04 E10 judgments. This sibling
clarifies REGISTRATION_E10.md's coverage language before scientific collection;
only burned operational fixtures have run. Original registration bytes remain
unchanged. No scientific N, prefix menu, continuation, threshold, cap or label
changes. The 0-unit bridge reuses the first 44/42/42 eligible E7 source worlds,
not an expanded census of prefix outcomes for every screened eligible world.

## Selected versus missing

Per mix: N screened worlds, e eligible, k selected eligible (44/42/42).
Unselected eligible worlds are `not_selected_by_design`, never zero observations
and never an instruction to collect more gameplay. `selected_missing` means an
actually selected source lacks any of its 24 training outcomes or six paired
confirmation rows; this makes the scientific envelope INCOMPLETE. Retain total
eligible-unmeasured counts and a worst-case finite-population sensitivity row.
Selection is first quota in frozen source/roster order, before prefix outcomes;
population inference assumes exchangeability within each mix and selection
independent of prefix outcomes conditional on eligibility. Report that assumption.

## Population U and uncertainty

Collapse each source's 24 training outcomes to U_i=max(0,max24 delta_i).
q=e/N, mu=mean selected U_i; population estimate theta=mean_m(q_m*mu_m).
Ineligible worlds contribute zero by abstention through q; do not append them
to the conditional sample or divide observed eligible gains by screened N.
Confirmed-only U and productive gross supply use the same two-phase weighting.
Measured unsuccessful selected confirmations contribute zero; missing selected
confirmations remain missing. Every source counts once across its ancestors.

Primary working Student-t display uses a conservative first-order delta-method
variance for overlapping prevalence and selected-source samples. Let
v_q=q*(1-q)/(N-1), v_mu=s_U^2/k, and
v_m=(abs(mu)*sqrt(v_q)+q*sqrt(v_mu))^2. This Cauchy covariance bound does not
assume independence of the overlapping samples. Equal-mix V=sum(v_m)/9,
d_m=min(N-1,k-1), df=V^2/sum[(v_m/9)^2/d_m]. UB90=min(B,theta+t.isf(.1,df)*sqrt(V));
when V=0 the bound equals theta. B=2501.5 is the universal H5000 score ceiling:
reset length1, at most one food increment/frame, sum_{t=1..5000}(1+t)/5000.
This working t bound is an approximation, not exact finite-sample coverage.
FINITE_MENU_INSUFFICIENT iff this complete valid training U UB90<40; equality
is INCONCLUSIVE. Missing selected rows precede that rule as INCOMPLETE.

Also report rigorous bounded-mean stopping sensitivity separately: allocate
alpha=.1/6 to each of 3 mix prevalence upper bounds and 3 conditional-U upper
bounds. Prevalence uses alpha_N=alpha/[N(N+1)] with Hoeffding union over N;
conditional U uses its fixed-k Hoeffding bound with range [0,B]. Clip q to [0,1]
and U to [0,B], multiply per mix and average. This is an additional jointly
one-sided 90% upper sensitivity under the stated sampling assumptions; it does
not replace the registered working-t label. Preserve the original per-mix
Wilson and anytime displays as additional rows.

Productive Wilson denominators are selected source counts, natural productive
fraction is q times that selected fraction, and gross supply is q times mean
confirmed productive tail mass across selected sources. Use the entire B8 dead
tail including its terminal death step, hence (5000-death_frame+1) frames.
Report per-mix, ancestor and sealer coverage, equal-mix gross/tau, r>=.163 and
feasible tau=(40+C)/gross<=1. Historical E7 point labels do not admit delay-only
prefixes. Archive admission needs complete data, historical TRANSFER_ADMISSIBLE,
productive coverage, feasible tau and U not FINITE_MENU_INSUFFICIENT. C from
natural-world evidence or explicitly labelled 0/5.9/10 planning sensitivities.
