# Run Results Summary

**This file has been rewritten twice.** History, for anyone auditing this project:

1. Originally reported test primary 0.5953 (delta +0.0007). That number came from a shallow
   early run that never actually finalized the champion -- see the README's "Bugs found and
   fixed" section.
2. After fixing the finalize bug, the real one-time test check on the original champion
   (causal-history DeepFM, unregularized) came back at test primary 0.5889, delta **-0.0057** --
   worse than baseline despite winning validation by +0.0031.
3. This version, after re-screening under a stricter protocol and spending a second (and, per
   this project's own rules, final for now) one-time test check on the best surviving
   candidate.

## Stricter re-screening (this run)

Per the concern that the original champion was picked by comparing many variants against one
fixed, biased validation split, `rescreen_champion.py` added a second, unbiased validation
slice (`log_random_4_22_to_5_08_pure.csv`, same date window as official valid, 2022-04-22..28,
never touching test) and required 5-seed agreement, not just seed 0.

| config | official valid (mean, 5 seeds) | unbiased valid (mean, 5 seeds) | passes dual gate? |
|---|---|---|---|
| FM baseline | 0.6016 | 0.3645 | -- |
| champion_original (emb_dim=12, hidden=96, dropout=0.1, wd=1e-6) | 0.6041 (+0.0025) | 0.3751 (+0.0106) | **yes**, 5/5 seeds both slices |
| champion_small_capacity (emb_dim=6, hidden=48) | 0.6038 (+0.0022) | 0.3746 (+0.0101) | **yes**, 5/5 seeds both slices |
| champion_regularized (dropout=0.3, weight_decay=1e-4) | 0.6038 (+0.0023) | 0.3745 (+0.0100) | **yes**, 5/5 seeds both slices |

**Finding: all three configs pass equally.** The gate doesn't discriminate between them --
differences between configs (~0.0003) are inside one standard deviation. This is itself an
informative negative result: the dual-validation check we built specifically targets
exposure-bias overfitting (screening against one biased split), and it does NOT show any of
these models are more prone to that than FM. Read together with the result below, that
points the likely explanation for the earlier test-set failure away from exposure-bias
overfitting and toward **temporal drift**: both validation slices share the same date window
(Apr 22-28); test is Apr 29-May 8, a week-plus later, and nothing in this gate probes that.

## One-time test-set results (two candidates checked; see note below on why we stopped at two)

| | GAUC | nDCG@5 | test primary | delta vs. FM baseline |
|---|---|---|---|---|
| Official FM baseline | 0.6610 | 0.5282 | **0.5946** | -- |
| champion_original (unregularized) | 0.6523 | 0.5255 | 0.5889 | -0.0057 |
| champion_regularized (dropout=0.3, wd=1e-4) | 0.6554 | 0.5265 | 0.5909 | **-0.0037** |

Regularization closed about a third of the gap (-0.0057 -> -0.0037) but did **not** flip the
sign. Combined with the dual-validation-gate result above, this is reasonably strong evidence
that the remaining gap is temporal (patterns that hold within the Apr 22-28 window, checked
two ways, but decay by Apr 29-May 8) rather than the exposure-bias overfitting the new gate
was built to catch. Regularizing further would likely keep closing the gap slowly at the cost
of validation signal, not fix the underlying mismatch.

**We deliberately did not spend a third one-time test check on `champion_small_capacity`.**
Two checks (original, then regularized) is already the edge of what this project's own
protocol should tolerate without becoming exactly the "keep touching test until something
wins" pattern the protocol exists to prevent. If you want a third check, that's a real
decision to make explicitly, not something to run by default.

## Recommendation (unchanged)

**Submit the official FM baseline.** Neither champion beats it on the one-time test check.
If you want to keep pursuing the causal-history direction, the next experiment that would
actually test the temporal-drift hypothesis (rather than re-testing the same hypothesis a
third way) is training on a rolling/recency-weighted window, or evaluating a train/valid split
with an explicit temporal gap between them (e.g. train on 04-08..04-18, validate on 04-25..04-28,
leaving a buffer) to make validation itself sensitive to drift the current one-week-adjacent
split doesn't expose.

Artifacts:
- `run_logs/rescreen_results.json` -- all 15 (3 configs x 5 seeds) dual-validation records.
- `run_logs/final_result.json` -- champion_regularized's one-time test result (current).
- `run_logs/final_result.previous.json` -- champion_original's one-time test result (superseded, kept for the record).
- `rescreen_champion.py` -- the re-screening sweep (resumable).
- `finalize_and_score.py` -- one-time test scoring, now config-aware and gated on rescreen_champion.py having actually passed.
