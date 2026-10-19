# Source-data and release checks

## 1. Dataset availability
Original benchmark CSVs were not present in the available original results archive.
Independent reruns require those exact 19 files or confirmed equivalent releases;
use the recorded SHA-256 values rather than assuming same-named CSVs are identical.

## 2. Manuscript-vs-run discrepancy (must resolve)
The finalized manuscript, Table I, lists **61 metrics** for AEEEM, while the
actual full-run dataset inventory and class feature schema identify **19
numeric predictors**. JIRA uses 65 and PROMISE uses 20 in both sources.
Please verify whether 61 was a historical dataset description, count of raw
attributes, or an accidental carryover, then revise the manuscript text if
needed. No numerical values were changed in this package.

## 3. Potential outcome-derived covariates (must investigate)
The AEEEM archived feature schema includes `nonTrivialBugs`, `majorBugs`,
`criticalBugs`, and `highPriorityBugs`. Their names suggest possible
outcome-derived information. Establish when each feature becomes available
relative to the predicted defect outcome. If they contain future or
label-dependent information, exclude these fields and rerun the experiment;
the current results cannot establish absence of this kind of leakage.

## 4. Reproducibility boundaries
* Recorded run software: Python 3.12.3; NumPy 2.4.0; pandas 3.0.3;
  SciPy 1.17.1; scikit-learn 1.8.0.
* Script SHA-256: `7b44f52a4dd3fe95d8d8645b3822ccb34f7d9908c4d9e57c96bef4dff9727754`.
* The complete source-data rerun has NOT been performed within this package
  construction; the archive is a preserved prior full run.
* Path strings referring to the former author's local environment were
  replaced with `<DATA_ROOT>` / `<ORIGINAL_RUN_ROOT>` in the public copy.
  Numerical results and recorded input hashes were not changed.
* The published seven figure PNGs are provided as manuscript assets, while
  the figure-generation script reproduces their underlying values from CSVs.

## 5. Before public release
Verify ownership/redistribution permissions; confirm final dataset provenance;
re-run if leakage is established; test fresh environment; make a public repository
with a permanent archive/DOI; add authors and licence after institutional review.
