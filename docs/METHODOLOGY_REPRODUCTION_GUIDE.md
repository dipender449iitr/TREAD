# How the saved artifacts implement the methodology

## Five phases of the framework

1. **Inputs and preprocessing.** Within each family, hold out one target project,
   align numeric predictors, fit source-only imputation/constant-column removal/
   median-IQR scaling, then apply transformations to target metrics. Inspect
   `source_feature_preprocessing.csv` and `feature_alignment.csv`.
2. **Class-specific GMM representations.** Fit candidate diagonal GMMs separately
   for y=0 and y=1, BIC selection within a tolerance followed by project-resampling
   partition stability. Inspect `gmm_selection.csv` and each `gmm_candidates.csv`.
3. **Source-evidence characterization.** Project-held-out logistic-regression and
   random-forest probes produce log-loss transfer difficulty D. Same/opposite class
   nearest-neighbor comparisons produce local ambiguity A. H is their convex
   combination. Inspect `representation_profiles.csv`, `probe_fold_diagnostics.csv`.
4. **Target-relative diagnosis.** Source leave-one-out q-nearest-neighbor distances
   determine the cutoff; target-to-source distances identify covered target
   observations, yielding U. Covered target observations determine equal-prior
   within-class GMM responsibilities and T. Inspect `proximity.csv` per target
   plus `representation_profiles.csv`.
5. **Output.** Per-component target-weighted hardness R_y,g=T_y,g H_y,g;
   per-class R_y is the sum over components. U remains separate. Inspect
   `target_diagnoses.csv` and `targets/*/*/diagnosis.json`.

## Protocol

19 leave-one-project-out targets: 5 AEEEM, 9 JIRA, 5 PROMISE. All target metrics
may be used during the diagnosis; no target labels may be used until independent
outcome evaluation. RQ2 uses the target project as its statistical unit and
averages the 10 learner seeds before project-level comparisons.

## Separating evidence and model claims

The diagnosis does not add or rebalance target classifier training instances.
Class-specific R estimates are not calibrated probabilities, and geometric
proximity U is not an instance-level error label. The study evaluates observed
correlations; the package retains both supported and non-supported diagnostics.
