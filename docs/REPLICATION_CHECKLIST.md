# Replication checklist for coauthors and artifact reviewers

## Available now

- [x] Final manuscript PDF and corresponding five-phase framework image.
- [x] Original one-click source, checked against original manifest SHA-256.
- [x] Full 19-target archive: per-target diagnosis, GMM candidates,
      per-component profiles, probe folds, module predictions and downstream scores.
- [x] Benchmark inventory, input SHA-256 checksums, feature schema and exclusions.
- [x] Numeric checks for manuscript Tables II–VI and all Figure 5 ablations.
- [x] Rebuild of all seven Results panels from archived tables.
- [x] RQ1 controlled geometry and label-invariance checks, retained as archived evidence.
- [x] RQ2 outside/inside loss differences, downstream models, calibration,
      oracle-affinity and fixed-source diagnostic checks.
- [x] RQ3 external controls, component variants, GMM selection,
      lambda/q/alpha grids, signed-log and covariance sensitivity,
      support-filter/influence checks, exploratory project triage.
- [x] Captured Python/library versions and source-run parameters.
- [x] Local author-machine directory path fields sanitized for sharing.

## Required before claiming full reproducibility or publishing a public artifact

- [ ] Obtain/share the exact 19 benchmark CSV input files, within redistribution terms,
      or provide immutable public download instructions and checksum validation.
- [ ] Resolve AEEEM 19 actual run predictors vs. 61 manuscript Table I metrics.
- [ ] Verify whether four bug-related AEEEM fields are available before the predicted
      outcome; if leakage is present, exclude them and rerun/revise the paper.
- [ ] Independently rerun the original full script from the verified data in a clean
      Python 3.12 environment, not just a smoke-mode test.
- [ ] Compare the rerun output with the published archived values and record any
      library-version or stochastic numerical differences.
- [ ] Decide repository licence, data redistribution rights, and anonymization
      policy; never assert third-party dataset ownership.
- [ ] Record a stable public artifact identifier/DOI and README URL once published.
