# Traceability: manuscript statements -> archived artifacts

All paths below are relative to `results/archived_experiment/` unless prefixed.
Every target additionally has detailed files under `targets/<family>/<target>/`.

| Manuscript element | Main saved artifacts |
|---|---|
| Benchmark Table I and source features | `dataset_inventory.csv`, `family_feature_schemas.json`, `feature_alignment.csv`, `excluded_columns.csv`, `dataset_quality_audit.csv` |
| Protocol/settings/seeds and model versions | `run_manifest.json`, `analysis_manifest.json`, `execution.log`, `integrity_report.json` |
| GMM within-class partitions and selection | `gmm_selection.csv`, `representation_profiles.csv`, `targets/*/*/gmm_candidates.csv`, `targets/*/*/groups.csv` |
| Component sizes, project counts, stability | `representation_project_composition.csv`, `representation_project_validity.csv`, `targets/*/*/group_project_counts.csv`, `targets/*/*/group_project_validity.csv` |
| Source-probe transfer difficulty | `representation_profiles.csv`, `probe_fold_diagnostics.csv`, `targets/*/*/probe_folds.csv` |
| Local ambiguity and source hardness | `representation_profiles.csv`, `targets/*/*/groups.csv` |
| Proximity test and controlled spatial cases | `target_diagnoses.csv`, `controlled_construct_checks.csv`, `targets/*/*/proximity.csv`, `class_specific_proximity.csv` |
| Target affinities and class summaries | `representation_profiles.csv`, `target_diagnoses.csv`, `responsibility_concentration.csv` |
| RQ1 Table II and Figure 2 | `representation_profiles.csv`, `target_diagnoses.csv`, `gmm_selection.csv`, `figures/published/fig_rq1*.png` |
| RQ1 source-profile stability and fixed-source checks | `representation_project_validity.csv`, `fixed_source_integrity.json`, `fixed_source_summary.csv` |
| RQ1 label-invariance control | `evaluation_only_label_negative_control.csv` |
| RQ2 Table III and Figures 3–4 | `diagnostic_associations.csv`, `project_rq2_analysis.csv`, `targets/*/*/module_predictions.csv`, `figures/published/fig_rq2*.png` |
| RQ2 Table IV (GBT downstream) | `downstream_full_metrics.csv`, `downstream_subset_metrics.csv` |
| Inside vs outside proximity contrast | `proximity_loss_comparison.csv`, `targets/*/*/subset_metrics.csv` |
| Oracle-affinity diagnostic | `oracle_affinity_comparison.csv`, `oracle_affinity_associations.csv`, `targets/*/*/oracle_affinity.csv` |
| RQ3 Figure 5 (five variants by class) | `ablation_associations.csv`, `ablation_diagnostics.csv`, `figures/published/fig_rq3*.png` |
| RQ3 Table V (entropy, MMD, separability) | `external_diagnostic_associations.csv`, `predictive_uncertainty.csv` |
| RQ3 Table VI (alpha by family) | `sensitivity_results.csv` |
| RQ3 lambda/q and modeling alternatives | `sensitivity_results.csv`, `model_variant_associations.csv`, `model_variant_sensitivity.csv`, `downstream_calibration_sensitivity.csv` |
| RQ3 low-support and project influence | `class_count_sensitivity.csv`, `leave_one_target_influence.csv` |
| RQ3 exploratory triage | `retrospective_project_triage.csv`, `exploratory_incremental_prediction.csv` |
| Retrospective prevalence contrasts | `retrospective_prevalence_checks.csv` |
| Supplementary proximity references | `class_specific_proximity.csv`, `representation_project_associations.csv`, `family_centered_associations.csv` |
| Family-level source variation | `representation_project_validity.csv`, `representation_project_composition.csv` |
| Per-target diagnostic and prediction traces | `targets/*/*/diagnosis.json`, `targets/*/*/full_metrics.csv`, `targets/*/*/module_predictions.csv` |

**Scope:** This mapping identifies archived evidence. It does not claim every
supplementary result was rerun here, and it does not resolve the AEEEM feature
provenance concerns recorded in `docs/DATA_PROVENANCE_AND_OPEN_ISSUES.md`.
