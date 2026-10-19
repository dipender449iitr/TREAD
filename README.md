# TREAD replication package — MSR 2027

**Paper:** Beyond Source Similarity: Diagnosing Evidence Adequacy for Defect Prediction.
**Method:** TREAD (Target-Relative Evidence Adequacy Diagnosis).

## What is included
* `code/TREAD_MSR_FINAL_ONE_CLICK.py`: original, unmodified one-click source;
  its SHA-256 matches the archived `run_manifest.json`.
* `results/archived_experiment/`: complete sanitized copy of the archived
  full-run outputs (19 targets; per-component and module-level traces;
  comparisons, sensitivity analyses, fixed-source checks, manifests).
* `figures/published/`: the seven manuscript results panel PNGs.
* `figures/framework/`: manuscript framework image.
* `scripts/`: paper-number verifier, original-benchmark hash checker, and
  figure re-creator **reading the recorded CSVs**, plus package inventory.
* `docs/ARTIFACT_TO_MANUSCRIPT_MAP.md`: mapping of each claim/analysis to files.
* `docs/DATA_PROVENANCE_AND_OPEN_ISSUES.md`: data-provenance and release checklist.

## Quick start: inspect and reproduce manuscript statistics

From the directory containing this README:

```bash
python -m pip install -r environment/requirements-recorded.txt
python scripts/check_manuscript_values.py
python scripts/regenerate_figures.py
```

The first script independently builds table-number checks directly from the
archived CSVs; the second regenerates the seven result figures in
`verification/regenerated_figures/`. The existing `figures/published/` PNGs
remain unchanged.

## Full experiment rerun (requires source benchmark CSVs)

Place the 19 *exact* benchmark CSVs in `data/Full Dataset-MSR/` in family
subfolders as described in `data/README.md`, then:

```bash
python scripts/check_inputs.py
python code/TREAD_MSR_FINAL_ONE_CLICK.py \
  --data-root 'data/Full Dataset-MSR' \
  --output rerun_results \
  --mode full --seeds 10
```

The full run includes project-held-out source probes, class-conditional GMM
selection, source-hardness and ambiguity, unlabeled-target proximity/affinity,
independent downstream predictions, ablations, sensitivities and diagnostics.


