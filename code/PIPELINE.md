## End-to-end pipeline (reconstructed)

This repository snapshot did not contain the original `data/work/` experiment scripts anymore (they were deleted and the project is not a git repo), therefore the **exact** original code cannot be recovered.

This `upload/` package provides a **reconstructed** pipeline that matches the reported workflow and produces **compatible intermediate artifacts** (filenames + columns) so that:

- steps can be run end-to-end starting from `data/raw/merged_gw_25-26.csv` (and the merged multi-season data under `data/raw/`)
- outputs feed into downstream steps without manual edits
- final deliverables (predictions CSV, figures, SHAP plots, tables) are generated deterministically given the same environment + seed

### Directory conventions

- `data/raw/` input CSVs (already merged/cleaned)
- `data/processed/` generated flat datasets and split indices
- `artifacts/npz/` sequence tensors (e.g., `processed_data_v2_L30*.npz`)
- `artifacts/models/` saved model checkpoints
- `artifacts/cache/` cached intermediate outputs (e.g., stage-1 probabilities)
- `artifacts/pred/` prediction CSVs
- `tables/` all comparison tables (baseline, ablation, sweep logs)
- `figures/` all final PNG figures
- `logs/` run logs

### High-level steps (intended)

1. Build flat dataset(s) and splits
2. Generate sequence tensors with `seq_len=30`
3. Train stage-1 temporal model and cache probabilities
4. Train stage-2 LightGBM on flattened sequences + cached probabilities
5. Predict holdout season (25–26) and export paper-ready CSV
6. Run hyperparameter sweeps / ablations and export tables
7. Run SHAP and export SHAP figures

