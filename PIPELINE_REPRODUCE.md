## Paper reproduction pipeline (GitHub-ready)

**Paper:** *PatchTST-LightGBM Hybrid for Fantasy Premier League High-Scoring Player Prediction*  
**Venue:** DSAI (EI conference)

This repository contains a **paper-aligned, end-to-end reproducible pipeline** for:

- Building the **paper 20D dataset** with season-based splits
- Generating **leakage-safe sequences** (\(t\) label predicted from \([t-L, \dots, t-1]\))
- Training the **Hybrid model**:
  - Stage-1: PatchTST (ensemble over seeds)
  - Stage-2: LightGBM on \(Z=[\mathrm{flat}(X); \mu_p; \sigma_p]\)
- Producing paper tables (Table 4/5/6) and paper-style SHAP figures (Fig 5–7)

### Directory map (repository root)

- **`code/`**: all runnable scripts (paper-aligned entry points)
- **`data/processed/`**: generated datasets (CSV)
- **`artifacts/npz/`**: generated numpy bundles (`paper20d_L30.npz`)
- **`tables/`**: generated tables (CSV)
- **`figures/`**: generated figures (PNG)
- **`local_only/`**: *not for GitHub* (large/intermediate files, local runs)

### Environment

Recommended (conda):

```bash
cd <this-repo>
conda env create -f environment.yml
conda activate fpl-upload
```

Or pip:

```bash
cd <this-repo>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce (paper-aligned)

Run from the repository root (paths in scripts assume this working directory):

```bash
python code/data_loader_paper20d.py
python code/sequence_generator_paper20d.py

# Tables
python code/run_table4_paper_ablation.py
python code/run_table5_paper_comparison.py

# Holdout predictions + examples (Table 6)
python code/run_predict_holdout_paper.py
python code/make_table6_holdout_examples.py

# SHAP (Fig 5–7) + triptych
python code/shap_paper_analysis.py
```

### What gets written (expected outputs)

- **Dataset**
  - `data/processed/flat_paper20d.csv`
  - `data/processed/flat_paper20d_meta.json`
- **Sequences**
  - `artifacts/npz/paper20d_L30.npz`
  - `data/processed/paper20d_sequences_holdout_meta_L30.csv`
- **Tables**
  - `tables/` (Table 4/5 outputs + Table 6 examples)
- **Holdout predictions**
  - `data/predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv` (schema used by downstream figure scripts)
- **Figures**
  - `figures/shap_paper_fig5_top10_bar.png`
  - `figures/shap_paper_fig6_summary.png`
  - `figures/shap_paper_fig7_temporal_last10.png`
  - `figures/shap_paper_triptych.png`

### GitHub upload guidance

To keep the repo lightweight and reproducible:

- Commit **code + configs + small metadata**.
- Treat `artifacts/`, `figures/`, `tables/`, and large `data/` outputs as **generated**.
- Put any large local-only assets into `local_only/` and keep them out of version control.

See `.gitignore` in this folder for suggested exclusions.

