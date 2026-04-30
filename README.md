## PatchTST–LightGBM Hybrid for FPL — reproduction code

**Paper:** *PatchTST-LightGBM Hybrid for Fantasy Premier League High-Scoring Player Prediction*  
**Venue:** DSAI (EI conference)

This repository is the **paper-aligned reproduction package** for our Fantasy Premier League (FPL) experiments:

- Paper **20D feature** dataset + **season-based splits**
- **Leakage-safe** sequence generation: predict round \(t\) from \([t-L,\dots,t-1]\)
- Hybrid model:
  - Stage-1 PatchTST (ensemble over seeds)
  - Stage-2 LightGBM on \(Z=[\mathrm{flat}(X); \mu_p; \sigma_p]\)
- Paper tables + SHAP figures (Fig 5–7)

If you want a single entrypoint summary, see `PIPELINE_REPRODUCE.md`.

### 🔬 Key Results (Paper)

- **SOTA F1**: 0.8122 (PatchTST–LightGBM Hybrid, L=30)
- **vs. Pure GBDT**: +0.32% (0.8081 → 0.8122)
- **vs. End-to-end Transformer**: +0.81% (0.8041 → 0.8122)
- **Holdout Season (2025–26)**: well-calibrated probabilities with Top-10 uplift
- **Interpretability**: SHAP-validated two-stage mechanism (PatchTST probability dominates, GBDT calibrates)

**Core insight**: long-window temporal encoding (L=30) + uncertainty-aware GBDT fusion outperforms both pure statistical ML and naive deep learning.

### Quickstart

From the repository root:

```bash
python code/data_loader_paper20d.py
python code/sequence_generator_paper20d.py
python code/run_table4_paper_ablation.py
python code/run_table5_paper_comparison.py
python code/run_predict_holdout_paper.py
python code/make_table6_holdout_examples.py
python code/shap_paper_analysis.py
```

### Data source & attribution

- **Data source**: Official FPL API (`https://fantasy.premierleague.com/api/`), online resource, 2025.
- **Attribution request** (from the original data source author message):
  - If you use data from this project for a website/blog post, please add a link back to this repository as the data source.

More details: `DATA_SOURCES.md`.

### Environment

Conda (recommended):

```bash
cd <this-repo>
conda env create -f environment.yml
conda activate fpl-upload
```

Pip:

```bash
cd <this-repo>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce the full pipeline (paper-aligned)

Run from the repository root (~8–15 min on CPU; ~3–5 min with GPU for Stage-1):

```bash
# Step 1: Data preparation (~1 min)
python code/data_loader_paper20d.py
python code/sequence_generator_paper20d.py

# Step 2: Main experiments (~5–10 min)
python code/run_table4_paper_ablation.py
python code/run_table5_paper_comparison.py

# Step 3: Holdout evaluation + SHAP (~2–4 min)
python code/run_predict_holdout_paper.py
python code/make_table6_holdout_examples.py

python code/shap_paper_analysis.py
```

### Repository structure

| Path | Purpose |
|---|---|
| `code/` | Paper-aligned reproduction scripts |
| `code/data_loader_paper20d.py` | Build 20D feature set + season-based splits |
| `code/sequence_generator_paper20d.py` | Leakage-safe L=30 sequence generation |
| `code/run_table4_paper_ablation.py` | Reproduce Table 4 (Ablation Study) |
| `code/run_table5_paper_comparison.py` | Reproduce Table 5 (Cross-model Comparison) |
| `code/run_predict_holdout_paper.py` | 2025–26 holdout season prediction |
| `code/make_table6_holdout_examples.py` | Reproduce Table 6 holdout examples |
| `code/shap_paper_analysis.py` | Generate SHAP figures (Fig 5–7) + triptych |
| `data/` | Small inputs; `data/processed/` is generated (gitignored) |
| `artifacts/` | Model artifacts and NPZ bundles (gitignored) |
| `tables/` | Generated result CSVs (Table 4/5/6) (gitignored) |
| `figures/` | Generated PNG figures (gitignored) |
| `local_only/` | Large scratch space (gitignored) |

### Outputs

Generated outputs are written under:

- `data/processed/`
- `artifacts/`
- `tables/`
- `figures/`

Notes:

- `data/predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv` is included as an example input used by some downstream figure scripts.
- Larger intermediate outputs should go into `local_only/` (ignored by `.gitignore`).

### Paper

- **Venue**: DSAI (EI Conference), 2026
- **Preprint**: (add arXiv/DOI link when available)
- **Key figure**: two-stage Hybrid architecture (add a PNG link here if you export it, e.g. `figures/model_architecture_final.png`)

### Citation

Software citation metadata is provided in `CITATION.cff`.

If you want a BibTeX block for your paper (fill in authors / proceedings volume if applicable):

```bibtex
@inproceedings{Li2026PatchTST,
  title     = {PatchTST-LightGBM Hybrid for Fantasy Premier League High-Scoring Player Prediction},
  author    = {Yaxuan Li},
  booktitle = {Proceedings of the 3rd International Conference on Digital Society and Artificial Intelligence (DSAI)},
  year      = {2026},
  note      = {Accepted for publication}
}
```

Data source reference (IEEE-style):

> Official FPL API, `https://fantasy.premierleague.com/api/`, Online resource, 2025.

### License

This repository includes an MIT `LICENSE`. If your paper/venue requires a different license, replace it before publishing.

