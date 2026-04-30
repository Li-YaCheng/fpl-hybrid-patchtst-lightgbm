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

Run from the repository root:

```bash
python code/data_loader_paper20d.py
python code/sequence_generator_paper20d.py

python code/run_table4_paper_ablation.py
python code/run_table5_paper_comparison.py

python code/run_predict_holdout_paper.py
python code/make_table6_holdout_examples.py

python code/shap_paper_analysis.py
```

### Repository structure

- `code/`: all reproduction scripts (paper-aligned entry points)
- `data/`: small inputs; `data/processed/` is generated
- `artifacts/`: generated model artifacts and NPZ bundles
- `tables/`: generated CSV tables
- `figures/`: generated PNG figures
- `local_only/`: local large scratch (ignored by `.gitignore`)

### Outputs

Generated outputs are written under:

- `data/processed/`
- `artifacts/`
- `tables/`
- `figures/`

Notes:

- `data/predict_25_26_hybrid_predictions_patchtst_seed42_padpred.csv` is included as an example input used by some downstream figure scripts.
- Larger intermediate outputs should go into `local_only/` (ignored by `.gitignore`).

### Citation

Software citation metadata is provided in `CITATION.cff`.

If you want a BibTeX block for your paper (fill in authors / proceedings volume if applicable):

```bibtex
@inproceedings{<your_key>,
  title     = {PatchTST-LightGBM Hybrid for Fantasy Premier League High-Scoring Player Prediction},
  author    = {<Authors>},
  booktitle = {Proceedings of DSAI},
  year      = {2026},
}
```

Data source reference (IEEE-style):

> Official FPL API, `https://fantasy.premierleague.com/api/`, Online resource, 2025.

### License

This repository includes an MIT `LICENSE`. If your paper/venue requires a different license, replace it before publishing.

