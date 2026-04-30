"""
One-shot horizontal comparison runner.

Goal: you only need TWO commands in practice:
  1) python run_horizontal_comparison.py   (baselines + ours + final table)
  2) python run_ablation_lgbm_L30.py       (ablation table)

This script runs:
  - run_baseline_comparison.py
  - final_comparison.py   (fills "ours" with real metrics)
and produces `tables/final_comparison_results.csv`.
"""

from __future__ import annotations

import importlib


def main() -> None:
    rbc = importlib.import_module("run_baseline_comparison")
    fc = importlib.import_module("final_comparison")
    rbc.main()
    fc.main()


if __name__ == "__main__":
    main()

