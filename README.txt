# KBO Player Projection System

## Overview
A Marcel-style projection system for KBO (Korea Baseball Organization) hitters, developed as a research paper for the Asian American Interdisciplinary Research Journal (AAIRJ). The system projects player performance using multi-year weighted averages, empirical Bayes regression to the mean, and delta-method aging curves.

Note: Pitcher projections were explored during development but are not included in the final analysis due to insufficient data to produce reliable estimates. Pitchers are acknowledged as a limitation and direction for future work.

## Pipeline
1. Data preparation and cleaning (01_data_prep.py)
2. Birth year collection for age calculations (02_scrape_ages.py)
3. Empirical Bayes regression to the mean — m-value tuning (03_regression_mean.py, 04b_tune_regression.py)
4. Delta-method aging curves (04_age_curves.py, 04c_playing_time.py)
5. Marcel projections (05_projections.py)
6. Ablation study — component contribution analysis (05b_ablation.py)
7. Temporal backtesting across 2021–2024 held-out seasons (06_validation.py)
8. Figure generation (07_figures.py)
9. Anchor age comparison (27 vs 24) (anchor_comparison.py)

## Key Findings
- Marcel projections outperform both the prior-year-only and league-average baselines across all five hitter statistics (BB%, K%, HR%, ISO, wOBA)
- Optimal regression-to-mean parameter m ≈ 1 for KBO hitters with 300+ PA
- Aging curves peak near age 24–27 depending on the statistic, consistent with MLB research
- Anchor age choice (27 vs 24) has zero effect on projection accuracy — year-over-year deltas cancel out the anchor constant

## Metrics Evaluated
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson correlation (r)

## Tech Stack
Python, pandas, numpy, scipy, matplotlib

## Project Structure
    code/        — analysis pipeline scripts
    data/        — processed CSVs
    outputs/     — formatted research paper and figures
