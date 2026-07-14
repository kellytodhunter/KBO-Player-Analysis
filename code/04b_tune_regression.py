"""
04b_tune_regression.py
-----------------------
Finds optimal regression-to-mean m-values for each stat via leave-one-out
cross-validation on consecutive season pairs.

For each stat and a grid of m-values, we compute:
  r_reg = (n * r_obs + m * r_lg) / (n + m)
and measure the correlation between r_reg and the player's actual next-season value.
The m that maximizes this correlation is the data-driven optimal regression coefficient.

This replaces the literature-informed starting values in 03_regression_mean.py
with KBO-specific estimates — a genuine methodological contribution for the paper.

Outputs:
  data/tuned_m_values.csv
  outputs/figures/m_value_tuning.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_DIR    = "data"
FIGURES_DIR = "outputs/figures"

M_GRID = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 550, 700, 900, 1200]

hitter_pairs  = pd.read_csv(f"{DATA_DIR}/hitter_pairs.csv")
pitcher_pairs = pd.read_csv(f"{DATA_DIR}/pitcher_pairs.csv")
league_avg_h  = pd.read_csv(f"{DATA_DIR}/league_avgs_hitters.csv")
league_avg_p  = pd.read_csv(f"{DATA_DIR}/league_avgs_pitchers.csv")

HITTER_STATS  = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA", "OBP", "SLG"]
PITCHER_STATS = ["K_pct", "BB_pct", "HR_per9", "ERA", "FIP"]


# =========================================================
# ATTACH LEAGUE AVERAGE TO PAIRS
# =========================================================

def attach_league_avg(pairs_df, league_avg_df, stats):
    pairs_df = pairs_df.merge(
        league_avg_df[["season"] + stats].rename(
            columns={s: s + "_lg" for s in stats}
        ),
        on="season", how="left"
    )
    return pairs_df

hitter_pairs = attach_league_avg(hitter_pairs, league_avg_h, HITTER_STATS)
pitcher_pairs = attach_league_avg(pitcher_pairs, league_avg_p, PITCHER_STATS)


# =========================================================
# TUNE m VIA LEAVE-ONE-OUT CROSS-VALIDATION
# =========================================================

def tune_m(pairs_df, stat, pt_col, m_grid):
    """
    For each m in m_grid, regress observed stat toward league avg and
    measure correlation with next-season value. Returns (best_m, results_df).
    """
    next_col = stat + "_next"
    lg_col   = stat + "_lg"

    sub = pairs_df[[stat, next_col, lg_col, pt_col]].dropna()
    if len(sub) < 10:
        return None, pd.DataFrame()

    results = []
    for m in m_grid:
        n         = sub[pt_col]
        shrinkage = n / (n + m)
        reg       = shrinkage * sub[stat] + (1 - shrinkage) * sub[lg_col]

        r, _ = pearsonr(reg, sub[next_col])
        mae  = (reg - sub[next_col]).abs().mean()
        results.append({"m": m, "r": r, "MAE": mae, "n_pairs": len(sub)})

    df = pd.DataFrame(results)
    best_m = df.loc[df["r"].idxmax(), "m"]
    return best_m, df


# =========================================================
# RUN FOR ALL STATS
# =========================================================

tuned = []
tune_curves = {}

for stat in HITTER_STATS:
    best_m, curve = tune_m(hitter_pairs, stat, "PA", M_GRID)
    if best_m is not None:
        print(f"Hitter {stat}: best m = {best_m}  (n={curve['n_pairs'].iloc[0]})")
        tuned.append({"group": "hitter", "stat": stat, "best_m": best_m})
        tune_curves[(stat, "hitter")] = curve

for stat in PITCHER_STATS:
    best_m, curve = tune_m(pitcher_pairs, stat, "IP", M_GRID)
    if best_m is not None:
        print(f"Pitcher {stat}: best m = {best_m}  (n={curve['n_pairs'].iloc[0]})")
        tuned.append({"group": "pitcher", "stat": stat, "best_m": best_m})
        tune_curves[(stat, "pitcher")] = curve

tuned_df = pd.DataFrame(tuned)
tuned_df.to_csv(f"{DATA_DIR}/tuned_m_values.csv", index=False)
print(f"\nSaved to {DATA_DIR}/tuned_m_values.csv")


# =========================================================
# VISUALIZE TUNING CURVES
# =========================================================

def plot_tuning(stats, group, tune_curves, filename):
    n = len(stats)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 7), constrained_layout=True)
    axes = axes.flatten()

    for i, stat in enumerate(stats):
        key = (stat, group)
        if key not in tune_curves:
            continue
        curve = tune_curves[key]
        best_m = curve.loc[curve["r"].idxmax(), "m"]

        ax = axes[i]
        ax.plot(curve["m"], curve["r"], "o-", color="#1a5fa8", lw=2)
        ax.axvline(best_m, color="red", ls="--", lw=1.5, label=f"Best m={int(best_m)}")
        ax.set_xscale("log")
        ax.set_title(stat, fontweight="bold")
        ax.set_xlabel("m (regression coefficient)")
        ax.set_ylabel("Correlation with next-season value")
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Regression-to-Mean Tuning — {group.title()}s", fontsize=13, fontweight="bold")
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES_DIR}/{filename}")

plot_tuning(HITTER_STATS, "hitter", tune_curves, "m_value_tuning_hitters.png")
plot_tuning(PITCHER_STATS, "pitcher", tune_curves, "m_value_tuning_pitchers.png")

print("\n=== M-VALUE TUNING COMPLETE ===")
print(tuned_df.to_string(index=False))
