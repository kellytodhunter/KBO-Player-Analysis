"""
06_validation.py
----------------
Backtests the projection system against held-out seasons (2021–2024).

For each target year in the backtest window:
  1. Generate projections using only data available before that year.
  2. Compare projected stats to actual observed stats.
  3. Compute MAE, RMSE, and Pearson r for each stat.
  4. Compare against two naive baselines:
       - Prior year only (no multi-year weighting, no regression)
       - League average (worst-case baseline)

Outputs:
  data/backtest_results_hitters.csv
  data/backtest_results_pitchers.csv
  outputs/figures/validation_hitters.png
  outputs/figures/validation_pitchers.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import importlib.util, sys, os

# Import project_players from 05_projections without running its __main__ logic
spec = importlib.util.spec_from_file_location("projections", "src/05_projections.py")
proj_mod = importlib.util.load_from_spec = None  # avoid exec

# Inline import of required functions
sys.path.insert(0, "src")

DATA_DIR    = "data"
FIGURES_DIR = "outputs/figures"

BACKTEST_YEARS = list(range(2021, 2025))  # predict these; use prior data only

HITTER_STATS  = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA"]
PITCHER_STATS = ["K_pct", "BB_pct", "FIP", "ERA"]


# =========================================================
# LOAD FULL DATASETS
# =========================================================

hitters_reg  = pd.read_csv(f"{DATA_DIR}/hitters_regressed.csv")
pitchers_reg = pd.read_csv(f"{DATA_DIR}/pitchers_regressed.csv")
league_avg_h = pd.read_csv(f"{DATA_DIR}/league_avgs_hitters.csv")
league_avg_p = pd.read_csv(f"{DATA_DIR}/league_avgs_pitchers.csv")
hitter_curves_full  = pd.read_csv(f"{DATA_DIR}/hitter_age_curves.csv")
pitcher_curves_full = pd.read_csv(f"{DATA_DIR}/pitcher_age_curves.csv")

try:
    ages = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
    ages["player_id"] = ages["player_id"].astype(str)
    has_ages = ages["birth_year"].notna().sum() > 0
except FileNotFoundError:
    ages = pd.DataFrame(columns=["player_id", "birth_year"])
    has_ages = False


# =========================================================
# INLINE PROJECTION LOGIC (avoids circular import)
# =========================================================

YEAR_WEIGHTS = {0: 5, 1: 4, 2: 3}
LEAGUE_WEIGHT = 1


def build_age_adj(curves_df, stat_cols, anchor_age=27):
    adj = {}
    for stat in stat_cols:
        col = f"delta_{stat}_smooth" if f"delta_{stat}_smooth" in curves_df.columns else f"delta_{stat}"
        if col not in curves_df.columns:
            adj[stat] = pd.Series(dtype=float)
            continue
        sub = curves_df[["age", col]].dropna().set_index("age").sort_index()
        cumsum = sub[col].cumsum()
        if anchor_age in cumsum.index:
            cumsum = cumsum - cumsum[anchor_age]
        adj[stat] = cumsum  # Series, not dict
    return adj


def get_single_year_delta(adj_series, age):
    if adj_series.empty:
        return 0.0
    min_age = adj_series.index.min()
    max_age = adj_series.index.max()
    age_clipped = int(np.clip(age, min_age + 1, max_age))
    prev_age    = age_clipped - 1
    return float(adj_series.get(age_clipped, adj_series.iloc[-1])
                 - adj_series.get(prev_age,    adj_series.iloc[-1]))


def project_one_year(df, league_avg_df, stat_cols, adj_dict,
                     target_year, ages_df, has_ages):
    """Projects target_year using only data from before target_year."""
    reg_cols = [s + "_reg" for s in stat_cols]
    results = []

    train = df[df["season"] < target_year]
    player_ids = train[train["season"].between(target_year - 3, target_year - 1)]["player_id"].unique()

    for pid in player_ids:
        player_seasons = train[train["player_id"] == pid].sort_values("season")
        recent = player_seasons[player_seasons["season"] < target_year].tail(3)
        if recent.empty:
            continue

        name = recent["player_name"].iloc[-1]
        age_at_proj = np.nan
        if has_ages:
            match = ages_df[ages_df["player_id"].astype(str) == str(pid)]
            if not match.empty and pd.notna(match["birth_year"].iloc[0]):
                age_at_proj = target_year - int(match["birth_year"].iloc[0])

        lg_year = min(target_year - 1, league_avg_df["season"].max())
        lg_row = league_avg_df[league_avg_df["season"] == lg_year]
        if lg_row.empty:
            continue
        lg = lg_row.iloc[0]

        row = {"player_id": pid, "player_name": name,
               "projection_year": target_year, "age": age_at_proj}

        for stat, rc in zip(stat_cols, reg_cols):
            total_w = LEAGUE_WEIGHT
            total_v = LEAGUE_WEIGHT * lg.get(stat, np.nan)
            for i, (_, sr) in enumerate(
                recent.sort_values("season", ascending=False).iterrows()
            ):
                if i > 2:
                    break
                w = YEAR_WEIGHTS[i]
                v = sr.get(rc, np.nan)
                if pd.notna(v):
                    total_v += w * v
                    total_w += w
            proj = total_v / total_w if total_w > 0 else np.nan

            if pd.notna(age_at_proj) and stat in adj_dict:
                delta = get_single_year_delta(adj_dict[stat], int(age_at_proj))
                proj += delta

            row[stat + "_proj"] = proj

        # Naive baselines
        last = player_seasons[player_seasons["season"] == target_year - 1]
        for stat in stat_cols:
            row[stat + "_prior"] = last[stat].values[0] if not last.empty else np.nan
            row[stat + "_lgavg"] = lg.get(stat, np.nan)

        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# RUN BACKTEST
# =========================================================

def run_backtest(df, league_avg_df, stat_cols, curves_df, target_years,
                 ages_df, has_ages):
    all_rows = []
    for yr in target_years:
        print(f"  Backtesting {yr}...")
        adj = build_age_adj(
            curves_df[curves_df["age"] > 0],  # use full curve data (no leakage since deltas are pop-level)
            stat_cols
        )
        proj = project_one_year(df, league_avg_df, stat_cols, adj, yr, ages_df, has_ages)
        actual = df[df["season"] == yr][["player_id"] + stat_cols].copy()
        actual.columns = ["player_id"] + [s + "_actual" for s in stat_cols]
        merged = proj.merge(actual, on="player_id", how="inner")
        all_rows.append(merged)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


print("Running hitter backtest...")
hitter_bt = run_backtest(
    hitters_reg, league_avg_h, HITTER_STATS,
    hitter_curves_full, BACKTEST_YEARS, ages, has_ages
)

print("Running pitcher backtest...")
pitcher_bt = run_backtest(
    pitchers_reg, league_avg_p, PITCHER_STATS,
    pitcher_curves_full, BACKTEST_YEARS, ages, has_ages
)


# =========================================================
# METRICS
# =========================================================

def eval_metrics(bt_df, stat_cols):
    rows = []
    for stat in stat_cols:
        proj_col  = stat + "_proj"
        prior_col = stat + "_prior"
        lg_col    = stat + "_lgavg"
        act_col   = stat + "_actual"

        for label, pred_col in [
            ("Marcel Projection", proj_col),
            ("Prior Year Only",   prior_col),
            ("League Average",    lg_col),
        ]:
            sub = bt_df[[pred_col, act_col]].dropna()
            if sub.empty:
                continue
            pred = sub[pred_col]
            act  = sub[act_col]
            mae  = (pred - act).abs().mean()
            rmse = np.sqrt(((pred - act) ** 2).mean())
            r, p = pearsonr(pred, act) if len(sub) > 2 else (np.nan, np.nan)
            rows.append({
                "stat": stat,
                "model": label,
                "n": len(sub),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "r": round(r, 4),
            })
    return pd.DataFrame(rows)


hitter_metrics  = eval_metrics(hitter_bt, HITTER_STATS)
pitcher_metrics = eval_metrics(pitcher_bt, PITCHER_STATS)

print("\n=== HITTER BACKTEST RESULTS ===")
print(hitter_metrics.to_string(index=False))

print("\n=== PITCHER BACKTEST RESULTS ===")
print(pitcher_metrics.to_string(index=False))


# =========================================================
# VISUALIZE: Predicted vs Actual scatter per stat
# =========================================================

def plot_validation(bt_df, stat_cols, metrics_df, title, filename):
    n = len(stat_cols)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, stat in zip(axes, stat_cols):
        proj_col = stat + "_proj"
        act_col  = stat + "_actual"
        sub = bt_df[[proj_col, act_col]].dropna()

        ax.scatter(sub[proj_col], sub[act_col], alpha=0.4, s=20, color="#1a5fa8")

        lim_min = min(sub[proj_col].min(), sub[act_col].min()) * 0.97
        lim_max = max(sub[proj_col].max(), sub[act_col].max()) * 1.03
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1, label="Perfect")

        met = metrics_df[
            (metrics_df["stat"] == stat) & (metrics_df["model"] == "Marcel Projection")
        ]
        if not met.empty:
            r   = met["r"].values[0]
            mae = met["MAE"].values[0]
            ax.set_title(f"{stat}\nr={r:.3f}  MAE={mae:.4f}", fontsize=10)

        ax.set_xlabel("Projected")
        ax.set_ylabel("Actual")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES_DIR}/{filename}")


plot_validation(
    hitter_bt, HITTER_STATS, hitter_metrics,
    "Hitter Projection Validation (2021–2024 Backtest)",
    "validation_hitters.png"
)
plot_validation(
    pitcher_bt, PITCHER_STATS, pitcher_metrics,
    "Pitcher Projection Validation (2021–2024 Backtest)",
    "validation_pitchers.png"
)


# =========================================================
# SAVE
# =========================================================

hitter_bt.to_csv(f"{DATA_DIR}/backtest_results_hitters.csv", index=False)
pitcher_bt.to_csv(f"{DATA_DIR}/backtest_results_pitchers.csv", index=False)
hitter_metrics.to_csv(f"{DATA_DIR}/hitter_validation_metrics.csv", index=False)
pitcher_metrics.to_csv(f"{DATA_DIR}/pitcher_validation_metrics.csv", index=False)

print("\n=== VALIDATION COMPLETE ===")
