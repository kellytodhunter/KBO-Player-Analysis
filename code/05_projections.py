"""
05_projections.py
-----------------
Builds multi-year weighted projections for KBO hitters and pitchers.

Method (Marcel-style, adapted for KBO):
  1. Take regressed rate stats from years T, T-1, T-2 (weights 5/4/3).
  2. Blend with league average at weight 1 (additional reliability shrinkage).
  3. Apply cumulative age adjustment from 04_age_curves.py.
  4. Output projected stats for each player in year T+1.

Usage:
  python src/05_projections.py                    # projects 2025 players → 2026
  python src/05_projections.py --target-year 2023 # backtesting mode

Outputs:
  data/hitter_projections_{year}.csv
  data/pitcher_projections_{year}.csv
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd

DATA_DIR = "data"

HITTER_STATS  = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA", "OBP", "SLG"]
PITCHER_STATS = ["K_pct", "BB_pct", "HR_per9", "ERA", "FIP"]

YEAR_WEIGHTS = {0: 5, 1: 4, 2: 3}  # T, T-1, T-2 (index = years ago)
LEAGUE_WEIGHT = 1                   # extra shrinkage toward league average


# =========================================================
# LOAD
# =========================================================

hitters  = pd.read_csv(f"{DATA_DIR}/hitters_regressed.csv")
pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_regressed.csv")
league_avg_h = pd.read_csv(f"{DATA_DIR}/league_avgs_hitters.csv")
league_avg_p = pd.read_csv(f"{DATA_DIR}/league_avgs_pitchers.csv")
hitter_curves  = pd.read_csv(f"{DATA_DIR}/hitter_age_curves.csv")
pitcher_curves = pd.read_csv(f"{DATA_DIR}/pitcher_age_curves.csv")

try:
    ages = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
    ages["player_id"] = ages["player_id"].astype(str)
    has_ages = ages["birth_year"].notna().sum() > 0
except FileNotFoundError:
    ages = pd.DataFrame(columns=["player_id", "birth_year"])
    has_ages = False


# =========================================================
# AGE ADJUSTMENT LOOKUP
# =========================================================

def build_age_adjustment(curves_df, stat_cols, anchor_age=27):
    """
    Returns dict: {stat: pd.Series indexed by age (cumulative delta from anchor)}.
    Stored as Series so we can safely clip age lookups to the data range.
    Positive = better than at anchor_age.
    """
    adj = {}
    for stat in stat_cols:
        col_smooth = f"delta_{stat}_smooth"
        col_raw    = f"delta_{stat}"
        col = col_smooth if col_smooth in curves_df.columns else col_raw
        if col not in curves_df.columns:
            adj[stat] = pd.Series(dtype=float)
            continue
        sub = curves_df[["age", col]].dropna().set_index("age").sort_index()
        cumsum = sub[col].cumsum()
        if anchor_age in cumsum.index:
            cumsum = cumsum - cumsum[anchor_age]
        adj[stat] = cumsum  # keep as Series, NOT dict
    return adj


def get_single_year_delta(adj_series, age):
    """
    Return the expected change from age-1 → age, clipped to data range.
    Avoids the bug where out-of-range ages default to 0 and create
    a large spurious delta against the final cumulative value.
    """
    if adj_series.empty:
        return 0.0
    min_age = adj_series.index.min()
    max_age = adj_series.index.max()
    age_clipped = int(np.clip(age, min_age + 1, max_age))
    prev_age    = age_clipped - 1
    return float(adj_series.get(age_clipped, adj_series.iloc[-1])
                 - adj_series.get(prev_age,    adj_series.iloc[-1]))


hitter_adj  = build_age_adjustment(hitter_curves, HITTER_STATS)
pitcher_adj = build_age_adjustment(pitcher_curves, PITCHER_STATS)


# =========================================================
# MULTI-YEAR WEIGHTED PROJECTION
# =========================================================

def project_players(df, league_avg_df, stat_cols, adj_dict, target_year,
                    ages_df, pt_col, reg_suffix="_reg"):
    """
    Projects target_year stats for all players who appeared in at least
    one of the three prior seasons.

    Returns a DataFrame with one row per player.
    """
    regressed_cols = [s + reg_suffix for s in stat_cols]
    results = []

    player_ids = df[df["season"].between(target_year - 3, target_year - 1)][
        "player_id"
    ].unique()

    for pid in player_ids:
        player_seasons = df[df["player_id"] == pid].sort_values("season")

        # Collect up to 3 most recent seasons before target_year
        recent = player_seasons[player_seasons["season"] < target_year].tail(3)
        if recent.empty:
            continue

        name = recent["player_name"].iloc[-1]

        # Age at target_year
        age_at_proj = np.nan
        if has_ages:
            match = ages_df[ages_df["player_id"].astype(str) == str(pid)]
            if not match.empty and pd.notna(match["birth_year"].iloc[0]):
                age_at_proj = target_year - int(match["birth_year"].iloc[0])

        # League average for target_year - 1 (most recent known)
        lg_year = min(target_year - 1, league_avg_df["season"].max())
        lg_row = league_avg_df[league_avg_df["season"] == lg_year]
        if lg_row.empty:
            continue
        lg = lg_row.iloc[0]

        row = {
            "player_id": pid,
            "player_name": name,
            "projection_year": target_year,
            "age": age_at_proj,
            "n_seasons": len(recent),
        }
        row[pt_col + "_proj"] = recent[pt_col].mean()  # naive PA/IP estimate

        for stat, rc in zip(stat_cols, regressed_cols):
            # Build weighted average across available years
            total_weight = LEAGUE_WEIGHT
            weighted_sum = LEAGUE_WEIGHT * lg.get(stat, np.nan)

            for i, (_, season_row) in enumerate(
                recent.sort_values("season", ascending=False).iterrows()
            ):
                years_ago = i  # 0 = most recent
                if years_ago > 2:
                    break
                w = YEAR_WEIGHTS[years_ago]
                val = season_row.get(rc, np.nan)
                if pd.notna(val):
                    weighted_sum += w * val
                    total_weight += w

            if total_weight == 0 or pd.isna(weighted_sum):
                row[stat + "_proj"] = np.nan
                continue

            proj = weighted_sum / total_weight

            # Age adjustment: add expected change from (age-1) → age
            if pd.notna(age_at_proj) and stat in adj_dict:
                delta = get_single_year_delta(adj_dict[stat], int(age_at_proj))
                proj += delta

            row[stat + "_proj"] = proj

        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# RUN
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--target-year", type=int, default=2026)
args, _ = parser.parse_known_args()
target_year = args.target_year

print(f"Projecting for season: {target_year}")

hitter_proj = project_players(
    hitters, league_avg_h, HITTER_STATS, hitter_adj,
    target_year, ages, "PA"
)
pitcher_proj = project_players(
    pitchers, league_avg_p, PITCHER_STATS, pitcher_adj,
    target_year, ages, "IP"
)


# =========================================================
# PRINT TOP PROJECTIONS
# =========================================================

print(f"\n=== TOP 15 HITTERS BY PROJECTED wOBA ({target_year}) ===")
cols = ["player_name", "age", "n_seasons", "wOBA_proj", "BB_pct_proj", "K_pct_proj", "HR_pct_proj"]
cols = [c for c in cols if c in hitter_proj.columns]
print(hitter_proj.sort_values("wOBA_proj", ascending=False)[cols].head(15).to_string(index=False))

print(f"\n=== TOP 15 PITCHERS BY PROJECTED FIP ({target_year}) ===")
cols = ["player_name", "age", "n_seasons", "FIP_proj", "K_pct_proj", "BB_pct_proj", "ERA_proj"]
cols = [c for c in cols if c in pitcher_proj.columns]
print(pitcher_proj.sort_values("FIP_proj", ascending=True)[cols].head(15).to_string(index=False))


# =========================================================
# SAVE
# =========================================================

hitter_proj.to_csv(f"{DATA_DIR}/hitter_projections_{target_year}.csv", index=False)
pitcher_proj.to_csv(f"{DATA_DIR}/pitcher_projections_{target_year}.csv", index=False)

print(f"\n=== PROJECTIONS COMPLETE ===")
print(f"Hitters projected: {len(hitter_proj)}")
print(f"Pitchers projected: {len(pitcher_proj)}")
if not has_ages:
    print("NOTE: Age adjustments disabled — run 02_scrape_ages.py to enable.")
