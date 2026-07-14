"""
01_data_prep.py
---------------
Loads raw KBO data, engineers rate statistics, builds consecutive season pairs,
and computes per-season league averages. All downstream modules depend on the
outputs written here.

Outputs (data/):
  hitters_clean.csv       - per-season hitter records with rate stats
  pitchers_clean.csv      - per-season pitcher records with rate stats
  hitter_pairs.csv        - consecutive (year N, year N+1) hitter pairs
  pitcher_pairs.csv       - consecutive (year N, year N+1) pitcher pairs
  league_avgs_hitters.csv - per-season league mean rate stats (hitters)
  league_avgs_pitchers.csv- per-season league mean rate stats (pitchers)
"""

import pandas as pd
import numpy as np

RAW_FILE = "KBO_Player_Analytics_2016_2025.xlsx"
DATA_DIR = "data"

# Minimum playing time thresholds (same as source data qualification)
MIN_PA = 300
MIN_IP = 100


# =========================================================
# LOAD
# =========================================================

hitters = pd.read_excel(RAW_FILE, sheet_name="Hitters")
pitchers = pd.read_excel(RAW_FILE, sheet_name="Pitchers")

# Coerce numeric columns that may have loaded as object
for df in [hitters, pitchers]:
    for col in df.columns:
        if col not in ("player_id", "player_name", "team"):
            df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================================================
# HITTER RATE STATS
# =========================================================

h = hitters.copy()

h["BB_pct"]  = h["BB"] / h["PA"]
h["K_pct"]   = h["SO"] / h["PA"]
h["HR_pct"]  = h["HR"] / h["PA"]
h["ISO"]     = h["SLG"] - h["AVG"]

# wOBA approximation from counting stats (KBO league weights ~ MLB 2019)
# wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR) / PA
h["1B"] = h["H"] - h["2B"] - h["3B"] - h["HR"]
h["wOBA"] = (0.69 * h["BB"] + 0.72 * h["HP"] + 0.89 * h["1B"] + 1.27 * h["2B"] + 1.62 * h["3B"] + 2.10 * h["HR"]) / h["PA"]

h = h[h["PA"] >= MIN_PA].copy()
h = h.sort_values(["player_id", "season"]).reset_index(drop=True)


# =========================================================
# PITCHER RATE STATS
# =========================================================

p = pitchers.copy()

p["K_pct"]   = p["SO"] / p["TBF"]
p["BB_pct"]  = p["BB"] / p["TBF"]
p["HR_per9"] = p["HR"] * 9 / p["IP"]
p["K_minus_BB"] = p["K_pct"] - p["BB_pct"]

p = p[p["IP"] >= MIN_IP].copy()
p = p.sort_values(["player_id", "season"]).reset_index(drop=True)


# =========================================================
# CONSECUTIVE SEASON PAIRS
# =========================================================

HITTER_RATE_COLS = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA", "OBP", "SLG", "wRC+", "WAR", "oWAR"]
PITCHER_RATE_COLS = ["K_pct", "BB_pct", "HR_per9", "K_minus_BB", "ERA", "FIP", "WHIP", "WAR"]


def build_pairs(df, rate_cols, pt_col):
    """
    For each player, join season N to season N+1.
    Only keeps truly consecutive seasons (no gap years).
    """
    df = df.copy()
    next_yr = df[["player_id", "player_name", "season", pt_col] + rate_cols].copy()
    next_yr = next_yr.rename(columns={c: c + "_next" for c in rate_cols + [pt_col]})
    next_yr["season"] = next_yr["season"] - 1  # align to prior season

    pairs = df.merge(next_yr, on=["player_id", "player_name", "season"], how="inner")

    # Drop rows where age data will later be missing (placeholder — age added in 02)
    return pairs.reset_index(drop=True)


hitter_pairs = build_pairs(h, HITTER_RATE_COLS, "PA")
pitcher_pairs = build_pairs(p, PITCHER_RATE_COLS, "IP")


# =========================================================
# LEAGUE AVERAGES BY SEASON
# =========================================================

# PA-weighted means no heavy-usage players count more
def weighted_league_avg(df, rate_cols, weight_col):
    rows = []
    for season, grp in df.groupby("season"):
        row = {"season": season}
        w = grp[weight_col]
        for col in rate_cols:
            valid = grp[[col, weight_col]].dropna()
            if valid.empty:
                row[col] = np.nan
            else:
                row[col] = np.average(valid[col], weights=valid[weight_col])
        rows.append(row)
    return pd.DataFrame(rows)


league_avg_h = weighted_league_avg(h, HITTER_RATE_COLS, "PA")
league_avg_p = weighted_league_avg(p, PITCHER_RATE_COLS, "IP")


# =========================================================
# SAVE
# =========================================================

h.to_csv(f"{DATA_DIR}/hitters_clean.csv", index=False)
p.to_csv(f"{DATA_DIR}/pitchers_clean.csv", index=False)
hitter_pairs.to_csv(f"{DATA_DIR}/hitter_pairs.csv", index=False)
pitcher_pairs.to_csv(f"{DATA_DIR}/pitcher_pairs.csv", index=False)
league_avg_h.to_csv(f"{DATA_DIR}/league_avgs_hitters.csv", index=False)
league_avg_p.to_csv(f"{DATA_DIR}/league_avgs_pitchers.csv", index=False)

print("=== DATA PREP COMPLETE ===")
print(f"Hitters: {len(h)} season-rows, {h['player_id'].nunique()} unique players")
print(f"Pitchers: {len(p)} season-rows, {p['player_id'].nunique()} unique players")
print(f"Hitter consecutive pairs: {len(hitter_pairs)}")
print(f"Pitcher consecutive pairs: {len(pitcher_pairs)}")
print(f"\nHitter rate stats preview:")
print(h[["player_name", "season", "BB_pct", "K_pct", "HR_pct", "ISO", "wOBA"]].head(5).to_string(index=False))
print(f"\nLeague avg wOBA by season:")
print(league_avg_h[["season", "wOBA", "BB_pct", "K_pct"]].to_string(index=False))
