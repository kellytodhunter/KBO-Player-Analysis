"""
04_age_curves.py
----------------
Estimates KBO player aging curves using the delta method.

Method (standard in baseball research, e.g. Lichtman 2009):
  For each consecutive-season pair (year N → year N+1), compute
  delta = stat_next - stat_current. Group deltas by age at season N.
  The aging curve is the smoothed average delta at each age.

  Unlike raw averages, the delta method controls for survivorship bias
  because we only compare a player against themselves.

Requires: data/player_ages.csv (from 02_scrape_ages.py)
          data/hitter_pairs.csv, data/pitcher_pairs.csv

If player_ages.csv is not yet available, the script falls back to a
placeholder age (27) with a warning — aging adjustments will be zero
until real ages are loaded.

Outputs:
  data/hitter_age_curves.csv   - mean delta per age, per stat
  data/pitcher_age_curves.csv
  outputs/figures/hitter_age_curves.png
  outputs/figures/pitcher_age_curves.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import uniform_filter1d

DATA_DIR    = "data"
FIGURES_DIR = "outputs/figures"

HITTER_STATS  = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA", "OBP", "SLG"]
PITCHER_STATS = ["K_pct", "BB_pct", "HR_per9", "ERA", "FIP"]

PEAK_AGE_RANGE = (24, 30)   # ages used for smoothing anchor


# =========================================================
# LOAD
# =========================================================

hitter_pairs  = pd.read_csv(f"{DATA_DIR}/hitter_pairs.csv")
pitcher_pairs = pd.read_csv(f"{DATA_DIR}/pitcher_pairs.csv")

try:
    ages = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
    ages["player_id"] = ages["player_id"].astype(str)
    has_ages = ages["birth_year"].notna().sum() > 0
except FileNotFoundError:
    ages = pd.DataFrame(columns=["player_id", "birth_year"])
    has_ages = False

if not has_ages:
    warnings.warn(
        "player_ages.csv not found or empty. "
        "Run 02_scrape_ages.py first. Age curves will be flat (no adjustment)."
    )


def attach_age(pairs_df, ages_df, pt_col):
    """Add age = season - birth_year to a pairs dataframe."""
    pairs_df = pairs_df.copy()
    pairs_df["player_id"] = pairs_df["player_id"].astype(str)

    if ages_df.empty or not has_ages:
        pairs_df["age"] = 27  # neutral placeholder
        return pairs_df

    pairs_df = pairs_df.merge(
        ages_df[["player_id", "birth_year"]].dropna(),
        on="player_id", how="left"
    )
    pairs_df["age"] = pairs_df["season"] - pairs_df["birth_year"]

    missing = pairs_df["age"].isna().sum()
    if missing:
        warnings.warn(f"{missing} pairs missing age data; dropping them.")
        pairs_df = pairs_df.dropna(subset=["age"])

    pairs_df["age"] = pairs_df["age"].astype(int)
    return pairs_df


hitter_pairs  = attach_age(hitter_pairs,  ages, "PA")
pitcher_pairs = attach_age(pitcher_pairs, ages, "IP")


# =========================================================
# DELTA METHOD
# =========================================================

def compute_deltas(pairs_df, stat_cols, pt_col):
    """
    Compute year-over-year deltas for each stat, weighted by harmonic mean
    of PA/IP in year N and year N+1 (gives less weight to part-time seasons).
    """
    df = pairs_df.copy()
    for stat in stat_cols:
        next_col = stat + "_next"
        if next_col in df.columns:
            df[f"delta_{stat}"] = df[next_col] - df[stat]

    # Harmonic mean playing time weight
    pt_next = pt_col + "_next"
    if pt_next in df.columns:
        df["weight"] = 2 / (1 / df[pt_col] + 1 / df[pt_next])
    else:
        df["weight"] = df[pt_col]

    return df


def age_curve(deltas_df, stat_cols, min_n=5):
    """
    Aggregate deltas by age. Returns a DataFrame with columns:
    age, n, delta_{stat}, se_{stat} for each stat.
    """
    delta_cols = [f"delta_{s}" for s in stat_cols if f"delta_{s}" in deltas_df.columns]
    rows = []
    for age, grp in deltas_df.groupby("age"):
        row = {"age": age, "n": len(grp)}
        for col in delta_cols:
            valid = grp[[col, "weight"]].dropna()
            if len(valid) >= min_n:
                w = valid["weight"]
                d = valid[col]
                row[col] = np.average(d, weights=w)
                row[f"se_{col}"] = d.std() / np.sqrt(len(d))
            else:
                row[col] = np.nan
                row[f"se_{col}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("age").reset_index(drop=True)


def smooth_curve(series, window=3):
    """Apply uniform smoothing; preserve NaN edges."""
    arr = series.values.astype(float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return series
    smoothed = uniform_filter1d(np.where(valid, arr, 0), size=window)
    counts   = uniform_filter1d(valid.astype(float), size=window)
    result   = np.where(valid, smoothed / np.maximum(counts, 1e-9), np.nan)
    return pd.Series(result, index=series.index)


hitter_deltas  = compute_deltas(hitter_pairs, HITTER_STATS, "PA")
pitcher_deltas = compute_deltas(pitcher_pairs, PITCHER_STATS, "IP")

hitter_curves  = age_curve(hitter_deltas, HITTER_STATS)
pitcher_curves = age_curve(pitcher_deltas, PITCHER_STATS)

# Smooth each delta column
for stat in HITTER_STATS:
    col = f"delta_{stat}"
    if col in hitter_curves.columns:
        hitter_curves[col + "_smooth"] = smooth_curve(hitter_curves[col])

for stat in PITCHER_STATS:
    col = f"delta_{stat}"
    if col in pitcher_curves.columns:
        pitcher_curves[col + "_smooth"] = smooth_curve(pitcher_curves[col])


# =========================================================
# CUMULATIVE AGING ADJUSTMENT
# =========================================================
# Convert per-year deltas into a cumulative curve anchored at age 27.
# adjustment(age) = sum of deltas from 27 to age.

def cumulative_adjustment(curves_df, stat_cols, anchor_age=27):
    """
    Returns a dict: {stat: pd.Series indexed by age}.
    Positive = player is better at that age than at anchor_age.
    """
    adjustments = {}
    for stat in stat_cols:
        col = f"delta_{stat}_smooth"
        if col not in curves_df.columns:
            col = f"delta_{stat}"
        sub = curves_df[["age", col]].dropna()
        if sub.empty:
            continue
        sub = sub.set_index("age").sort_index()
        cumsum = sub[col].cumsum()
        # Anchor: subtract value at anchor_age so adjustment(anchor_age) = 0
        if anchor_age in cumsum.index:
            cumsum = cumsum - cumsum[anchor_age]
        adjustments[stat] = cumsum
    return adjustments


hitter_adj  = cumulative_adjustment(hitter_curves, HITTER_STATS)
pitcher_adj = cumulative_adjustment(pitcher_curves, PITCHER_STATS)


# =========================================================
# VISUALIZE
# =========================================================

def plot_curves(curves_df, adjustments, stat_cols, title, filename, stat_labels=None):
    n_stats = len(stat_cols)
    fig, axes = plt.subplots(
        nrows=(n_stats + 1) // 2, ncols=2,
        figsize=(12, 3.5 * ((n_stats + 1) // 2)),
        constrained_layout=True
    )
    axes = axes.flatten()
    stat_labels = stat_labels or {s: s for s in stat_cols}

    for i, stat in enumerate(stat_cols):
        ax = axes[i]
        col_smooth = f"delta_{stat}_smooth"
        col_raw    = f"delta_{stat}"

        if col_raw in curves_df.columns:
            ax.bar(
                curves_df["age"], curves_df[col_raw],
                color="#aec6e8", alpha=0.5, label="Observed delta", zorder=1
            )
        if col_smooth in curves_df.columns:
            ax.plot(
                curves_df["age"], curves_df[col_smooth],
                color="#1a5fa8", lw=2, label="Smoothed", zorder=2
            )

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvspan(PEAK_AGE_RANGE[0], PEAK_AGE_RANGE[1], alpha=0.08,
                   color="gold", label="Peak range")
        ax.set_title(stat_labels.get(stat, stat), fontsize=11, fontweight="bold")
        ax.set_xlabel("Age")
        ax.set_ylabel("Δ per season")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.legend(fontsize=7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIGURES_DIR}/{filename}")


HITTER_LABELS = {
    "BB_pct": "Walk Rate (BB%)",
    "K_pct":  "Strikeout Rate (K%)",
    "HR_pct": "HR Rate (HR/PA)",
    "ISO":    "Isolated Power (ISO)",
    "wOBA":   "wOBA",
    "OBP":    "OBP",
    "SLG":    "SLG",
}
PITCHER_LABELS = {
    "K_pct":   "Strikeout Rate (K%)",
    "BB_pct":  "Walk Rate (BB%)",
    "HR_per9": "HR/9",
    "ERA":     "ERA",
    "FIP":     "FIP",
}

plot_curves(hitter_curves, hitter_adj, HITTER_STATS,
            "KBO Hitter Aging Curves (Delta Method)", "hitter_age_curves.png", HITTER_LABELS)
plot_curves(pitcher_curves, pitcher_adj, PITCHER_STATS,
            "KBO Pitcher Aging Curves (Delta Method)", "pitcher_age_curves.png", PITCHER_LABELS)


# =========================================================
# SAVE
# =========================================================

hitter_curves.to_csv(f"{DATA_DIR}/hitter_age_curves.csv", index=False)
pitcher_curves.to_csv(f"{DATA_DIR}/pitcher_age_curves.csv", index=False)

print("\n=== AGE CURVES COMPLETE ===")
if not has_ages:
    print("WARNING: No real age data — curves are flat. Run 02_scrape_ages.py.")
else:
    print(f"Hitter curve ages: {hitter_curves['age'].min()}–{hitter_curves['age'].max()}")
    print(f"Pitcher curve ages: {pitcher_curves['age'].min()}–{pitcher_curves['age'].max()}")
    print("\nSample hitter wOBA deltas by age:")
    print(hitter_curves[["age", "n", "delta_wOBA"]].dropna().to_string(index=False))
