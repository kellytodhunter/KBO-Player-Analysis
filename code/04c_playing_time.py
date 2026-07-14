"""
04c_playing_time.py
-------------------
Projects playing time (PA for hitters, IP for pitchers) for the next season.

Method:
  PA/IP projection = weighted average of prior 3 seasons (5/4/3),
  regressed toward the median qualified playing time, then adjusted
  downward for age (older players accumulate fewer PA/IP on average).

Playing time projections are used by 05_projections.py to rank players
by counting-stat value (e.g. projected HR = HR_pct_proj × PA_proj).

Outputs:
  data/hitter_pt_projections.csv
  data/pitcher_pt_projections.csv
  outputs/figures/pt_age_curve.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR    = "data"
FIGURES_DIR = "outputs/figures"

YEAR_WEIGHTS = {0: 5, 1: 4, 2: 3}
LEAGUE_WEIGHT = 1

# Regression anchors — median qualified PA/IP
MEDIAN_QUALIFIED_PA = 500
MEDIAN_QUALIFIED_IP = 155
M_PA = 200   # PA regression coefficient
M_IP = 60    # IP regression coefficient (pitchers more variable)


# =========================================================
# LOAD
# =========================================================

hitters  = pd.read_csv(f"{DATA_DIR}/hitters_clean.csv")
pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_clean.csv")

try:
    ages = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
    ages["player_id"] = ages["player_id"].astype(str)
    has_ages = ages["birth_year"].notna().sum() > 0
except FileNotFoundError:
    ages = pd.DataFrame(columns=["player_id", "birth_year"])
    has_ages = False


# =========================================================
# PLAYING TIME AGE CURVE
# =========================================================

def pt_age_curve(df, pt_col, ages_df):
    """Compute mean PA/IP by age using delta method on consecutive pairs."""
    df = df.copy()
    df["player_id"] = df["player_id"].astype(str)

    if ages_df.empty or not has_ages:
        return pd.DataFrame()

    df = df.merge(ages_df[["player_id", "birth_year"]].dropna(), on="player_id", how="left")
    df["age"] = df["season"] - df["birth_year"]

    rows = []
    for age, grp in df.groupby("age"):
        if len(grp) >= 5:
            rows.append({"age": int(age), "mean_pt": grp[pt_col].mean(), "n": len(grp)})

    return pd.DataFrame(rows).sort_values("age")


h_pt_curve = pt_age_curve(hitters, "PA", ages)
p_pt_curve = pt_age_curve(pitchers, "IP", ages)


# =========================================================
# AGE PLAYING TIME SCALAR
# =========================================================

def build_pt_scalar(pt_curve, pt_col, anchor_age=27):
    """
    Returns dict {age: scalar} where scalar is the ratio of mean PT at that
    age relative to the anchor age. Used to adjust raw PT projection.
    """
    if pt_curve.empty:
        return {}
    sub = pt_curve.set_index("age")["mean_pt"]
    if anchor_age not in sub.index:
        anchor_age = int(sub.index[sub.index >= anchor_age][0]) if any(sub.index >= anchor_age) else sub.index[0]
    anchor_val = sub[anchor_age]
    return (sub / anchor_val).to_dict()


h_pt_scalar = build_pt_scalar(h_pt_curve, "PA")
p_pt_scalar = build_pt_scalar(p_pt_curve, "IP")


# =========================================================
# PROJECT PLAYING TIME
# =========================================================

def project_pt(df, pt_col, m, median_qualified, pt_scalar, ages_df, target_year):
    results = []
    player_ids = df[df["season"].between(target_year - 3, target_year - 1)]["player_id"].unique()

    for pid in player_ids:
        psn = df[df["player_id"] == pid].sort_values("season")
        recent = psn[psn["season"] < target_year].tail(3)
        if recent.empty:
            continue

        name = recent["player_name"].iloc[-1]

        # Weighted average of PA/IP
        total_w, total_v = LEAGUE_WEIGHT, LEAGUE_WEIGHT * median_qualified
        for i, (_, r) in enumerate(recent.sort_values("season", ascending=False).iterrows()):
            if i > 2:
                break
            w = YEAR_WEIGHTS[i]
            total_v += w * r[pt_col]
            total_w += w
        pt_proj = total_v / total_w

        # Regress toward median qualified
        n = recent[pt_col].mean()
        shrink = n / (n + m)
        pt_proj = shrink * pt_proj + (1 - shrink) * median_qualified

        # Age scalar
        age = np.nan
        if has_ages:
            match = ages_df[ages_df["player_id"].astype(str) == str(pid)]
            if not match.empty and pd.notna(match["birth_year"].iloc[0]):
                age = target_year - int(match["birth_year"].iloc[0])

        if pd.notna(age) and pt_scalar:
            scalar = pt_scalar.get(
                int(np.clip(age, min(pt_scalar), max(pt_scalar))), 1.0
            )
            pt_proj *= scalar

        results.append({
            "player_id": pid, "player_name": name,
            "projection_year": target_year, "age": age,
            f"{pt_col}_proj": round(pt_proj),
        })

    return pd.DataFrame(results)


TARGET_YEAR = 2026

h_pt = project_pt(hitters, "PA", M_PA, MEDIAN_QUALIFIED_PA, h_pt_scalar, ages, TARGET_YEAR)
p_pt = project_pt(pitchers, "IP", M_IP, MEDIAN_QUALIFIED_IP, p_pt_scalar, ages, TARGET_YEAR)


# =========================================================
# VISUALIZE PT AGE CURVE
# =========================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

if not h_pt_curve.empty:
    ax1.bar(h_pt_curve["age"], h_pt_curve["mean_pt"], color="#aec6e8", alpha=0.6)
    ax1.set_title("Hitter PA by Age", fontweight="bold")
    ax1.set_xlabel("Age"); ax1.set_ylabel("Mean PA")

if not p_pt_curve.empty:
    ax2.bar(p_pt_curve["age"], p_pt_curve["mean_pt"], color="#f4a460", alpha=0.6)
    ax2.set_title("Pitcher IP by Age", fontweight="bold")
    ax2.set_xlabel("Age"); ax2.set_ylabel("Mean IP")

fig.suptitle("KBO Playing Time by Age (Qualified Players)", fontsize=12, fontweight="bold")
plt.savefig(f"{FIGURES_DIR}/pt_age_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIGURES_DIR}/pt_age_curve.png")


# =========================================================
# SAVE
# =========================================================

h_pt.to_csv(f"{DATA_DIR}/hitter_pt_projections.csv", index=False)
p_pt.to_csv(f"{DATA_DIR}/pitcher_pt_projections.csv", index=False)

print("\n=== PLAYING TIME PROJECTIONS ===")
print(f"\nTop 10 hitters by projected PA ({TARGET_YEAR}):")
print(h_pt.sort_values("PA_proj", ascending=False)[["player_name","age","PA_proj"]].head(10).to_string(index=False))
print(f"\nTop 10 pitchers by projected IP ({TARGET_YEAR}):")
print(p_pt.sort_values("IP_proj", ascending=False)[["player_name","age","IP_proj"]].head(10).to_string(index=False))
