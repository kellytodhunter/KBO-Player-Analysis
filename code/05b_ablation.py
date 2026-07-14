"""
05b_ablation.py
---------------
Ablation study: quantifies the contribution of each Marcel component.

Compares four model variants across the 2021-2024 backtest window:
  1. League average only          (worst-case floor)
  2. Prior year only              (naive baseline)
  3. Marcel without age adjustment (multi-year weighting only)
  4. Marcel with age adjustment   (full system)

For each variant and each stat, reports MAE, RMSE, and Pearson r.
The difference between variants 3 and 4 isolates the age adjustment contribution.
The difference between variants 2 and 3 isolates the multi-year weighting contribution.

Outputs:
  data/ablation_results.csv
  outputs/figures/ablation_hitters.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

DATA_DIR    = "data"
FIGURES_DIR = "outputs/figures"

HITTER_STATS = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA"]

STAT_LABELS = {
    "BB_pct": "Walk rate (BB%)",
    "K_pct":  "Strikeout rate (K%)",
    "HR_pct": "HR rate (HR/PA)",
    "ISO":    "Isolated power (ISO)",
    "wOBA":   "wOBA",
}


# =========================================================
# LOAD BACKTEST DATA
# =========================================================

bt = pd.read_csv(f"{DATA_DIR}/backtest_results_hitters.csv")

# We need to reconstruct "Marcel without age" — we have the full Marcel projection
# and the actual outcomes. To isolate age contribution we need to re-run projections
# without age adjustment. Instead, we derive it from what we stored:
# Marcel_no_age = Marcel_proj - age_delta
# We don't have age_delta stored directly, so we rebuild from the age curves.

hitters_reg   = pd.read_csv(f"{DATA_DIR}/hitters_regressed.csv")
league_avg_h  = pd.read_csv(f"{DATA_DIR}/league_avgs_hitters.csv")
hitter_curves = pd.read_csv(f"{DATA_DIR}/hitter_age_curves.csv")
ages          = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
ages["player_id"] = ages["player_id"].astype(str)

YEAR_WEIGHTS  = {0: 5, 1: 4, 2: 3}
LEAGUE_WEIGHT = 1
BACKTEST_YEARS = [2021, 2022, 2023, 2024]


# =========================================================
# REBUILD AGE ADJUSTMENT LOOKUP
# =========================================================

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
        adj[stat] = cumsum
    return adj

def get_delta(adj_series, age):
    if adj_series.empty or pd.isna(age):
        return 0.0
    min_a, max_a = adj_series.index.min(), adj_series.index.max()
    a = int(np.clip(age, min_a + 1, max_a))
    return float(adj_series.get(a, adj_series.iloc[-1])
                 - adj_series.get(a - 1, adj_series.iloc[-1]))

age_adj = build_age_adj(hitter_curves, HITTER_STATS)


# =========================================================
# GENERATE PROJECTIONS FOR ALL FOUR VARIANTS
# =========================================================

def project_one(df, league_avg_df, stat_cols, target_year, ages_df, age_adj, use_age):
    reg_cols = [s + "_reg" for s in stat_cols]
    train    = df[df["season"] < target_year]
    results  = []

    pids = train[train["season"].between(target_year - 3, target_year - 1)]["player_id"].unique()

    for pid in pids:
        recent = train[train["player_id"] == pid].sort_values("season").tail(3)
        if recent.empty:
            continue

        name   = recent["player_name"].iloc[-1]
        lg_row = league_avg_df[league_avg_df["season"] == target_year - 1]
        if lg_row.empty:
            continue
        lg = lg_row.iloc[0]

        age = np.nan
        match = ages_df[ages_df["player_id"].astype(str) == str(pid)]
        if not match.empty and pd.notna(match["birth_year"].iloc[0]):
            age = target_year - int(match["birth_year"].iloc[0])

        row = {"player_id": pid, "player_name": name,
               "projection_year": target_year, "age": age}

        for stat, rc in zip(stat_cols, reg_cols):
            total_w = LEAGUE_WEIGHT
            total_v = LEAGUE_WEIGHT * lg.get(stat, np.nan)
            for i, (_, sr) in enumerate(recent.sort_values("season", ascending=False).iterrows()):
                if i > 2: break
                v = sr.get(rc, np.nan)
                if pd.notna(v):
                    total_v += YEAR_WEIGHTS[i] * v
                    total_w += YEAR_WEIGHTS[i]
            proj = total_v / total_w if total_w else np.nan

            if use_age:
                proj += get_delta(age_adj[stat], age)

            row[stat + "_proj"] = proj

            # Baselines
            last = recent[recent["season"] == target_year - 1]
            row[stat + "_prior"] = last[stat].values[0] if not last.empty else np.nan
            row[stat + "_lgavg"] = lg.get(stat, np.nan)

        results.append(row)
    return pd.DataFrame(results)


all_with_age    = []
all_without_age = []

for yr in BACKTEST_YEARS:
    wa  = project_one(hitters_reg, league_avg_h, HITTER_STATS, yr, ages, age_adj, use_age=True)
    woa = project_one(hitters_reg, league_avg_h, HITTER_STATS, yr, ages, age_adj, use_age=False)
    actual = hitters_reg[hitters_reg["season"] == yr][["player_id"] + HITTER_STATS].rename(
        columns={s: s + "_actual" for s in HITTER_STATS})
    wa  = wa.merge(actual,  on="player_id", how="inner")
    woa = woa.merge(actual, on="player_id", how="inner")
    all_with_age.append(wa)
    all_without_age.append(woa)

bt_with    = pd.concat(all_with_age,    ignore_index=True)
bt_without = pd.concat(all_without_age, ignore_index=True)


# =========================================================
# COMPUTE METRICS FOR ALL VARIANTS
# =========================================================

def metrics(pred, actual):
    sub = pd.DataFrame({"p": pred, "a": actual}).dropna()
    if len(sub) < 3:
        return dict(r=np.nan, MAE=np.nan, RMSE=np.nan, n=len(sub))
    mae  = (sub.p - sub.a).abs().mean()
    rmse = np.sqrt(((sub.p - sub.a)**2).mean())
    r, _ = pearsonr(sub.p, sub.a)
    return dict(r=round(r,4), MAE=round(mae,4), RMSE=round(rmse,4), n=len(sub))

rows = []
for stat in HITTER_STATS:
    act = bt_with[stat + "_actual"]
    for label, pred in [
        ("League average",           bt_with[stat + "_lgavg"]),
        ("Prior year only",          bt_with[stat + "_prior"]),
        ("Marcel (no age adj)",      bt_without[stat + "_proj"]),
        ("Marcel (full system)",     bt_with[stat + "_proj"]),
    ]:
        m = metrics(pred, act)
        rows.append({"stat": stat, "model": label, **m})

ablation_df = pd.DataFrame(rows)
ablation_df.to_csv(f"{DATA_DIR}/ablation_results.csv", index=False)

print("=== ABLATION STUDY RESULTS ===\n")
for stat in HITTER_STATS:
    print(f"--- {STAT_LABELS[stat]} ---")
    sub = ablation_df[ablation_df["stat"] == stat][["model","MAE","r","n"]]
    print(sub.to_string(index=False))
    # Quantify age contribution
    no_age = sub[sub["model"]=="Marcel (no age adj)"]["r"].values[0]
    full   = sub[sub["model"]=="Marcel (full system)"]["r"].values[0]
    delta  = full - no_age
    print(f"  Age adjustment contribution to r: {'+' if delta>=0 else ''}{delta:.4f}\n")


# =========================================================
# FIGURE: MODEL COMPARISON BY STAT
# =========================================================

MODELS   = ["League average", "Prior year only", "Marcel (no age adj)", "Marcel (full system)"]
COLORS   = ["#D3D1C7", "#888780", "#85B7EB", "#185FA5"]
PATTERNS = ["//", "..", "", ""]

fig, axes = plt.subplots(1, len(HITTER_STATS), figsize=(15, 5), constrained_layout=True)

for ax, stat in zip(axes, HITTER_STATS):
    sub = ablation_df[ablation_df["stat"] == stat].set_index("model")
    rs  = [sub.loc[m, "r"] if m in sub.index else 0 for m in MODELS]
    maes = [sub.loc[m, "MAE"] if m in sub.index else 0 for m in MODELS]

    x   = np.arange(len(MODELS))
    bars = ax.bar(x, rs, color=COLORS, width=0.65, zorder=2)

    # Annotate r value
    for bar, r_val in zip(bars, rs):
        if not np.isnan(r_val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{r_val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(STAT_LABELS[stat], fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Lg avg", "Prior yr", "Marcel\n(no age)", "Marcel\n(full)"], fontsize=8)
    ax.set_ylabel("Pearson r with next-season value" if stat == HITTER_STATS[0] else "")
    ax.set_ylim(-0.15, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", lw=0.4, color="#e0e0e0", zorder=0)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("KBO Hitter Projection Ablation Study — Pearson r (2021–2024 backtest)",
             fontsize=13, fontweight="bold")

legend_elements = [plt.Rectangle((0,0),1,1, color=c, label=m)
                   for c, m in zip(COLORS, MODELS)]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))

plt.savefig(f"{FIGURES_DIR}/ablation_hitters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {FIGURES_DIR}/ablation_hitters.png")
print("=== ABLATION COMPLETE ===")
