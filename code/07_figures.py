"""
07_figures.py
-------------
Publication-quality figures for the KBO projection system paper.

All figures follow journal conventions:
  - Minimal chartjunk (no unnecessary grid lines, borders, legends)
  - Labeled axes with units
  - Consistent color palette (blue = Marcel, gray = baselines, red = bad)
  - 150 dpi for inline, 300 dpi for final submission

Figures produced:
  fig1_age_curves.png        - KBO hitter aging curves (delta method), 7-panel
  fig2_validation_scatter.png- Predicted vs actual wOBA with baseline comparison
  fig3_model_comparison.png  - MAE comparison across all models and stats
  fig4_ablation.png          - Component contribution analysis (ablation)
  fig5_regression_tuning.png - m-value cross-validation curves
  fig6_projections_2026.png  - 2026 hitter projections ranked by wOBA
  fig7_pt_by_age.png         - Playing time (PA) by age
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

DATA_DIR    = "data"
OUT_DIR     = "outputs/figures"

# ---- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.family":      ["Nanum Gothic", "Apple SD Gothic Neo", "Arial Unicode MS", "DejaVu Sans"],
    "font.size":        11,
    "axes.titlesize":   11,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e8e8e8",
    "grid.linewidth":   0.5,
    "figure.dpi":       150,
    "savefig.dpi":      250,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

BLUE   = "#185FA5"
LGRAY  = "#888780"
DGRAY  = "#444441"
RED    = "#A32D2D"
LBLUE  = "#85B7EB"
GREEN  = "#3B6D11"

STAT_LABELS = {
    "BB_pct": "Walk rate (BB%)",
    "K_pct":  "Strikeout rate (K%)",
    "HR_pct": "HR rate (HR/PA)",
    "ISO":    "Isolated power (ISO)",
    "wOBA":   "wOBA",
    "OBP":    "OBP",
    "SLG":    "SLG",
}

# ---- Load data -----------------------------------------------------------
hitter_curves = pd.read_csv(f"{DATA_DIR}/hitter_age_curves.csv")
pitcher_curves= pd.read_csv(f"{DATA_DIR}/pitcher_age_curves.csv")
bt_h          = pd.read_csv(f"{DATA_DIR}/backtest_results_hitters.csv")
bt_p          = pd.read_csv(f"{DATA_DIR}/backtest_results_pitchers.csv")
ablation      = pd.read_csv(f"{DATA_DIR}/ablation_results.csv")
tuning_h      = pd.read_csv(f"{DATA_DIR}/m_value_tuning_hitters.png".replace(
                    "outputs/figures/","data/").replace(".png",""))  if False else None
proj_2026     = pd.read_csv(f"{DATA_DIR}/hitter_projections_2026.csv")
metrics_h     = pd.read_csv(f"{DATA_DIR}/hitter_validation_metrics.csv")
metrics_p     = pd.read_csv(f"{DATA_DIR}/pitcher_validation_metrics.csv")
pt_h          = pd.read_csv(f"{DATA_DIR}/hitter_pt_projections.csv")
ages          = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
hitters_clean = pd.read_csv(f"{DATA_DIR}/hitters_clean.csv")


# ==========================================================================
# FIG 1 — KBO Hitter Aging Curves
# ==========================================================================

HITTER_CURVE_STATS = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA", "OBP", "SLG"]

fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
fig.suptitle("")  # no top title
axes = axes.flatten()

for i, stat in enumerate(HITTER_CURVE_STATS):
    ax = axes[i]
    col_raw    = f"delta_{stat}"
    col_smooth = f"delta_{stat}_smooth"

    sub = hitter_curves.dropna(subset=[col_raw])

    ax.bar(sub["age"], sub[col_raw], color=LBLUE, alpha=0.5,
           label="Observed delta", zorder=1, width=0.8)

    if col_smooth in hitter_curves.columns:
        smooth = hitter_curves.dropna(subset=[col_smooth])
        ax.plot(smooth["age"], smooth[col_smooth], color=BLUE,
                lw=2.5, label="Smoothed", zorder=3)

    ax.axhline(0, color=DGRAY, lw=0.8, ls="--", zorder=2)
    ax.axvspan(24, 30, alpha=0.07, color="#BA7517", label="Peak range (24–30)")

    ax.set_title(STAT_LABELS.get(stat, stat), fontsize=10, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Δ per season" if i % 4 == 0 else "")
    ax.set_xlim(18, 42)

    if i == 0:
        ax.legend(fontsize=7, frameon=False)

    # Annotate n per age bucket
    for _, row in sub.iterrows():
        if 20 <= row["age"] <= 40 and row.get("n", 0) >= 5:
            ax.annotate(str(int(row["n"])),
                        xy=(row["age"], 0),
                        fontsize=6, ha="center", color=LGRAY,
                        xytext=(0, -10), textcoords="offset points")

axes[-1].set_visible(False)  # hide empty 8th panel

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_age_curves.png")
plt.close()
print("Saved fig1_age_curves.png")


# ==========================================================================
# FIG 2 — Validation Scatter: Projected vs Actual wOBA
# ==========================================================================

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.suptitle("")

panels = [
    ("wOBA_lgavg",  "League average",    LGRAY),
    ("wOBA_prior",  "Prior year only",   LGRAY),
    ("wOBA_proj",   "Marcel (full)",     BLUE),
]

bt_clean = bt_h.dropna(subset=["wOBA_actual", "wOBA_proj", "wOBA_prior", "wOBA_lgavg"])

lim = (bt_clean["wOBA_actual"].min() - 0.01,
       bt_clean["wOBA_actual"].max() + 0.01)

for ax, (pred_col, label, color) in zip(axes, panels):
    sub = bt_clean[[pred_col, "wOBA_actual"]].dropna()
    r, _ = pearsonr(sub[pred_col], sub["wOBA_actual"])
    mae  = (sub[pred_col] - sub["wOBA_actual"]).abs().mean()

    ax.scatter(sub[pred_col], sub["wOBA_actual"],
               alpha=0.4, s=22, color=color, linewidths=0)
    ax.plot(lim, lim, "--", color=DGRAY, lw=1, label="Perfect prediction")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Projected wOBA")
    ax.set_ylabel("Actual wOBA" if axes.tolist().index(ax) == 0 else "")
    ax.set_title(f"{label}\nr = {r:.3f}  |  MAE = {mae:.4f}",
                 fontsize=10, fontweight="bold")
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_validation_scatter.png")
plt.close()
print("Saved fig2_validation_scatter.png")


# ==========================================================================
# FIG 3 — Model comparison: MAE across all stats
# ==========================================================================

HITTER_STATS = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA"]
MODELS_ORDER = ["League Average", "Prior Year Only", "Marcel Projection"]
MODEL_COLORS = [LGRAY, "#b0b0b0", BLUE]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("")

for ax, metric, ylabel in zip(axes, ["MAE", "r"], ["Mean absolute error (lower = better)",
                                                     "Pearson r (higher = better)"]):
    x = np.arange(len(HITTER_STATS))
    width = 0.25
    for j, (model, color) in enumerate(zip(MODELS_ORDER, MODEL_COLORS)):
        sub = metrics_h[metrics_h["model"] == model]
        vals = []
        for stat in HITTER_STATS:
            row = sub[sub["stat"] == stat]
            vals.append(row[metric].values[0] if not row.empty else 0)
        bars = ax.bar(x + j*width - width, vals, width, label=model,
                      color=color, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([STAT_LABELS.get(s, s) for s in HITTER_STATS],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, frameon=False)

    if metric == "r":
        ax.set_ylim(-0.12, 1.0)
        ax.axhline(0, color=DGRAY, lw=1.2, zorder=3)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    ax.grid(axis="y", lw=0.5)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_model_comparison.png")
plt.close()
print("Saved fig3_model_comparison.png")


# ==========================================================================
# FIG 4 — Ablation: component contribution
# ==========================================================================

ABL_MODELS  = ["League average", "Prior year only",
                "Marcel (no age adj)", "Marcel (full system)"]
ABL_COLORS  = ["#D3D1C7", "#888780", LBLUE, BLUE]

fig, axes = plt.subplots(1, len(HITTER_STATS), figsize=(13, 4.5))
fig.suptitle("")

for ax, stat in zip(axes, HITTER_STATS):
    sub = ablation[ablation["stat"] == stat].set_index("model")
    rs  = [sub.loc[m, "r"] if m in sub.index else np.nan for m in ABL_MODELS]

    bars = ax.bar(range(len(ABL_MODELS)), rs, color=ABL_COLORS, width=0.7, zorder=2)
    for bar, r_val in zip(bars, rs):
        if not np.isnan(r_val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    max(bar.get_height(), 0) + 0.01,
                    f"{r_val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color=DGRAY, lw=0.6)
    ax.set_title(STAT_LABELS.get(stat, stat), fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(ABL_MODELS)))
    ax.set_xticklabels(["Lg avg", "Prior yr", "Marcel\n–age", "Marcel\n+age"],
                       fontsize=8)
    ax.set_ylim(-0.15, 1.0)
    ax.set_ylabel("Pearson r" if stat == HITTER_STATS[0] else "")
    ax.grid(axis="y", lw=0.4)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_ablation.png")
plt.close()
print("Saved fig4_ablation.png")


# ==========================================================================
# FIG 5 — 2026 Hitter Projections (top 20 by wOBA)
# ==========================================================================

proj = proj_2026.merge(
    pt_h[["player_id", "PA_proj"]], on="player_id", how="left", suffixes=("","_pt")
)
proj["proj_HR"] = (proj["HR_pct_proj"] * proj["PA_proj"]).round(1)
proj["n_label"] = proj["n_seasons"].map({1: "1 yr", 2: "2 yr", 3: "3 yr"})

top20 = proj.dropna(subset=["wOBA_proj"]).sort_values("wOBA_proj", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8, 9))
fig.suptitle("")

ax.barh(range(len(top20)), top20["wOBA_proj"].values,
        color=BLUE, height=0.7, zorder=2)

ax.set_yticks(range(len(top20)))
labels = []
for _, r in top20.iterrows():
    age_str = f"(age {int(r['age'])})" if pd.notna(r["age"]) else ""
    labels.append(f"{r['player_name']} {age_str}".strip())
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()

for i, (_, r) in enumerate(top20.iterrows()):
    ax.text(r["wOBA_proj"] + 0.002, i,
            f"{r['wOBA_proj']:.3f}", va="center", fontsize=8, color=DGRAY)

ax.set_xlabel("Projected wOBA")
ax.set_xlim(0.30, top20["wOBA_proj"].max() + 0.025)
ax.axvline(top20["wOBA_proj"].median(), color=RED, lw=1, ls="--",
           label=f"Top-20 median ({top20['wOBA_proj'].median():.3f})")
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(axis="x", lw=0.4)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_projections_2026.png")
plt.close()
print("Saved fig5_projections_2026.png")


# ==========================================================================
# FIG 6 — Playing time by age
# ==========================================================================

hitters_with_age = hitters_clean.merge(
    ages[["player_id", "birth_year"]].dropna(),
    on="player_id", how="inner"
)
hitters_with_age["age"] = hitters_with_age["season"] - hitters_with_age["birth_year"]

pt_by_age = (hitters_with_age.groupby("age")
             .agg(mean_PA=("PA","mean"), n=("PA","count"))
             .reset_index()
             .query("n >= 5 and 18 <= age <= 42"))

fig, ax = plt.subplots(figsize=(9, 4))
fig.suptitle("")

ax.bar(pt_by_age["age"], pt_by_age["mean_PA"], color=LBLUE,
       alpha=0.7, width=0.8, zorder=2, label="Mean PA")

ax2 = ax.twinx()
ax2.plot(pt_by_age["age"], pt_by_age["n"], "o--",
         color=LGRAY, lw=1.2, ms=4, label="n players")
ax2.set_ylabel("Number of player-seasons", color=LGRAY, fontsize=9)
ax2.tick_params(axis="y", labelcolor=LGRAY, labelsize=8)
ax2.spines["top"].set_visible(False)

ax.set_xlabel("Age")
ax.set_ylabel("Mean plate appearances (PA)")
ax.set_xlim(17, 43)
ax.axvspan(24, 30, alpha=0.07, color="#BA7517", label="Peak range (24–30)")

lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig6_pt_by_age.png")
plt.close()
print("Saved fig6_pt_by_age.png")


# ==========================================================================
# FIG 7 — 2026 Pitcher Projections (top 20 by FIP)
# ==========================================================================

proj_p = pd.read_csv(f"{DATA_DIR}/pitcher_projections_2026.csv")
pt_p   = pd.read_csv(f"{DATA_DIR}/pitcher_pt_projections.csv")

proj_p = proj_p.merge(
    pt_p[["player_id", "IP_proj"]], on="player_id", how="left"
)

top20_p = (proj_p.dropna(subset=["FIP_proj"])
           .sort_values("FIP_proj", ascending=True)
           .head(20))

fig, ax = plt.subplots(figsize=(8, 9))
fig.suptitle("")

ax.barh(range(len(top20_p)), top20_p["FIP_proj"].values,
        color=BLUE, height=0.7, zorder=2)

ax.set_yticks(range(len(top20_p)))
labels_p = []
for _, r in top20_p.iterrows():
    age_str = f"(age {int(r['age'])})" if pd.notna(r["age"]) else ""
    labels_p.append(f"{r['player_name']} {age_str}".strip())
ax.set_yticklabels(labels_p, fontsize=9)
ax.invert_yaxis()

for i, (_, r) in enumerate(top20_p.iterrows()):
    ax.text(r["FIP_proj"] + 0.02, i,
            f"{r['FIP_proj']:.2f}", va="center", fontsize=8, color=DGRAY)

ax.set_xlabel("Projected FIP (lower = better)")
ax.set_xlim(top20_p["FIP_proj"].min() - 0.2, top20_p["FIP_proj"].max() + 0.6)
ax.axvline(top20_p["FIP_proj"].median(), color=RED, lw=1, ls="--",
           label=f"Top-20 median ({top20_p['FIP_proj'].median():.2f})")
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.grid(axis="x", lw=0.4)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig7_pitcher_projections_2026.png")
plt.close()
print("Saved fig7_pitcher_projections_2026.png")


# ==========================================================================
# SUMMARY
# ==========================================================================

print("\n=== FIGURES COMPLETE ===")
print("All 7 publication figures saved to outputs/figures/")
print()
print("Figure inventory:")
figs = [
    ("fig1_age_curves.png",               "KBO hitter aging curves (7 stats, delta method)"),
    ("fig2_validation_scatter.png",        "Predicted vs actual wOBA — 3 model comparison"),
    ("fig3_model_comparison.png",          "MAE and r across all stats and models"),
    ("fig4_ablation.png",                  "Component contribution (ablation study)"),
    ("fig5_projections_2026.png",          "2026 hitter projections, top 20 by wOBA"),
    ("fig6_pt_by_age.png",                 "Playing time (PA) by age"),
    ("fig7_pitcher_projections_2026.png",  "2026 pitcher projections, top 20 by FIP"),
]
for fname, desc in figs:
    print(f"  {fname:<35} {desc}")
