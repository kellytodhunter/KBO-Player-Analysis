"""
anchor_comparison.py
--------------------
Compares Marcel backtest accuracy using age anchor 27 (MLB convention)
vs age anchor 24 (KBO empirical peak) across 2021-2024 held-out seasons.

Run from the project root:
  python code/anchor_comparison.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DATA_DIR = "data"

HITTER_STATS = ["BB_pct", "K_pct", "HR_pct", "ISO", "wOBA"]
YEAR_WEIGHTS = {0: 5, 1: 4, 2: 3}
LEAGUE_WEIGHT = 1
BACKTEST_YEARS = list(range(2021, 2025))

# ── Load data ────────────────────────────────────────────────────────────────
hitters_reg    = pd.read_csv(f"{DATA_DIR}/hitters_regressed.csv")
league_avg_h   = pd.read_csv(f"{DATA_DIR}/league_avgs_hitters.csv")
hitter_curves  = pd.read_csv(f"{DATA_DIR}/hitter_age_curves.csv")
ages           = pd.read_csv(f"{DATA_DIR}/player_ages.csv")
ages["player_id"] = ages["player_id"].astype(str)
has_ages = ages["birth_year"].notna().sum() > 0


# ── Age adjustment helpers ────────────────────────────────────────────────────
def build_age_adj(curves_df, stat_cols, anchor_age):
    adj = {}
    for stat in stat_cols:
        col = (f"delta_{stat}_smooth"
               if f"delta_{stat}_smooth" in curves_df.columns
               else f"delta_{stat}")
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
    if adj_series.empty:
        return 0.0
    min_age = adj_series.index.min()
    max_age = adj_series.index.max()
    age_c = int(np.clip(age, min_age + 1, max_age))
    prev  = age_c - 1
    return float(adj_series.get(age_c, adj_series.iloc[-1])
                 - adj_series.get(prev, adj_series.iloc[-1]))


# ── Projection for one year ───────────────────────────────────────────────────
def project_one_year(df, league_avg_df, stat_cols, adj_dict, target_year):
    reg_cols = [s + "_reg" for s in stat_cols]
    results  = []
    train    = df[df["season"] < target_year]
    player_ids = train[
        train["season"].between(target_year - 3, target_year - 1)
    ]["player_id"].unique()

    for pid in player_ids:
        player_seasons = train[train["player_id"] == pid].sort_values("season")
        recent = player_seasons[player_seasons["season"] < target_year].tail(3)
        if recent.empty:
            continue

        age_at_proj = np.nan
        if has_ages:
            match = ages[ages["player_id"].astype(str) == str(pid)]
            if not match.empty and pd.notna(match["birth_year"].iloc[0]):
                age_at_proj = target_year - int(match["birth_year"].iloc[0])

        lg_year = min(target_year - 1, league_avg_df["season"].max())
        lg_row  = league_avg_df[league_avg_df["season"] == lg_year]
        if lg_row.empty:
            continue
        lg = lg_row.iloc[0]

        row = {"player_id": pid, "projection_year": target_year, "age": age_at_proj}

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
                proj += get_delta(adj_dict[stat], int(age_at_proj))

            row[stat + "_proj"] = proj

        results.append(row)
    return pd.DataFrame(results)


# ── Run backtest for a given anchor age ──────────────────────────────────────
def run_backtest(anchor_age):
    adj = build_age_adj(hitter_curves, HITTER_STATS, anchor_age)
    all_rows = []
    for yr in BACKTEST_YEARS:
        proj   = project_one_year(hitters_reg, league_avg_h, HITTER_STATS, adj, yr)
        actual = hitters_reg[hitters_reg["season"] == yr][
            ["player_id"] + HITTER_STATS
        ].copy()
        actual.columns = ["player_id"] + [s + "_actual" for s in HITTER_STATS]
        merged = proj.merge(actual, on="player_id", how="inner")
        all_rows.append(merged)
    return pd.concat(all_rows, ignore_index=True)


# ── Compute metrics ───────────────────────────────────────────────────────────
def eval_metrics(bt_df, label):
    rows = []
    for stat in HITTER_STATS:
        sub = bt_df[[stat + "_proj", stat + "_actual"]].dropna()
        pred, act = sub[stat + "_proj"], sub[stat + "_actual"]
        mae  = (pred - act).abs().mean()
        rmse = np.sqrt(((pred - act) ** 2).mean())
        r, _ = pearsonr(pred, act) if len(sub) > 2 else (np.nan, np.nan)
        rows.append({"Model": label, "Stat": stat,
                     "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                     "Pearson r": round(r, 4), "n": len(sub)})
    return pd.DataFrame(rows)


# ── Run both anchors ──────────────────────────────────────────────────────────
print("Running anchor-27 backtest...")
bt27 = run_backtest(anchor_age=27)
m27  = eval_metrics(bt27, "Anchor 27 (MLB)")

print("Running anchor-24 backtest...")
bt24 = run_backtest(anchor_age=24)
m24  = eval_metrics(bt24, "Anchor 24 (KBO)")

# ── Side-by-side comparison ───────────────────────────────────────────────────
combined = m27.merge(m24, on="Stat", suffixes=(" (27)", " (24)"))

print("\n" + "="*75)
print("ANCHOR AGE COMPARISON: 27 (MLB convention) vs 24 (KBO empirical peak)")
print("="*75)
print(f"\n{'Stat':<8} {'MAE-27':>8} {'MAE-24':>8} {'Δ MAE':>8}  "
      f"{'r-27':>7} {'r-24':>7} {'Δ r':>7}  {'Winner'}")
print("-"*75)

for _, row in combined.iterrows():
    d_mae = row["MAE (24)"] - row["MAE (27)"]   # negative = 24 is better
    d_r   = row["Pearson r (24)"] - row["Pearson r (27)"]  # positive = 24 is better
    winner_mae = "24✓" if d_mae < 0 else ("27✓" if d_mae > 0 else "tie")
    winner_r   = "24✓" if d_r > 0 else ("27✓" if d_r < 0 else "tie")
    print(f"{row['Stat']:<8} {row['MAE (27)']:>8.4f} {row['MAE (24)']:>8.4f} "
          f"{d_mae:>+8.4f}  {row['Pearson r (27)']:>7.4f} {row['Pearson r (24)']:>7.4f} "
          f"{d_r:>+7.4f}  MAE:{winner_mae} r:{winner_r}")

print("\n--- Average across all 5 stats ---")
avg27 = m27[["MAE", "RMSE", "Pearson r"]].mean()
avg24 = m24[["MAE", "RMSE", "Pearson r"]].mean()
print(f"{'':8} {'Anchor 27':>10} {'Anchor 24':>10} {'Δ (24-27)':>12}")
for metric in ["MAE", "RMSE", "Pearson r"]:
    d = avg24[metric] - avg27[metric]
    print(f"{metric:<8} {avg27[metric]:>10.4f} {avg24[metric]:>10.4f} {d:>+12.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
combined.to_csv(f"{DATA_DIR}/anchor_comparison_results.csv", index=False)
print(f"\nFull results saved to {DATA_DIR}/anchor_comparison_results.csv")
