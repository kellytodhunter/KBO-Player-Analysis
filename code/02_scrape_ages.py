"""
02_scrape_ages.py
-----------------
Pulls player birth years from Namu Wiki (namu.wiki) — no login, no geo-restriction.
Birth year is extracted from category links like /분류:1984년_출생 on each player's page.

Foreign players without a Namu Wiki page are skipped (birth_year = NaN).

Outputs:
  data/player_ages.csv  - player_id, player_name, birth_year
"""

import re
import time
import requests
import pandas as pd
from urllib.parse import quote

DATA_DIR   = "data"
NAMU_URL   = "https://namu.wiki/w/{name}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


# =========================================================
# LOAD PLAYER LIST
# =========================================================

hitters  = pd.read_csv(f"{DATA_DIR}/hitters_clean.csv")
pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_clean.csv")

all_players = pd.concat([
    hitters[["player_id", "player_name"]],
    pitchers[["player_id", "player_name"]],
]).drop_duplicates(subset="player_id").dropna(subset=["player_id"])

all_players["player_id"] = all_players["player_id"].astype(int).astype(str)
print(f"Players to fetch: {len(all_players)}\n")


# =========================================================
# SCRAPE
# =========================================================

def fetch_birth_year_namu(session, player_name):
    """
    Fetch Namu Wiki page for player_name and extract birth year.
    Returns (birth_year: int | None).
    """
    url  = NAMU_URL.format(name=quote(player_name))
    resp = session.get(url, timeout=10)

    if resp.status_code == 404:
        return None  # page doesn't exist

    resp.raise_for_status()
    html = resp.text

    # Primary: URL-encoded category link — /분류:1984년_출생
    # %EB%85%84 = 년, %EC%B6%9C%EC%83%9D = 출생
    match = re.search(r'/(\d{4})%EB%85%84[^"]*%EC%B6%9C%EC%83%9D', html)
    if match:
        year = int(match.group(1))
        if 1960 <= year <= 2006:
            return year

    # Fallback: plain text "1984년...출생" within 60 chars
    match2 = re.search(r'(19[6-9]\d|200[0-6])년.{0,60}출생', html)
    if match2:
        return int(match2.group(1))

    return None


results = []

for idx, row in enumerate(all_players.itertuples(), 1):
    pid  = row.player_id
    name = row.player_name

    try:
        birth_year = fetch_birth_year_namu(session, name)
        results.append({
            "player_id":   pid,
            "player_name": name,
            "birth_year":  birth_year,
        })
        status = str(birth_year) if birth_year else "not found"
        print(f"[{idx}/{len(all_players)}] {name}: {status}")

    except Exception as e:
        print(f"[{idx}/{len(all_players)}] {name}: ERROR — {e}")
        results.append({"player_id": pid, "player_name": name, "birth_year": None})

    time.sleep(0.5)


# =========================================================
# SAVE
# =========================================================

ages_df = pd.DataFrame(results)

# Merge manual birth year patches (foreign imports not on Namu Wiki)
MANUAL_FILE = f"{DATA_DIR}/manual_birth_years.csv"
import os
if os.path.exists(MANUAL_FILE):
    manual = pd.read_csv(MANUAL_FILE)
    manual = manual[manual["birth_year"].notna() & (manual["birth_year"] != "")]
    manual["player_id"] = manual["player_id"].astype(str)
    manual["birth_year"] = manual["birth_year"].astype(int)
    ages_df["player_id"] = ages_df["player_id"].astype(str)
    # Fill in any missing birth year with the manual value
    for _, patch in manual.iterrows():
        mask = ages_df["player_id"] == patch["player_id"]
        ages_df.loc[mask, "birth_year"] = patch["birth_year"]
    print(f"Applied {len(manual)} manual birth year patches")

ages_df.to_csv(f"{DATA_DIR}/player_ages.csv", index=False)

found   = ages_df["birth_year"].notna().sum()
missing = ages_df[ages_df["birth_year"].isna()]["player_name"].tolist()

print(f"\n=== DONE ===")
print(f"Birth years found: {found}/{len(ages_df)}")
if missing:
    print(f"Still missing ({len(missing)}): {missing}")
print(f"Saved to {DATA_DIR}/player_ages.csv")
