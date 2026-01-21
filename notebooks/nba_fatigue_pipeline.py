import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from geopy.distance import geodesic

# -----------------------------
# Config
# -----------------------------
SEASON = "2023-24"
RAW_OUT = os.path.join("data_raw", "games_2023_24_raw.csv")
TEAM_GAME_OUT = os.path.join(
    "data_processed", "team_games_2324_with_fatigue.csv")
MATCHUP_OUT = os.path.join("data_processed", "nba_2324_fatigue_matchups.csv")

# -----------------------------
# Ensure folders exist
# -----------------------------
os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)

# -----------------------------
# Pull games (one row per TEAM per GAME)
# -----------------------------
print("Pulling NBA games for season:", SEASON)

gf = leaguegamefinder.LeagueGameFinder(
    season_nullable=SEASON,
    league_id_nullable="00"
)
games = gf.get_data_frames()[0].copy()

games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
games = games.sort_values(["TEAM_NAME", "GAME_DATE"])

games.to_csv(RAW_OUT, index=False)
print("Saved raw games to:", RAW_OUT, "| rows:", len(games))

# -----------------------------
# Rest features
# -----------------------------
games["days_rest"] = games.groupby("TEAM_NAME")["GAME_DATE"].diff().dt.days - 1
games["days_rest"] = games["days_rest"].fillna(5).clip(lower=0)
games["is_b2b"] = (games["days_rest"] == 0).astype(int)

# Home/away from MATCHUP
games["is_home"] = games["MATCHUP"].str.contains("vs.").astype(int)

# -----------------------------
# Arena coordinates (30 teams)
# -----------------------------
team_arena_coords = {
    "Atlanta Hawks": (33.7573, -84.3963),
    "Boston Celtics": (42.3662, -71.0621),
    "Brooklyn Nets": (40.6826, -73.9754),
    "Charlotte Hornets": (35.2251, -80.8392),
    "Chicago Bulls": (41.8807, -87.6742),
    "Cleveland Cavaliers": (41.4965, -81.6882),
    "Dallas Mavericks": (32.7905, -96.8103),
    "Denver Nuggets": (39.7487, -105.0077),
    "Detroit Pistons": (42.3411, -83.0553),
    "Golden State Warriors": (37.7680, -122.3877),
    "Houston Rockets": (29.7508, -95.3621),
    "Indiana Pacers": (39.7639, -86.1555),
    # using Crypto.com Arena coords
    "LA Clippers": (34.0430, -118.2673),
    "Los Angeles Lakers": (34.0430, -118.2673),   # Crypto.com Arena
    "Memphis Grizzlies": (35.1382, -90.0506),
    "Miami Heat": (25.7814, -80.1870),
    "Milwaukee Bucks": (43.0451, -87.9172),
    "Minnesota Timberwolves": (44.9795, -93.2760),
    "New Orleans Pelicans": (29.9490, -90.0821),
    "New York Knicks": (40.7505, -73.9934),
    "Oklahoma City Thunder": (35.4634, -97.5151),
    "Orlando Magic": (28.5392, -81.3839),
    "Philadelphia 76ers": (39.9012, -75.1720),
    "Phoenix Suns": (33.4458, -112.0712),
    "Portland Trail Blazers": (45.5316, -122.6668),
    "Sacramento Kings": (38.5802, -121.4997),
    "San Antonio Spurs": (29.4269, -98.4375),
    "Toronto Raptors": (43.6435, -79.3791),
    "Utah Jazz": (40.7683, -111.9011),
    "Washington Wizards": (38.8981, -77.0209),
}

# -----------------------------
# Opponent parsing (MATCHUP uses abbreviations)
# Example: "ATL vs. CLE" or "ATL @ PHI"
# We'll build an abbreviation -> team name map from the dataset itself.
# -----------------------------
abbr_to_team = (
    games[["TEAM_ABBREVIATION", "TEAM_NAME"]]
    .drop_duplicates()
    .set_index("TEAM_ABBREVIATION")["TEAM_NAME"]
    .to_dict()
)


def extract_opponent_abbr(matchup: str) -> str:
    # last token is opponent abbreviation in nba_api matchup strings
    # "ATL vs. CLE" -> "CLE"
    # "ATL @ PHI"   -> "PHI"
    parts = str(matchup).split()
    return parts[-1] if parts else None


games["opponent_abbr"] = games["MATCHUP"].apply(extract_opponent_abbr)
games["opponent_team"] = games["opponent_abbr"].map(abbr_to_team)

# -----------------------------
# Assign game location coords:
# home game -> team's arena
# away game -> opponent's arena
# -----------------------------


def get_game_coords(row) -> tuple | None:
    if row["is_home"] == 1:
        return team_arena_coords.get(row["TEAM_NAME"])
    return team_arena_coords.get(row["opponent_team"])


games["game_coords"] = games.apply(get_game_coords, axis=1)

# Check for missing coords (should be 0; if not, we’ll fix names)
missing_coords = games["game_coords"].isna().sum()
print("Missing game_coords rows:", missing_coords)

# -----------------------------
# Travel miles: previous game location -> current game location (per team)
# -----------------------------
games["prev_game_coords"] = games.groupby("TEAM_NAME")["game_coords"].shift(1)


def calc_miles(prev_c, curr_c) -> float:
    if prev_c is None or curr_c is None or pd.isna(prev_c) or pd.isna(curr_c):
        return 0.0
    return float(geodesic(prev_c, curr_c).miles)


games["travel_miles"] = games.apply(lambda r: calc_miles(
    r["prev_game_coords"], r["game_coords"]), axis=1)

# Fatigue score (simple but not “mostly zero”)
# - B2B is a strong penalty
# - travel scaled so cross-country jumps show up
games["fatigue_score"] = (2.0 * games["is_b2b"]) + \
    (games["travel_miles"] / 750.0)

print("\nTravel miles summary:")
print(games["travel_miles"].describe())
print("\nFatigue score summary:")
print(games["fatigue_score"].describe())

# Save processed team-game file
games.to_csv(TEAM_GAME_OUT, index=False)
print("\nSaved processed team-game data to:", TEAM_GAME_OUT)

# -----------------------------
# Build one-row-per-game matchups (Tableau-ready)
# -----------------------------
match_cols = [
    "GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "WL", "PTS",
    "days_rest", "is_b2b", "travel_miles", "fatigue_score"
]
g = games[match_cols].copy()

home = g[g["MATCHUP"].str.contains("vs.")].copy()
away = g[g["MATCHUP"].str.contains("@")].copy()

home = home.rename(columns={
    "TEAM_NAME": "home_team",
    "WL": "home_wl",
    "PTS": "home_pts",
    "days_rest": "home_days_rest",
    "is_b2b": "home_is_b2b",
    "travel_miles": "home_travel_miles",
    "fatigue_score": "home_fatigue_score",
    "MATCHUP": "home_matchup"
})

away = away.rename(columns={
    "TEAM_NAME": "away_team",
    "WL": "away_wl",
    "PTS": "away_pts",
    "days_rest": "away_days_rest",
    "is_b2b": "away_is_b2b",
    "travel_miles": "away_travel_miles",
    "fatigue_score": "away_fatigue_score",
    "MATCHUP": "away_matchup"
})

matchups = home.merge(away, on=["GAME_ID", "GAME_DATE"], how="inner")

matchups["home_win"] = (matchups["home_wl"] == "W").astype(int)
matchups["point_diff"] = matchups["home_pts"] - matchups["away_pts"]

matchups["rest_diff"] = matchups["home_days_rest"] - matchups["away_days_rest"]
matchups["travel_diff"] = matchups["home_travel_miles"] - \
    matchups["away_travel_miles"]
matchups["fatigue_diff"] = matchups["home_fatigue_score"] - \
    matchups["away_fatigue_score"]

matchups.to_csv(MATCHUP_OUT, index=False)
print("\nSaved Tableau-ready matchups to:", MATCHUP_OUT)
print("Matchup rows (games):", len(matchups))
print(matchups[[
    "GAME_ID", "GAME_DATE", "home_team", "away_team",
    "home_pts", "away_pts", "home_win",
    "rest_diff", "travel_diff", "fatigue_diff"
]].head(8))
