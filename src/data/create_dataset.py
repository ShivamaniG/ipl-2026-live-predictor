"""
Extracts match-level summary data from the ball-by-ball IPL.csv dataset.

Reads the real IPL ball-by-ball CSV and produces:
  - data/raw/matches.csv  (match-level summary: teams, toss, winner, venue, etc.)
  - data/raw/teams.json   (team metadata)
  - data/raw/player_stats.csv (per-season player batting/bowling stats)

Source: IPL.csv (ball-by-ball data, 2008-2025, ~278K deliveries, 1169 matches)
"""
import os
import sys
import re
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DIR, MATCHES_CSV, TEAMS_JSON, TEAM_ALIASES, BASE_DIR

IPL_CSV = os.path.join(BASE_DIR, "IPL.csv")

# Map season strings (e.g. "2007/08") to integer years
SEASON_TO_YEAR = {
    "2007/08": 2008, "2009": 2009, "2009/10": 2010,
    "2011": 2011, "2012": 2012, "2013": 2013, "2014": 2014,
    "2015": 2015, "2016": 2016, "2017": 2017, "2018": 2018,
    "2019": 2019, "2020/21": 2020, "2021": 2021, "2022": 2022,
    "2023": 2023, "2024": 2024, "2025": 2025,
}


def normalize_team(name: str) -> str:
    """Map full team name to abbreviation."""
    return TEAM_ALIASES.get(name, name)


def parse_win_outcome(outcome: str):
    """Parse '33 runs' or '5 wickets' into (win_by_runs, win_by_wickets)."""
    if pd.isna(outcome) or not isinstance(outcome, str):
        return 0, 0
    m = re.search(r"(\d+)\s+runs", outcome)
    if m:
        return int(m.group(1)), 0
    m = re.search(r"(\d+)\s+wickets?", outcome)
    if m:
        return 0, int(m.group(1))
    return 0, 0


def extract_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Extract match-level summary from ball-by-ball data."""
    # Get team1 (batting first) and team2 from first innings
    innings1 = df[df["innings"] == 1].groupby("match_id").agg(
        team1=("batting_team", "first"),
        team2=("bowling_team", "first"),
    ).reset_index()

    # Get match-level metadata (one row per match)
    match_meta = df.groupby("match_id").agg(
        date=("date", "first"),
        season=("season", "first"),
        year=("year", "first"),
        match_won_by=("match_won_by", "first"),
        win_outcome=("win_outcome", "first"),
        toss_winner=("toss_winner", "first"),
        toss_decision=("toss_decision", "first"),
        venue=("venue", "first"),
        city=("city", "first"),
        result_type=("result_type", "first"),
        stage=("stage", "first"),
    ).reset_index()

    matches = innings1.merge(match_meta, on="match_id")

    # Filter out no-results and ties with unknown winner
    matches = matches[matches["match_won_by"] != "Unknown"].copy()
    matches = matches[matches["result_type"].isin([None, "None", "nan", "tie", np.nan]) |
                      matches["result_type"].isna()].copy()
    # Keep ties only if there's a super over winner (match_won_by != Unknown)
    # At this point Unknown is already filtered out

    # Parse win margin
    parsed = matches["win_outcome"].apply(parse_win_outcome)
    matches["win_by_runs"] = parsed.apply(lambda x: x[0])
    matches["win_by_wickets"] = parsed.apply(lambda x: x[1])

    # Map season to integer year
    matches["season_year"] = matches["season"].map(SEASON_TO_YEAR)
    matches["season_year"] = matches["season_year"].fillna(matches["year"]).astype(int)

    # Normalize team names to abbreviations
    matches["team1"] = matches["team1"].apply(normalize_team)
    matches["team2"] = matches["team2"].apply(normalize_team)
    matches["winner"] = matches["match_won_by"].apply(normalize_team)
    matches["toss_winner"] = matches["toss_winner"].apply(normalize_team)

    # Sort by date
    matches = matches.sort_values(["season_year", "date", "match_id"]).reset_index(drop=True)
    matches["id"] = range(1, len(matches) + 1)

    result = matches[["id", "season_year", "team1", "team2", "toss_winner",
                       "toss_decision", "winner", "win_by_runs", "win_by_wickets",
                       "venue", "city", "stage"]].copy()
    result.columns = ["id", "season", "team1", "team2", "toss_winner",
                       "toss_decision", "winner", "win_by_runs", "win_by_wickets",
                       "venue", "city", "stage"]
    return result


def extract_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-season player batting and bowling stats from ball-by-ball data."""
    df = df.copy()
    df["season_year"] = df["season"].map(SEASON_TO_YEAR)
    df["season_year"] = df["season_year"].fillna(df["year"]).astype(int)

    # --- Batting stats ---
    batter_runs = df.groupby(["season_year", "batter", "batting_team"]).agg(
        runs_scored=("runs_batter", "sum"),
        balls_faced=("balls_faced", "sum"),
    ).reset_index()

    # Dismissals
    dismissed = df[df["player_out"].notna() & (df["player_out"] != "")]
    dismissals = dismissed.groupby(["season_year", "player_out"]).size().reset_index(name="dismissals")

    batter_runs = batter_runs.merge(
        dismissals,
        left_on=["season_year", "batter"],
        right_on=["season_year", "player_out"],
        how="left",
    )
    batter_runs["dismissals"] = batter_runs["dismissals"].fillna(0).astype(int)
    batter_runs["batting_avg"] = batter_runs["runs_scored"] / batter_runs["dismissals"].clip(lower=1)
    batter_runs["batting_sr"] = (
        batter_runs["runs_scored"] / batter_runs["balls_faced"].clip(lower=1)
    ) * 100

    # --- Bowling stats ---
    valid_balls = df[df["valid_ball"] == 1]
    bowler_stats = valid_balls.groupby(["season_year", "bowler", "bowling_team"]).agg(
        runs_conceded=("runs_bowler", "sum"),
        balls_bowled=("valid_ball", "sum"),
        wickets=("bowler_wicket", "sum"),
    ).reset_index()
    bowler_stats["economy"] = bowler_stats["runs_conceded"] / (
        bowler_stats["balls_bowled"] / 6
    ).clip(lower=1)
    bowler_stats["bowling_avg"] = bowler_stats["runs_conceded"] / bowler_stats["wickets"].clip(lower=1)

    # --- Combine into player stats ---
    # Batting side
    bat = batter_runs[batter_runs["runs_scored"] >= 50].copy()  # min 50 runs in season
    bat["team"] = bat["batting_team"].apply(normalize_team)
    bat["player_name"] = bat["batter"]
    bat = bat[["season_year", "player_name", "team", "runs_scored", "balls_faced",
               "dismissals", "batting_avg", "batting_sr"]].copy()

    # Bowling side
    bowl = bowler_stats[bowler_stats["wickets"] >= 3].copy()  # min 3 wickets in season
    bowl["team"] = bowl["bowling_team"].apply(normalize_team)
    bowl["player_name"] = bowl["bowler"]
    bowl = bowl[["season_year", "player_name", "team", "runs_conceded",
                  "balls_bowled", "wickets", "economy", "bowling_avg"]].copy()

    # Merge batting and bowling
    combined = bat.merge(
        bowl[["season_year", "player_name", "wickets", "economy", "bowling_avg"]],
        on=["season_year", "player_name"],
        how="outer",
    )
    # Fill missing
    for col in ["runs_scored", "batting_avg", "batting_sr", "balls_faced", "dismissals"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)
    for col in ["wickets", "economy", "bowling_avg"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # Fill team from bowling data where missing
    bowl_teams = bowl.set_index(["season_year", "player_name"])["team"]
    mask = combined["team"].isna()
    for idx in combined[mask].index:
        key = (combined.loc[idx, "season_year"], combined.loc[idx, "player_name"])
        if key in bowl_teams.index:
            combined.loc[idx, "team"] = bowl_teams.loc[key]

    # Determine role
    def assign_role(row):
        has_bat = row["runs_scored"] >= 50
        has_bowl = row["wickets"] >= 3
        if has_bat and has_bowl:
            return "All"
        elif has_bowl:
            return "Bowl"
        else:
            return "Bat"

    combined["role"] = combined.apply(assign_role, axis=1)
    combined = combined.rename(columns={"season_year": "season"})

    return combined[["season", "player_name", "team", "role",
                      "batting_avg", "batting_sr", "runs_scored",
                      "wickets", "bowling_avg", "economy"]]


def save_teams_json():
    """Save team metadata to JSON."""
    os.makedirs(RAW_DIR, exist_ok=True)
    teams = {
        "CSK":  {"name": "Chennai Super Kings",        "home": "MA Chidambaram Stadium",            "titles": 5, "founded": 2008},
        "MI":   {"name": "Mumbai Indians",             "home": "Wankhede Stadium",                  "titles": 5, "founded": 2008},
        "RCB":  {"name": "Royal Challengers Bengaluru","home": "M Chinnaswamy Stadium",             "titles": 1, "founded": 2008},
        "KKR":  {"name": "Kolkata Knight Riders",      "home": "Eden Gardens",                     "titles": 3, "founded": 2008},
        "DC":   {"name": "Delhi Capitals",             "home": "Arun Jaitley Stadium",              "titles": 0, "founded": 2008},
        "PBKS": {"name": "Punjab Kings",               "home": "Punjab Cricket Association IS Bindra Stadium", "titles": 0, "founded": 2008},
        "RR":   {"name": "Rajasthan Royals",           "home": "Sawai Mansingh Stadium",            "titles": 1, "founded": 2008},
        "SRH":  {"name": "Sunrisers Hyderabad",        "home": "Rajiv Gandhi International Cricket Stadium", "titles": 1, "founded": 2013},
        "LSG":  {"name": "Lucknow Super Giants",       "home": "BRSABV Ekana Cricket Stadium",      "titles": 0, "founded": 2022},
        "GT":   {"name": "Gujarat Titans",             "home": "Narendra Modi Stadium",             "titles": 1, "founded": 2022},
    }
    with open(TEAMS_JSON, "w") as f:
        json.dump(teams, f, indent=2)
    print(f"Saved teams.json ({len(teams)} teams)")


def save_matches_csv(matches_df: pd.DataFrame):
    """Save match-level data to CSV."""
    os.makedirs(RAW_DIR, exist_ok=True)
    matches_df.to_csv(MATCHES_CSV, index=False)
    n_seasons = matches_df["season"].nunique()
    print(f"Saved {len(matches_df)} real matches ({n_seasons} seasons) -> {MATCHES_CSV}")


def save_player_stats_csv(player_stats_df: pd.DataFrame):
    """Save player stats to CSV."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "player_stats.csv")
    player_stats_df.to_csv(path, index=False)
    print(f"Saved {len(player_stats_df)} player-season records -> {path}")


def _to_legacy_match_rows(matches_df: pd.DataFrame) -> list:
    """Return legacy tuple rows expected by older tests/utilities.

    Row format:
      (season, team1, team2, winner, venue, toss_winner,
       toss_decision, win_by_runs, win_by_wickets)
    """
    cols = [
        "season", "team1", "team2", "winner", "venue", "toss_winner",
        "toss_decision", "win_by_runs", "win_by_wickets",
    ]
    return [tuple(row[c] for c in cols) for _, row in matches_df.iterrows()]


def build_all_matches(return_format: str = "legacy"):
    """Load IPL.csv and extract match-level + player stats data.

    return_format:
      - "legacy": returns list[tuple] for backward-compatible callers/tests
      - "dataframes": returns (matches_df, player_stats_df)
    """
    print(f"Loading ball-by-ball data from {IPL_CSV}...")
    df = pd.read_csv(IPL_CSV, low_memory=False)
    print(f"  {len(df)} deliveries, {df['match_id'].nunique()} matches")

    print("Extracting match-level summary...")
    matches = extract_matches(df)
    print(f"  {len(matches)} valid matches extracted")

    print("Extracting player statistics...")
    player_stats = extract_player_stats(df)
    print(f"  {len(player_stats)} player-season records extracted")

    if return_format == "dataframes":
        return matches, player_stats
    return _to_legacy_match_rows(matches)


if __name__ == "__main__":
    save_teams_json()
    matches, player_stats = build_all_matches(return_format="dataframes")
    save_matches_csv(matches)
    save_player_stats_csv(player_stats)
    print("Dataset extraction complete.")
