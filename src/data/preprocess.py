"""
Data preprocessing: reads raw matches CSV and outputs a clean
processed DataFrame saved as CSV for feature engineering.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    MATCHES_CSV, PROCESSED_MATCHES_CSV, PROCESSED_DIR,
    ACTIVE_TEAMS_2026, RETIRED_TEAM_MAP,
)

# Teams that are retired (map to successors for continuity)
RETIRED = {"DC_OLD", "RPS", "KTK", "PW", "GL"}


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV)
    print(f"Loaded {len(df)} matches from {MATCHES_CSV}")
    return df


def normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Map retired franchise names to current successors for ML continuity."""
    for col in ["team1", "team2", "winner", "toss_winner"]:
        df[col] = df[col].replace(RETIRED_TEAM_MAP)
    # Drop any matches involving teams with no successor
    df = df[~df["team1"].isin({"KTK", "PW"}) & ~df["team2"].isin({"KTK", "PW"})].copy()
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """For each row add: team1_won (1 if team1 won, 0 if team2 won)."""
    df = df.copy()
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)
    # Drop draws / no-result
    df = df[df["winner"].notna() & (df["winner"] != "")]
    return df


def add_toss_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["toss_won_by_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)
    return df


def add_season_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by season and match_id to ensure temporal ordering."""
    df = df.sort_values(["season", "id"]).reset_index(drop=True)
    return df


def mirror_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes class imbalance by adding a mirrored copy of every match.

    For every match (team1 vs team2, winner=team1), we add the equivalent row
    with team1/team2 swapped and winner flipped. This:
      - Doubles the dataset size
      - Makes class distribution exactly 50/50
      - Eliminates position bias
    """
    mirrored = df.copy()
    mirrored["id"] = -mirrored["id"]
    mirrored["team1"] = df["team2"]
    mirrored["team2"] = df["team1"]
    mirrored["toss_winner"] = df["toss_winner"]
    mirrored["toss_won_by_team1"] = 1 - df["toss_won_by_team1"]
    mirrored["team1_won"] = 1 - df["team1_won"]

    combined = pd.concat([df, mirrored], ignore_index=True)
    combined = combined.sort_values(["season", "id"]).reset_index(drop=True)

    orig_pos = df["team1_won"].mean()
    new_pos = combined["team1_won"].mean()
    print(f"  Class balance before mirroring: {orig_pos:.2%} team1 wins")
    print(f"  Class balance  after mirroring: {new_pos:.2%} team1 wins (target: 50%)")
    return combined


def save_processed(df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    # Rename 'id' to 'match_id' for downstream compatibility
    df_out = df.copy()
    if "id" in df_out.columns and "match_id" not in df_out.columns:
        df_out = df_out.rename(columns={"id": "match_id"})
    df_out.to_csv(PROCESSED_MATCHES_CSV, index=False)
    print(f"Processed {len(df_out)} matches -> {PROCESSED_MATCHES_CSV}")
    print(f"Columns: {list(df_out.columns)}")


def run_preprocessing() -> pd.DataFrame:
    df = load_matches()
    df = normalize_teams(df)
    df = add_binary_target(df)
    df = add_toss_features(df)
    df = add_season_order(df)
    df = mirror_matches(df)
    save_processed(df)
    return df


if __name__ == "__main__":
    run_preprocessing()
