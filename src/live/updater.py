"""
Core update engine for live IPL 2026 score integration.

Responsibilities:
  - Load/save the 2026 schedule and completed results
  - Append completed matches to the training dataset
  - Re-run the preprocessing + feature engineering pipeline
  - Track model prediction accuracy across the season
"""
import os
import sys
import json
import shutil
import logging
from datetime import datetime
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    SCHEDULE_CSV, COMPLETED_2026_CSV, PREDICTIONS_LOG, LIVE_DIR,
    MATCHES_CSV, PROCESSED_MATCHES_CSV, FEATURES_CSV,
    ACTIVE_TEAMS_2026, TEAMS, FULL_NAME_TO_ABBR, TEAM_ALIASES,
    SCHEDULE_VENUE_MAP,
)
from src.live.fetch_scores import normalize_team_name, match_to_schedule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule loading
# ---------------------------------------------------------------------------

def load_schedule() -> pd.DataFrame:
    """Load the 2026 fixture schedule with normalized team names."""
    df = pd.read_csv(SCHEDULE_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Rename columns to internal names
    col_map = {
        "Match Number": "match_number",
        "Round Number": "round_number",
        "Date": "date",
        "Location": "venue",
        "Home Team": "home_team",
        "Away Team": "away_team",
        "Result": "result",
    }
    df = df.rename(columns=col_map)

    # Normalize team names
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M", errors="coerce")

    # Map venue names
    df["venue_canonical"] = df["venue"].map(SCHEDULE_VENUE_MAP).fillna(df["venue"])

    return df


# ---------------------------------------------------------------------------
# Completed results store
# ---------------------------------------------------------------------------

COMPLETED_COLUMNS = [
    "match_number", "date", "team1", "team2", "winner",
    "win_by_runs", "win_by_wickets", "venue", "toss_winner", "toss_decision",
]


def load_completed_results() -> pd.DataFrame:
    """Load completed 2026 results from local store."""
    if os.path.exists(COMPLETED_2026_CSV):
        df = pd.read_csv(COMPLETED_2026_CSV)
        return df
    return pd.DataFrame(columns=COMPLETED_COLUMNS)


def save_completed_results(df: pd.DataFrame):
    """Save completed results to local store."""
    os.makedirs(LIVE_DIR, exist_ok=True)
    df.to_csv(COMPLETED_2026_CSV, index=False)
    logger.info("Saved %d completed results to %s", len(df), COMPLETED_2026_CSV)


# ---------------------------------------------------------------------------
# Update with new results
# ---------------------------------------------------------------------------

def update_with_new_results(new_results: list, schedule_df: pd.DataFrame) -> int:
    """
    Match new API results to the schedule, deduplicate, save.
    Returns count of newly added matches.
    """
    completed_df = load_completed_results()
    existing_match_numbers = set(completed_df["match_number"].tolist())

    new_rows = []
    for result in new_results:
        match_num = match_to_schedule(result, schedule_df)
        if match_num < 0:
            continue
        if match_num in existing_match_numbers:
            continue

        row = {
            "match_number": match_num,
            "date": str(result.get("date", "")),
            "team1": result["team1"],
            "team2": result["team2"],
            "winner": result["winner"],
            "win_by_runs": result.get("win_by_runs", 0),
            "win_by_wickets": result.get("win_by_wickets", 0),
            "venue": result.get("venue", ""),
            "toss_winner": result.get("toss_winner", result["team1"]),
            "toss_decision": result.get("toss_decision", "field"),
        }
        new_rows.append(row)
        existing_match_numbers.add(match_num)
        logger.info("New result: Match %d — %s beat %s",
                     match_num, result["winner"],
                     result["team2"] if result["winner"] == result["team1"] else result["team1"])

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows, columns=COMPLETED_COLUMNS)
    completed_df = pd.concat([completed_df, new_df], ignore_index=True)
    completed_df = completed_df.sort_values("match_number").reset_index(drop=True)
    save_completed_results(completed_df)
    return len(new_rows)


# ---------------------------------------------------------------------------
# Append to training data
# ---------------------------------------------------------------------------

def append_to_training_data(completed_df: pd.DataFrame) -> int:
    """
    Append completed 2026 matches to data/raw/matches.csv.
    Skips no-result matches (rain, abandoned) — they have no winner.
    Returns count of newly appended rows.
    """
    matches_df = pd.read_csv(MATCHES_CSV)

    # Find existing 2026 matches already in the file
    existing_2026 = set()
    if "season" in matches_df.columns:
        m2026 = matches_df[matches_df["season"] == 2026]
        for _, row in m2026.iterrows():
            key = (str(row.get("team1", "")), str(row.get("team2", "")),
                   int(row.get("id", 0)))
            existing_2026.add(key)

    max_id = int(matches_df["id"].max()) if "id" in matches_df.columns and len(matches_df) > 0 else 0

    # Backup matches.csv before modifying
    backup_path = MATCHES_CSV + ".bak"
    shutil.copy2(MATCHES_CSV, backup_path)

    new_rows = []
    for _, result in completed_df.iterrows():
        # Check if already appended (by match_number in a simple way)
        # We use team pair + season as dedup key
        team1 = result["team1"]
        team2 = result["team2"]
        winner = result.get("winner", "")

        # Skip no-result matches (rain/abandoned) — no winner to train on
        if not winner or winner == "NR" or winner not in (team1, team2):
            continue

        already_exists = False
        if "season" in matches_df.columns:
            mask = (
                (matches_df["season"] == 2026) &
                (((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
                 ((matches_df["team1"] == team2) & (matches_df["team2"] == team1)))
            )
            if mask.any():
                already_exists = True

        if already_exists:
            continue

        max_id += 1
        venue = result.get("venue", "")
        # Map schedule venue to canonical if possible
        canonical = SCHEDULE_VENUE_MAP.get(venue, venue)

        toss_winner = result.get("toss_winner", team1)
        toss_decision = result.get("toss_decision", "field")
        if not toss_winner or toss_winner not in (team1, team2):
            toss_winner = team1

        row = {
            "id": max_id,
            "season": 2026,
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": result["winner"],
            "win_by_runs": result.get("win_by_runs", 0),
            "win_by_wickets": result.get("win_by_wickets", 0),
            "venue": canonical,
            "city": "",
            "stage": "League",
        }
        new_rows.append(row)

    if not new_rows:
        os.remove(backup_path)
        return 0

    try:
        new_df = pd.DataFrame(new_rows)
        matches_df = pd.concat([matches_df, new_df], ignore_index=True)
        matches_df.to_csv(MATCHES_CSV, index=False)
        os.remove(backup_path)
        logger.info("Appended %d matches to %s (total: %d)", len(new_rows), MATCHES_CSV, len(matches_df))
        return len(new_rows)
    except Exception as e:
        # Restore backup on failure
        shutil.copy2(backup_path, MATCHES_CSV)
        os.remove(backup_path)
        logger.error("Failed to append matches, restored backup: %s", e)
        raise


# ---------------------------------------------------------------------------
# Pipeline refresh
# ---------------------------------------------------------------------------

def run_pipeline_refresh(retrain: bool = False):
    """Re-run preprocessing, ingestion, and feature engineering with updated data."""
    from src.data.preprocess import run_preprocessing
    from src.data.ingest import run_ingestion
    from src.features.engineer import run_feature_engineering

    logger.info("Refreshing pipeline with updated match data...")

    logger.info("  Step 1/3: Preprocessing...")
    run_preprocessing()

    logger.info("  Step 2/3: Ingesting into SQLite...")
    run_ingestion()

    logger.info("  Step 3/3: Engineering features...")
    run_feature_engineering()

    if retrain:
        logger.info("  Step 4/4: Retraining models (--retrain flag)...")
        from src.models.trainer import run_training
        run_training()
        logger.info("  Retraining complete.")
    else:
        logger.info("  Skipping retraining (use --retrain to retrain models).")

    logger.info("Pipeline refresh complete.")


# ---------------------------------------------------------------------------
# Points table
# ---------------------------------------------------------------------------

def build_points_table(completed_df: pd.DataFrame) -> pd.DataFrame:
    """Build IPL-style points table from completed results."""
    stats = defaultdict(lambda: {"P": 0, "W": 0, "L": 0, "NR": 0, "Pts": 0})

    for _, row in completed_df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], str(row.get("winner", ""))
        stats[t1]["P"] += 1
        stats[t2]["P"] += 1

        if not winner or winner == "NR" or winner not in (t1, t2):
            # No result (rain/abandoned) — 1 point each
            stats[t1]["NR"] += 1
            stats[t2]["NR"] += 1
            stats[t1]["Pts"] += 1
            stats[t2]["Pts"] += 1
        elif winner == t1:
            stats[t1]["W"] += 1
            stats[t1]["Pts"] += 2
            stats[t2]["L"] += 1
        elif winner == t2:
            stats[t2]["W"] += 1
            stats[t2]["Pts"] += 2
            stats[t1]["L"] += 1

    rows = []
    for team in ACTIVE_TEAMS_2026:
        s = stats[team]
        rows.append({
            "Team": TEAMS.get(team, team),
            "Abbr": team,
            "P": s["P"],
            "W": s["W"],
            "L": s["L"],
            "NR": s["NR"],
            "Pts": s["Pts"],
            "NRR": 0.0,  # Not available from basic API data
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Pts", "NRR"], ascending=[False, False]).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    return df


# ---------------------------------------------------------------------------
# Prediction accuracy tracking
# ---------------------------------------------------------------------------

def _load_predictions_log() -> list:
    if os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG) as f:
            return json.load(f)
    return []


def _save_predictions_log(log: list):
    os.makedirs(LIVE_DIR, exist_ok=True)
    with open(PREDICTIONS_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


def log_upcoming_predictions(schedule_df: pd.DataFrame, completed_match_numbers: set):
    """
    Log model predictions for matches not yet completed.
    Only logs matches that haven't been logged before.
    """
    from src.prediction.match_predictor import predict_match

    log = _load_predictions_log()
    logged_matches = {entry["match_number"] for entry in log}

    new_logs = 0
    for _, row in schedule_df.iterrows():
        match_num = int(row["match_number"])
        if match_num in completed_match_numbers or match_num in logged_matches:
            continue

        try:
            result = predict_match(row["home_team"], row["away_team"])
            entry = {
                "match_number": match_num,
                "team1": row["home_team"],
                "team2": row["away_team"],
                "predicted_winner": result["predicted_winner"],
                "team1_win_prob": result["team1_win_prob"],
                "confidence": result["confidence"],
                "logged_at": str(datetime.now()),
            }
            log.append(entry)
            new_logs += 1
        except Exception as e:
            logger.warning("Failed to predict match %d: %s", match_num, e)

    if new_logs > 0:
        _save_predictions_log(log)
        logger.info("Logged predictions for %d upcoming matches.", new_logs)


def compute_accuracy_tracker(completed_df: pd.DataFrame) -> dict:
    """Compare pre-match predictions against actual outcomes."""
    log = _load_predictions_log()
    if not log:
        return {"total": 0, "correct": 0, "accuracy_pct": 0.0, "details": []}

    completed_winners = {}
    for _, row in completed_df.iterrows():
        completed_winners[int(row["match_number"])] = row["winner"]

    correct = 0
    total = 0
    details = []

    for entry in log:
        match_num = entry["match_number"]
        if match_num not in completed_winners:
            continue

        actual_winner = completed_winners[match_num]
        predicted = entry["predicted_winner"]
        is_correct = predicted == actual_winner
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "match_number": match_num,
            "predicted": predicted,
            "actual": actual_winner,
            "correct": is_correct,
            "confidence": entry.get("confidence", 0),
        })

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "details": details,
    }
