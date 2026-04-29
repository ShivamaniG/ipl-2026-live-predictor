"""
Replay completed 2026 matches as a rolling prediction backtest.

For each completed match in data/live/api_cache.json:
  1. Predict the match before adding its actual result.
  2. Record whether the prediction was correct.
  3. Add the actual result to an in-memory 2026 season state.
  4. Recompute the tournament winner table after each match.

Outputs:
  outputs/results/rolling_2026_match_accuracy.csv
  outputs/results/rolling_2026_winner_trends.csv
  outputs/results/rolling_2026_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    API_CACHE,
    COMPLETED_2026_CSV,
    PROCESSED_MATCHES_CSV,
    RESULTS_DIR,
    SCHEDULE_VENUE_MAP,
    TEAMS,
)
from src.live.fetch_scores import match_to_schedule, normalize_api_result
from src.live.updater import COMPLETED_COLUMNS, load_schedule
from src.models.ensemble_model import EnsembleModel
from src.models.xgboost_model import XGBoostModel
from src.prediction.predict_2026 import (
    bayesian_update_dynamic,
    build_matchup_features,
    rank_predictions,
    simulate_tournament_with_actuals,
)


@dataclass
class BacktestResult:
    match_rows: list[dict]
    trend_rows: list[dict]
    completed_df: pd.DataFrame


def load_model():
    try:
        model = EnsembleModel()
        model.load()
        return model, "ensemble"
    except FileNotFoundError:
        model = XGBoostModel()
        model.load()
        return model, "xgboost"


def load_cache_results(cache_path: str) -> list[dict]:
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)

    raw_results = cache if isinstance(cache, list) else cache.get("results", [])
    normalized = []
    for raw in raw_results:
        result = normalize_api_result(raw)
        winner_raw = str(raw.get("winner_full", "")).strip().lower()
        status_text = str(raw.get("status_text", "")).lower()
        if winner_raw in {"", "none", "nr", "no result"} or "abandoned" in status_text:
            result["winner"] = "NR"
        if result["winner"] in (result["team1"], result["team2"], "NR"):
            normalized.append(result)
    return normalized


def cache_result_count(cache_path: str) -> int:
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    return len(cache if isinstance(cache, list) else cache.get("results", []))


def attach_match_numbers(results, schedule_df: pd.DataFrame) -> list[dict]:
    rows = []
    seen = set()
    for result in results:
        match_number = match_to_schedule(result, schedule_df)
        if match_number < 0:
            print(
                "Warning: could not match schedule row for "
                f"{result['team1']} vs {result['team2']} on {result.get('date')}"
            )
            continue
        if match_number in seen:
            continue
        seen.add(match_number)
        schedule_match = schedule_df[schedule_df["match_number"] == match_number].iloc[0]
        enriched = dict(result)
        enriched["match_number"] = match_number
        enriched["schedule_date"] = schedule_match["date"]
        rows.append(enriched)

    # Replay strictly in fixture order, which is the real season chronology.
    rows.sort(key=lambda r: (pd.Timestamp(r["schedule_date"]), int(r["match_number"])))
    return rows


def completed_row(result: dict) -> dict:
    return {
        "match_number": int(result["match_number"]),
        "date": str(result.get("date") or ""),
        "team1": result["team1"],
        "team2": result["team2"],
        "winner": result["winner"],
        "win_by_runs": int(result.get("win_by_runs", 0) or 0),
        "win_by_wickets": int(result.get("win_by_wickets", 0) or 0),
        "venue": result.get("venue", ""),
        "toss_winner": result.get("toss_winner", ""),
        "toss_decision": result.get("toss_decision", "field"),
    }


def append_result_to_history(matches_df: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Append one 2026 actual result to the in-memory match history."""
    team1 = result["team1"]
    team2 = result["team2"]
    winner = result["winner"]
    if winner not in (team1, team2):
        return matches_df

    max_match_id = int(pd.to_numeric(matches_df["match_id"], errors="coerce").max())
    venue = SCHEDULE_VENUE_MAP.get(result.get("venue", ""), result.get("venue", ""))
    toss_winner = result.get("toss_winner", team1)
    if toss_winner not in (team1, team2):
        toss_winner = team1

    row = {
        "match_id": max_match_id + 1,
        "season": 2026,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": result.get("toss_decision", "field"),
        "winner": winner,
        "win_by_runs": int(result.get("win_by_runs", 0) or 0),
        "win_by_wickets": int(result.get("win_by_wickets", 0) or 0),
        "venue": venue,
        "city": "",
        "stage": "League",
        "team1_won": int(winner == team1),
        "toss_won_by_team1": int(toss_winner == team1),
        "toss_decision_bat": int(result.get("toss_decision", "field") == "bat"),
    }
    return pd.concat([matches_df, pd.DataFrame([row])], ignore_index=True)


def predict_match_before_result(model, history_df: pd.DataFrame, result: dict) -> dict:
    team1 = result["team1"]
    team2 = result["team2"]
    feats = build_matchup_features(team1, team2, history_df)

    toss_winner = result.get("toss_winner")
    if toss_winner in (team1, team2):
        feats["toss_won_by_team1"] = int(toss_winner == team1)
    feats["toss_decision_bat"] = int(result.get("toss_decision", "field") == "bat")

    t1_prob = float(model.predict_proba(feats)[:, 1].mean())
    t2_prob = 1.0 - t1_prob
    predicted = team1 if t1_prob >= 0.5 else team2

    return {
        "predicted_winner": predicted,
        "predicted_winner_name": TEAMS.get(predicted, predicted),
        "team1_win_probability": round(t1_prob * 100, 2),
        "team2_win_probability": round(t2_prob * 100, 2),
        "confidence": round(max(t1_prob, t2_prob) * 100, 2),
    }


def tournament_snapshot(model, history_df: pd.DataFrame, completed_df: pd.DataFrame) -> list[dict]:
    completed_results = {}
    for _, row in completed_df.iterrows():
        if row["winner"] in (row["team1"], row["team2"]):
            completed_results[frozenset({row["team1"], row["team2"]})] = row["winner"]

    model_probs = simulate_tournament_with_actuals(model, history_df, completed_results)
    final_probs = bayesian_update_dynamic(model_probs, len(completed_df))
    return rank_predictions(final_probs)


def run_backtest(cache_path: str = API_CACHE) -> BacktestResult:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    match_path = os.path.join(RESULTS_DIR, "rolling_2026_match_accuracy.csv")
    trend_path = os.path.join(RESULTS_DIR, "rolling_2026_winner_trends.csv")
    summary_path = os.path.join(RESULTS_DIR, "rolling_2026_summary.json")

    schedule_df = load_schedule()
    results = attach_match_numbers(load_cache_results(cache_path), schedule_df)
    model, model_name = load_model()
    history_df = pd.read_csv(PROCESSED_MATCHES_CSV)
    completed_df = pd.DataFrame(columns=COMPLETED_COLUMNS)

    match_rows = []
    trend_rows = []

    def write_reports(final: bool = False):
        match_df = pd.DataFrame(match_rows)
        trend_df = pd.DataFrame(trend_rows)
        match_df.to_csv(match_path, index=False)
        trend_df.to_csv(trend_path, index=False)

        if len(match_df) and "actual_winner" in match_df.columns:
            scored_df = match_df[match_df["actual_winner"] != "NR"]
        else:
            scored_df = match_df

        correct = int(scored_df["correct"].sum()) if len(scored_df) else 0
        total = int(len(scored_df))
        summary = {
            "generated_on": str(date.today()),
            "status": "complete" if final else "running",
            "model": model_name,
            "cache_path": cache_path,
            "matches_in_cache": cache_result_count(cache_path),
            "processed_matches": len(match_df),
            "matched_completed_matches": len(match_df),
            "scored_matches": total,
            "no_result_matches": int((match_df["actual_winner"] == "NR").sum()) if len(match_df) else 0,
            "correct_predictions": correct,
            "accuracy": round(correct / total * 100, 2) if total else 0.0,
            "latest_leader": trend_rows[-1]["leader_name"] if trend_rows else None,
            "latest_leader_probability": trend_rows[-1]["leader_probability"] if trend_rows else None,
            "outputs": {
                "match_accuracy_csv": match_path,
                "winner_trends_csv": trend_path,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return match_df, trend_df, summary

    print(f"Replaying {len(results)} matched results in schedule order...")

    for idx, result in enumerate(results, start=1):
        print(
            f"  {idx}/{len(results)}: Match {int(result['match_number'])} "
            f"{result['team1']} vs {result['team2']} "
            f"({result.get('date')})",
            flush=True,
        )
        pred = predict_match_before_result(model, history_df, result)
        actual = result["winner"]
        correct = pred["predicted_winner"] == actual if actual != "NR" else None

        match_rows.append({
            "match_number": result["match_number"],
            "date": str(result.get("date") or ""),
            "team1": result["team1"],
            "team2": result["team2"],
            "team1_name": TEAMS.get(result["team1"], result["team1"]),
            "team2_name": TEAMS.get(result["team2"], result["team2"]),
            "predicted_winner": pred["predicted_winner"],
            "predicted_winner_name": pred["predicted_winner_name"],
            "actual_winner": actual,
            "actual_winner_name": "No Result" if actual == "NR" else TEAMS.get(actual, actual),
            "correct": correct,
            "confidence": pred["confidence"],
            "team1_win_probability": pred["team1_win_probability"],
            "team2_win_probability": pred["team2_win_probability"],
            "venue": result.get("venue", ""),
            "toss_winner": result.get("toss_winner", ""),
            "toss_decision": result.get("toss_decision", ""),
        })

        completed_df = pd.concat(
            [completed_df, pd.DataFrame([completed_row(result)])],
            ignore_index=True,
        )
        history_df = append_result_to_history(history_df, result)

        rankings = tournament_snapshot(model, history_df, completed_df)
        trend_row = {
            "after_match": result["match_number"],
            "date": str(result.get("date") or ""),
            "matches_completed": len(completed_df),
            "leader": rankings[0]["team_id"],
            "leader_name": rankings[0]["team_name"],
            "leader_probability": rankings[0]["win_probability"],
        }
        for ranking in rankings:
            trend_row[f"{ranking['team_id']}_rank"] = ranking["rank"]
            trend_row[f"{ranking['team_id']}_probability"] = ranking["win_probability"]
        trend_rows.append(trend_row)

        write_reports(final=False)

    match_df, trend_df, summary = write_reports(final=True)

    print("\nRolling 2026 backtest complete")
    print(f"  Cache results:          {summary['matches_in_cache']}")
    print(f"  Matched to schedule:    {summary['matched_completed_matches']}")
    print(f"  Scored matches:         {summary['scored_matches']} ({summary['no_result_matches']} no-result)")
    print(f"  Correct predictions:    {summary['correct_predictions']}/{summary['scored_matches']}")
    print(f"  Accuracy:               {summary['accuracy']:.2f}%")
    print(f"  Latest leader:          {summary['latest_leader']} ({summary['latest_leader_probability']}%)")
    print(f"  Match report:           {match_path}")
    print(f"  Winner trend report:    {trend_path}")
    print(f"  Summary:                {summary_path}")

    return BacktestResult(match_rows, trend_rows, completed_df)


def write_completed_csv_from_cache(cache_path: str = API_CACHE) -> str:
    """Optional helper: replace completed_2026.csv with matched cache results."""
    schedule_df = load_schedule()
    results = attach_match_numbers(load_cache_results(cache_path), schedule_df)
    completed_df = pd.DataFrame([completed_row(r) for r in results], columns=COMPLETED_COLUMNS)
    completed_df.to_csv(COMPLETED_2026_CSV, index=False)
    print(f"Wrote {len(completed_df)} rows to {COMPLETED_2026_CSV}")
    return COMPLETED_2026_CSV


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling IPL 2026 prediction backtest")
    parser.add_argument("--cache", default=API_CACHE, help="Path to api_cache.json")
    parser.add_argument(
        "--write-completed",
        action="store_true",
        help="Also replace data/live/completed_2026.csv with matched cache results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args.cache)
    if args.write_completed:
        write_completed_csv_from_cache(args.cache)
