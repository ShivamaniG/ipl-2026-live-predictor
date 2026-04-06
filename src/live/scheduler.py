"""
CLI runner for live IPL 2026 score integration.

Usage:
  python -m src.live.scheduler --api-key YOUR_KEY --once
  python -m src.live.scheduler --api-key YOUR_KEY --interval 30
  python main.py --mode live --api-key YOUR_KEY
"""
import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)


def run_live_update(api_key: str = None, retrain: bool = False,
                    dry_run: bool = False, verbose: bool = True) -> dict:
    """
    Single live update cycle:
      1. Fetch completed matches from API
      2. Match to schedule, deduplicate
      3. Append to training data
      4. Refresh pipeline
      5. Re-predict with partial-season logic
      6. Track accuracy
    Returns a results dict for reporting.
    """
    from config import CRICAPI_KEY
    from src.live.fetch_scores import fetch_completed_matches
    from src.live.updater import (
        load_schedule, load_completed_results, update_with_new_results,
        append_to_training_data, run_pipeline_refresh,
        build_points_table, log_upcoming_predictions, compute_accuracy_tracker,
    )
    from src.prediction.predict_2026 import predict_2026_partial, print_predictions

    api_key = api_key or CRICAPI_KEY
    if not api_key:
        logger.warning("No API key provided. Set CRICAPI_KEY env var or pass --api-key.")
        logger.info("Running with local data only (no API fetch).")

    # Step 1: Load schedule
    schedule_df = load_schedule()
    completed_df = load_completed_results()
    completed_before = len(completed_df)

    # Step 2: Fetch from API (if key provided)
    if api_key:
        logger.info("Fetching live scores...")
        new_results = fetch_completed_matches(api_key)

        if new_results and not dry_run:
            new_count = update_with_new_results(new_results, schedule_df)
            if new_count > 0:
                logger.info("%d new match result(s) found.", new_count)
                completed_df = load_completed_results()
            else:
                logger.info("No new results since last update.")
        elif new_results and dry_run:
            logger.info("[DRY RUN] Would add %d results. Skipping file updates.", len(new_results))
    else:
        logger.info("Skipping API fetch (no key). Using existing local data.")

    completed_after = len(completed_df)
    new_matches = completed_after - completed_before

    results = {
        "matches_completed": completed_after,
        "new_matches": new_matches,
        "rankings": [],
        "points_table": None,
        "accuracy": {"total": 0, "correct": 0, "accuracy_pct": 0.0},
    }

    if completed_after == 0:
        logger.info("No completed matches yet. Running standard prediction.")
        from src.prediction.predict_2026 import predict_2026_winner
        rankings = predict_2026_winner()
        results["rankings"] = rankings
        if verbose:
            print_predictions(rankings)
        return results

    # Step 3: If new matches, refresh pipeline
    if new_matches > 0 and not dry_run:
        logger.info("Refreshing pipeline with %d new match(es)...", new_matches)
        append_to_training_data(completed_df)
        run_pipeline_refresh(retrain=retrain)

    # Step 4: Log predictions for upcoming matches (before they happen)
    completed_numbers = set(completed_df["match_number"].astype(int).tolist())
    if not dry_run:
        log_upcoming_predictions(schedule_df, completed_numbers)

    # Step 5: Partial-season prediction
    logger.info("Running partial-season prediction...")
    rankings = predict_2026_partial(completed_df)
    results["rankings"] = rankings

    # Step 6: Points table
    points_table = build_points_table(completed_df)
    results["points_table"] = points_table

    # Step 7: Accuracy tracking
    accuracy = compute_accuracy_tracker(completed_df)
    results["accuracy"] = accuracy

    # Step 8: Match-by-match predictions CSV
    if not dry_run:
        from src.prediction.match_predictor import predict_all_2026_matches
        predict_all_2026_matches()

    # Step 9: Report
    if verbose:
        print_live_report(results)

    return results


def print_live_report(results: dict):
    """Print formatted live update report."""
    print("\n" + "=" * 65)
    print("         IPL 2026 LIVE UPDATE REPORT")
    print("=" * 65)

    completed = results.get("matches_completed", 0)
    new = results.get("new_matches", 0)
    print(f"\n  Matches completed: {completed}/70")
    if new > 0:
        print(f"  New since last update: {new}")

    # Points table
    pts = results.get("points_table")
    if pts is not None and len(pts) > 0:
        print(f"\n  {'':>3} {'Team':<30} {'P':>3} {'W':>3} {'L':>3} {'Pts':>4}")
        print("  " + "-" * 50)
        for idx, row in pts.iterrows():
            print(f"  {idx:>3} {row['Team']:<30} {row['P']:>3} {row['W']:>3} "
                  f"{row['L']:>3} {row['Pts']:>4}")

    # Accuracy
    acc = results.get("accuracy", {})
    if acc.get("total", 0) > 0:
        print(f"\n  Model accuracy this season: {acc['correct']}/{acc['total']} "
              f"({acc['accuracy_pct']}%)")

    # Rankings
    rankings = results.get("rankings", [])
    if rankings:
        print(f"\n  {'Rank':<6} {'Team':<35} {'Win Probability':>15}")
        print("  " + "-" * 60)
        for r in rankings:
            bar = "#" * int(r["win_probability"] / 2)
            print(f"  {r['rank']:<4}   {r['team_name']:<33} "
                  f"{r['win_probability']:>6.2f}%  {bar}")

        w = rankings[0]
        remaining = w.get("matches_remaining", "?")
        print(f"\n  PREDICTED WINNER: {w['team_name']} ({w['win_probability']:.2f}%)")
        print(f"  Matches remaining: {remaining}")

    print("=" * 65)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="IPL 2026 Live Score Integration",
    )
    parser.add_argument("--api-key", default=None,
                        help="CricketData.org API key (or set CRICAPI_KEY env var)")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain models after adding new matches")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (default: loop)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Minutes between update checks (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and display results without updating files")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_cli_args()
    api_key = args.api_key or os.environ.get("CRICAPI_KEY", "")

    if args.once:
        run_live_update(api_key=api_key, retrain=args.retrain, dry_run=args.dry_run)
        return

    print(f"Live update loop started (interval: {args.interval} min). Press Ctrl+C to stop.")
    try:
        while True:
            run_live_update(api_key=api_key, retrain=args.retrain, dry_run=args.dry_run)
            logger.info("Next update in %d minutes...", args.interval)
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        print("\nLive update stopped.")


if __name__ == "__main__":
    main()
