"""
Fetch live/completed IPL 2026 match results from cricket APIs.

Primary:  CricketData.org (formerly CricAPI) — free tier, 100 req/day
Fallback: ESPN undocumented scoreboard API — no key, less reliable
"""
import os
import sys
import json
import re
import logging
from datetime import datetime, date

import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    CRICAPI_BASE_URL, ESPN_SCOREBOARD_URL, API_CACHE, LIVE_DIR,
    FULL_NAME_TO_ABBR, TEAM_ALIASES, ACTIVE_TEAMS_2026, SCHEDULE_VENUE_MAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API cache & rate limiter
# ---------------------------------------------------------------------------

def _load_api_cache() -> dict:
    if os.path.exists(API_CACHE):
        with open(API_CACHE) as f:
            return json.load(f)
    return {"date": "", "calls_today": 0, "results": []}


def _save_api_cache(cache: dict):
    os.makedirs(LIVE_DIR, exist_ok=True)
    with open(API_CACHE, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _check_rate_limit(cache: dict, max_daily: int = 90) -> bool:
    """Returns True if we can still make API calls today."""
    today = str(date.today())
    if cache.get("date") != today:
        cache["date"] = today
        cache["calls_today"] = 0
    return cache["calls_today"] < max_daily


def _increment_calls(cache: dict):
    cache["calls_today"] = cache.get("calls_today", 0) + 1


# ---------------------------------------------------------------------------
# Team name normalization
# ---------------------------------------------------------------------------

def normalize_team_name(full_name: str) -> str:
    """Convert a full team name to its abbreviation (e.g. 'Mumbai Indians' -> 'MI')."""
    if not full_name:
        return ""
    name = full_name.strip()
    # Direct match in FULL_NAME_TO_ABBR
    if name in FULL_NAME_TO_ABBR:
        abbr = FULL_NAME_TO_ABBR[name]
        # TEAM_ALIASES values may be abbreviations, FULL_NAME_TO_ABBR may give the same
        if abbr in ACTIVE_TEAMS_2026:
            return abbr
    # Try TEAM_ALIASES
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    # Fuzzy: check if any known team name is a substring
    for known, abbr in FULL_NAME_TO_ABBR.items():
        if known.lower() in name.lower() or name.lower() in known.lower():
            if abbr in ACTIVE_TEAMS_2026:
                return abbr
    logger.warning("Could not normalize team name: '%s'", full_name)
    return full_name


def parse_win_margin(status_text: str) -> tuple:
    """
    Extract (win_by_runs, win_by_wickets) from result text.
    E.g. 'Mumbai Indians won by 5 wickets' -> (0, 5)
         'CSK won by 33 runs' -> (33, 0)
    """
    if not status_text:
        return 0, 0
    m = re.search(r'(\d+)\s+run', status_text, re.IGNORECASE)
    if m:
        return int(m.group(1)), 0
    m = re.search(r'(\d+)\s+(?:wicket|wkt)', status_text, re.IGNORECASE)
    if m:
        return 0, int(m.group(1))
    return 0, 0


def extract_winner(status_text: str) -> str:
    """Extract the winning team's full name from status text like 'Mumbai Indians won by ...'."""
    if not status_text:
        return ""
    m = re.match(r'^(.+?)\s+won\s+by', status_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# CricketData.org (CricAPI) client
# ---------------------------------------------------------------------------

def _fetch_match_detail(api_key: str, match_id: str, cache: dict) -> dict:
    """Fetch detailed match info (including toss) from /match_info endpoint."""
    if not _check_rate_limit(cache):
        return {}
    url = f"{CRICAPI_BASE_URL}/match_info"
    params = {"apikey": api_key, "id": match_id}
    try:
        resp = requests.get(url, params=params, timeout=15)
        _increment_calls(cache)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {})
    except (requests.RequestException, ValueError) as e:
        logger.warning("match_info request failed for %s: %s", match_id, e)
        return {}


def fetch_completed_matches_cricapi(api_key: str) -> list:
    """
    Fetch completed IPL 2026 matches from CricketData.org.

    Strategy (API-efficient):
      1. /currentMatches (1 call) → find completed IPL match IDs
      2. /match_info per NEW match only (1 call each) → get full data + toss
      Skips matches already in cache. Total: ~2-3 calls per match day.
    """
    cache = _load_api_cache()
    if not _check_rate_limit(cache):
        logger.warning("CricAPI daily rate limit reached (90 calls). Using cached results.")
        return cache.get("results", [])

    # Track which matches we already have (by team pair + date)
    existing_keys = set()
    for r in cache.get("results", []):
        key = (r.get("team1_full", ""), r.get("team2_full", ""), r.get("date", ""))
        existing_keys.add(key)

    # Step 1: /currentMatches — find completed IPL matches (1 API call)
    new_matches = []  # (match_dict, match_id)

    url = f"{CRICAPI_BASE_URL}/currentMatches"
    params = {"apikey": api_key, "offset": 0}

    try:
        resp = requests.get(url, params=params, timeout=15)
        _increment_calls(cache)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("CricAPI /currentMatches request failed: %s", e)
        _save_api_cache(cache)
        return cache.get("results", [])

    if data.get("status") != "success" or not data.get("data"):
        _save_api_cache(cache)
        return cache.get("results", [])

    for match in data["data"]:
        series = match.get("series", "") or match.get("name", "")
        if "indian premier league" not in series.lower() and "ipl" not in series.lower():
            continue

        status = match.get("status", "")
        match_id = match.get("id", "")
        if "won" not in status.lower() or not match_id:
            continue

        # Skip if already cached
        teams = match.get("teams", [])
        if len(teams) >= 2:
            key = (teams[0], teams[1], match.get("date", ""))
            if key in existing_keys:
                continue

        new_matches.append((match, match_id))

    logger.info("Found %d NEW completed IPL matches.", len(new_matches))

    # Step 2: /match_info for each NEW match only (1 call per match)
    new_results = []
    for match, match_id in new_matches:
        status = match.get("status", "")
        teams = match.get("teams", [])
        if len(teams) < 2:
            continue

        winner_full = extract_winner(status)
        if not winner_full:
            winner_full = match.get("matchWinner", "")

        win_runs, win_wickets = parse_win_margin(status)

        # Fetch full details including toss
        toss_winner = ""
        toss_choice = ""
        venue = match.get("venue", "")

        detail = _fetch_match_detail(api_key, match_id, cache)
        if detail:
            toss_winner = detail.get("tossWinner", "")
            toss_choice = detail.get("tossChoice", "")
            if detail.get("venue"):
                venue = detail["venue"]

        # Normalize toss_choice: API returns "bowl" but pipeline uses "field"
        if toss_choice.lower() in ("bowl", "bowling", "field", "fielding"):
            toss_choice = "field"
        elif toss_choice.lower() in ("bat", "batting"):
            toss_choice = "bat"

        result = {
            "date": match.get("date", ""),
            "team1_full": teams[0],
            "team2_full": teams[1],
            "winner_full": winner_full,
            "venue": venue,
            "status_text": status,
            "win_by_runs": win_runs,
            "win_by_wickets": win_wickets,
            "toss_winner": toss_winner,
            "toss_choice": toss_choice,
        }
        new_results.append(result)

    # Merge: keep all cached results + add new ones
    all_results = cache.get("results", []) + new_results
    cache["results"] = all_results
    _save_api_cache(cache)
    return all_results


# ---------------------------------------------------------------------------
# ESPN fallback client
# ---------------------------------------------------------------------------

def fetch_completed_matches_espn() -> list:
    """
    Fetch completed IPL matches from ESPN's undocumented scoreboard API.
    No API key needed, but this endpoint may change without notice.
    """
    try:
        params = {"sport": "cricket", "region": "in"}
        resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.error("ESPN API request failed: %s", e)
        return []

    results = []
    sports = data.get("sports", [])
    for sport in sports:
        for league in sport.get("leagues", []):
            if "ipl" not in league.get("name", "").lower() and \
               "indian premier" not in league.get("name", "").lower():
                continue
            for event in league.get("events", []):
                status_type = event.get("status", {}).get("type", {})
                if not status_type.get("completed", False):
                    continue

                competitors = event.get("competitors", [])
                if len(competitors) < 2:
                    continue

                winner_full = ""
                for comp in competitors:
                    if comp.get("winner", False):
                        winner_full = comp.get("displayName", "")

                status_text = status_type.get("shortDetail", "")
                win_runs, win_wickets = parse_win_margin(status_text)

                result = {
                    "date": event.get("date", ""),
                    "team1_full": competitors[0].get("displayName", ""),
                    "team2_full": competitors[1].get("displayName", ""),
                    "winner_full": winner_full,
                    "venue": event.get("venue", {}).get("fullName", ""),
                    "status_text": status_text,
                    "win_by_runs": win_runs,
                    "win_by_wickets": win_wickets,
                    "toss_winner": "",
                    "toss_choice": "",
                }
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# Normalize raw API results
# ---------------------------------------------------------------------------

def normalize_api_result(raw: dict) -> dict:
    """Normalize a raw API result into pipeline-compatible format."""
    team1 = normalize_team_name(raw["team1_full"])
    team2 = normalize_team_name(raw["team2_full"])
    winner = normalize_team_name(raw["winner_full"])

    # Parse date
    date_str = raw.get("date", "")
    match_date = None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            match_date = datetime.strptime(date_str, fmt).date()
            break
        except (ValueError, TypeError):
            continue

    toss_winner = normalize_team_name(raw.get("toss_winner", ""))
    toss_choice = raw.get("toss_choice", "field").lower()
    if toss_choice not in ("bat", "field"):
        toss_choice = "field"

    return {
        "date": match_date,
        "team1": team1,
        "team2": team2,
        "winner": winner,
        "venue": raw.get("venue", ""),
        "win_by_runs": raw.get("win_by_runs", 0),
        "win_by_wickets": raw.get("win_by_wickets", 0),
        "toss_winner": toss_winner,
        "toss_decision": toss_choice,
    }


# ---------------------------------------------------------------------------
# Match API result to schedule
# ---------------------------------------------------------------------------

def match_to_schedule(result: dict, schedule_df: pd.DataFrame) -> int:
    """
    Match a normalized API result to a row in the 2026 schedule.
    Returns the match_number, or -1 if no match found.
    """
    result_date = result.get("date")
    result_teams = {result["team1"], result["team2"]}

    for _, row in schedule_df.iterrows():
        sched_teams = {row["home_team"], row["away_team"]}
        if result_teams != sched_teams:
            continue

        # Match on date if available
        if result_date and pd.notna(row.get("date")):
            sched_date = row["date"]
            if hasattr(sched_date, "date"):
                sched_date = sched_date.date()
            if result_date != sched_date:
                continue

        return int(row["match_number"])

    logger.warning("Could not match result to schedule: %s vs %s on %s",
                   result["team1"], result["team2"], result_date)
    return -1


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_completed_matches(api_key: str = None) -> list:
    """
    Fetch completed IPL 2026 match results.
    Tries CricAPI first (if api_key provided), falls back to ESPN.
    Returns list of normalized result dicts.
    """
    raw_results = []

    if api_key:
        logger.info("Fetching from CricketData.org (CricAPI)...")
        raw_results = fetch_completed_matches_cricapi(api_key)
        if raw_results:
            logger.info("CricAPI returned %d completed matches.", len(raw_results))

    if not raw_results:
        logger.info("Trying ESPN fallback...")
        raw_results = fetch_completed_matches_espn()
        if raw_results:
            logger.info("ESPN returned %d completed matches.", len(raw_results))

    if not raw_results:
        logger.info("No completed matches found from any API source.")
        return []

    # Normalize all results
    normalized = []
    for raw in raw_results:
        result = normalize_api_result(raw)
        # Validate: winner must be one of the two teams
        if result["winner"] not in (result["team1"], result["team2"]):
            logger.warning("Skipping invalid result: winner '%s' not in teams (%s, %s)",
                           result["winner"], result["team1"], result["team2"])
            continue
        # Validate: both teams must be active IPL teams
        if result["team1"] not in ACTIVE_TEAMS_2026 or result["team2"] not in ACTIVE_TEAMS_2026:
            logger.warning("Skipping non-IPL teams: %s vs %s", result["team1"], result["team2"])
            continue
        normalized.append(result)

    logger.info("Normalized %d valid IPL 2026 results.", len(normalized))
    return normalized