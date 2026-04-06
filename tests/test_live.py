"""Tests for live score integration module."""
import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestNormalizeTeamName(unittest.TestCase):
    """Test team name normalization from API to abbreviation."""

    def test_known_full_names(self):
        from src.live.fetch_scores import normalize_team_name
        self.assertEqual(normalize_team_name("Mumbai Indians"), "MI")
        self.assertEqual(normalize_team_name("Chennai Super Kings"), "CSK")
        self.assertEqual(normalize_team_name("Royal Challengers Bengaluru"), "RCB")
        self.assertEqual(normalize_team_name("Kolkata Knight Riders"), "KKR")
        self.assertEqual(normalize_team_name("Delhi Capitals"), "DC")
        self.assertEqual(normalize_team_name("Punjab Kings"), "PBKS")
        self.assertEqual(normalize_team_name("Rajasthan Royals"), "RR")
        self.assertEqual(normalize_team_name("Sunrisers Hyderabad"), "SRH")
        self.assertEqual(normalize_team_name("Lucknow Super Giants"), "LSG")
        self.assertEqual(normalize_team_name("Gujarat Titans"), "GT")

    def test_historical_aliases(self):
        from src.live.fetch_scores import normalize_team_name
        self.assertEqual(normalize_team_name("Kings XI Punjab"), "PBKS")
        self.assertEqual(normalize_team_name("Delhi Daredevils"), "DC")
        self.assertEqual(normalize_team_name("Royal Challengers Bangalore"), "RCB")

    def test_empty_string(self):
        from src.live.fetch_scores import normalize_team_name
        self.assertEqual(normalize_team_name(""), "")


class TestParseWinMargin(unittest.TestCase):
    """Test extraction of win margin from result text."""

    def test_runs(self):
        from src.live.fetch_scores import parse_win_margin
        self.assertEqual(parse_win_margin("CSK won by 33 runs"), (33, 0))

    def test_wickets(self):
        from src.live.fetch_scores import parse_win_margin
        self.assertEqual(parse_win_margin("Mumbai Indians won by 5 wickets"), (0, 5))

    def test_no_match(self):
        from src.live.fetch_scores import parse_win_margin
        self.assertEqual(parse_win_margin("Match drawn"), (0, 0))

    def test_empty(self):
        from src.live.fetch_scores import parse_win_margin
        self.assertEqual(parse_win_margin(""), (0, 0))


class TestExtractWinner(unittest.TestCase):

    def test_standard(self):
        from src.live.fetch_scores import extract_winner
        self.assertEqual(extract_winner("Mumbai Indians won by 5 wickets"), "Mumbai Indians")
        self.assertEqual(extract_winner("CSK won by 33 runs"), "CSK")

    def test_no_winner(self):
        from src.live.fetch_scores import extract_winner
        self.assertEqual(extract_winner("Match drawn"), "")


class TestMatchToSchedule(unittest.TestCase):
    """Test matching API results to schedule rows."""

    def setUp(self):
        self.schedule_df = pd.DataFrame([
            {"match_number": 1, "date": pd.Timestamp("2026-03-28"),
             "home_team": "RCB", "away_team": "SRH", "venue": "M Chinnaswamy Stadium"},
            {"match_number": 2, "date": pd.Timestamp("2026-03-29"),
             "home_team": "MI", "away_team": "KKR", "venue": "Wankhede Stadium"},
            {"match_number": 8, "date": pd.Timestamp("2026-04-04"),
             "home_team": "DC", "away_team": "MI", "venue": "Arun Jaitley Stadium"},
            {"match_number": 9, "date": pd.Timestamp("2026-04-04"),
             "home_team": "GT", "away_team": "RR", "venue": "Narendra Modi Stadium"},
        ])

    def test_exact_match(self):
        from src.live.fetch_scores import match_to_schedule
        result = {"team1": "RCB", "team2": "SRH", "date": date(2026, 3, 28)}
        self.assertEqual(match_to_schedule(result, self.schedule_df), 1)

    def test_reversed_teams(self):
        from src.live.fetch_scores import match_to_schedule
        result = {"team1": "SRH", "team2": "RCB", "date": date(2026, 3, 28)}
        self.assertEqual(match_to_schedule(result, self.schedule_df), 1)

    def test_double_header(self):
        from src.live.fetch_scores import match_to_schedule
        result1 = {"team1": "DC", "team2": "MI", "date": date(2026, 4, 4)}
        result2 = {"team1": "GT", "team2": "RR", "date": date(2026, 4, 4)}
        self.assertEqual(match_to_schedule(result1, self.schedule_df), 8)
        self.assertEqual(match_to_schedule(result2, self.schedule_df), 9)

    def test_no_match(self):
        from src.live.fetch_scores import match_to_schedule
        result = {"team1": "CSK", "team2": "PBKS", "date": date(2026, 3, 28)}
        self.assertEqual(match_to_schedule(result, self.schedule_df), -1)


class TestUpdater(unittest.TestCase):
    """Test updater functions."""

    def test_update_deduplication(self):
        import src.live.updater as updater_mod

        # Use temp dir for live data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_csv = os.path.join(tmpdir, "completed_2026.csv")

            # Patch the module-level constants in updater
            with patch.object(updater_mod, "LIVE_DIR", tmpdir), \
                 patch.object(updater_mod, "COMPLETED_2026_CSV", tmp_csv):

                schedule_df = pd.DataFrame([
                    {"match_number": 1, "date": pd.Timestamp("2026-03-28"),
                     "home_team": "RCB", "away_team": "SRH", "venue": "M Chinnaswamy Stadium"},
                ])

                result = {
                    "team1": "RCB", "team2": "SRH", "winner": "RCB",
                    "date": date(2026, 3, 28), "venue": "M Chinnaswamy Stadium",
                    "win_by_runs": 15, "win_by_wickets": 0,
                    "toss_winner": "RCB", "toss_decision": "bat",
                }

                # First add
                count1 = updater_mod.update_with_new_results([result], schedule_df)
                self.assertEqual(count1, 1)

                # Duplicate add
                count2 = updater_mod.update_with_new_results([result], schedule_df)
                self.assertEqual(count2, 0)

    def test_points_table(self):
        from src.live.updater import build_points_table

        completed_df = pd.DataFrame([
            {"team1": "MI", "team2": "CSK", "winner": "MI",
             "match_number": 1, "date": "", "win_by_runs": 0, "win_by_wickets": 5,
             "venue": "", "toss_winner": "", "toss_decision": ""},
            {"team1": "RCB", "team2": "KKR", "winner": "RCB",
             "match_number": 2, "date": "", "win_by_runs": 20, "win_by_wickets": 0,
             "venue": "", "toss_winner": "", "toss_decision": ""},
            {"team1": "MI", "team2": "RCB", "winner": "RCB",
             "match_number": 3, "date": "", "win_by_runs": 10, "win_by_wickets": 0,
             "venue": "", "toss_winner": "", "toss_decision": ""},
        ])

        table = build_points_table(completed_df)

        # RCB: 2W 0L = 4pts, MI: 1W 1L = 2pts, CSK: 0W 1L = 0pts, KKR: 0W 1L = 0pts
        rcb_row = table[table["Abbr"] == "RCB"].iloc[0]
        self.assertEqual(rcb_row["W"], 2)
        self.assertEqual(rcb_row["Pts"], 4)

        mi_row = table[table["Abbr"] == "MI"].iloc[0]
        self.assertEqual(mi_row["W"], 1)
        self.assertEqual(mi_row["L"], 1)
        self.assertEqual(mi_row["Pts"], 2)


class TestDynamicBayesianWeights(unittest.TestCase):
    """Test that Bayesian weights shift with season progress."""

    def test_early_season(self):
        from src.prediction.predict_2026 import get_dynamic_bayesian_weights
        w = get_dynamic_bayesian_weights(5)
        self.assertEqual(w["model"], 0.30)
        self.assertEqual(w["squad"], 0.35)

    def test_mid_season(self):
        from src.prediction.predict_2026 import get_dynamic_bayesian_weights
        w = get_dynamic_bayesian_weights(30)
        self.assertEqual(w["model"], 0.55)

    def test_late_season(self):
        from src.prediction.predict_2026 import get_dynamic_bayesian_weights
        w = get_dynamic_bayesian_weights(50)
        self.assertEqual(w["model"], 0.75)

    def test_playoffs(self):
        from src.prediction.predict_2026 import get_dynamic_bayesian_weights
        w = get_dynamic_bayesian_weights(60)
        self.assertEqual(w["model"], 0.85)

    def test_weights_sum_to_one(self):
        from src.prediction.predict_2026 import get_dynamic_bayesian_weights
        for n in [0, 10, 20, 30, 45, 50, 56, 60, 70]:
            w = get_dynamic_bayesian_weights(n)
            self.assertAlmostEqual(sum(w.values()), 1.0, places=5)


class TestLoadSchedule(unittest.TestCase):
    """Test schedule loading and normalization."""

    def test_loads_and_normalizes(self):
        from src.live.updater import load_schedule
        df = load_schedule()
        self.assertGreater(len(df), 0)
        # All team names should be abbreviations
        all_teams = set(df["home_team"].tolist() + df["away_team"].tolist())
        from config import ACTIVE_TEAMS_2026
        for team in all_teams:
            self.assertIn(team, ACTIVE_TEAMS_2026,
                          f"Team '{team}' not in active teams")

    def test_has_required_columns(self):
        from src.live.updater import load_schedule
        df = load_schedule()
        for col in ["match_number", "date", "home_team", "away_team", "venue"]:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
