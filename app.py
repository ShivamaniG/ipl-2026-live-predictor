"""
IPL 2026 Live Prediction Dashboard

Public:  Tournament rankings, points table, match predictions, prediction history
Admin:   Add match results, run ML pipeline (password protected)

Usage:   streamlit run app.py
"""
import os
import sys
import json
import glob
from datetime import date, datetime

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ACTIVE_TEAMS_2026, TEAMS, RESULTS_DIR, LIVE_DIR,
    COMPLETED_2026_CSV, PREDICTIONS_LOG,
)

HISTORY_DIR = os.path.join(LIVE_DIR, "history")

# Read password from Streamlit secrets (cloud) or env var (local)
def _get_admin_password():
    try:
        return st.secrets["IPL_ADMIN_PASSWORD"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("IPL_ADMIN_PASSWORD", "admin")

ADMIN_PASSWORD = _get_admin_password()


def ensure_setup():
    """Auto-run the pipeline if models don't exist (first deploy on Streamlit Cloud)."""
    from config import MODELS_DIR, FEATURES_CSV
    ensemble_path = os.path.join(MODELS_DIR, "ensemble.pkl")
    if not os.path.exists(ensemble_path) or not os.path.exists(FEATURES_CSV):
        st.info("First run detected. Setting up models... (this takes ~3-5 minutes)")
        with st.spinner("Running setup + training pipeline..."):
            from main import mode_setup, mode_train, mode_predict
            mode_setup()
            mode_train()
            mode_predict()
        st.success("Setup complete!")
        st.rerun()

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

def load_predictions():
    path = os.path.join(RESULTS_DIR, "prediction_2026.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_match_predictions():
    path = os.path.join(RESULTS_DIR, "ipl_2026_match_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_completed():
    if os.path.exists(COMPLETED_2026_CSV):
        return pd.read_csv(COMPLETED_2026_CSV)
    return pd.DataFrame()


def load_prediction_history():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "*.json")))
    history = []
    for f in files:
        date_str = os.path.basename(f).replace(".json", "")
        with open(f) as fh:
            data = json.load(fh)
            data["date"] = date_str
            history.append(data)
    return history


def save_prediction_snapshot():
    """Save today's tournament + match predictions as history snapshots."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    today = str(date.today())

    # Tournament rankings snapshot
    pred = load_predictions()
    if pred and "rankings" in pred:
        snapshot = {}
        for r in pred["rankings"]:
            snapshot[r["team_id"]] = r["win_probability"]
        with open(os.path.join(HISTORY_DIR, f"{today}.json"), "w") as f:
            json.dump(snapshot, f, indent=2)

    # Match predictions snapshot
    match_df = load_match_predictions()
    if match_df is not None:
        match_df.to_csv(os.path.join(HISTORY_DIR, f"{today}_matches.csv"), index=False)


def build_points_table(completed_df):
    from collections import defaultdict
    stats = defaultdict(lambda: {"P": 0, "W": 0, "L": 0, "NR": 0, "Pts": 0})
    for _, row in completed_df.iterrows():
        t1, t2, winner = row["team1"], row["team2"], str(row.get("winner", ""))
        stats[t1]["P"] += 1
        stats[t2]["P"] += 1
        if not winner or winner == "NR" or winner not in (t1, t2):
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
        rows.append({"Team": TEAMS.get(team, team), "Abbr": team,
                      "P": s["P"], "W": s["W"], "L": s["L"],
                      "NR": s["NR"], "Pts": s["Pts"]})
    df = pd.DataFrame(rows)
    df = df.sort_values(["Pts", "W"], ascending=[False, False]).reset_index(drop=True)
    df.index = df.index + 1
    return df


# ---------------------------------------------------------------------------
# Playoff logic
# ---------------------------------------------------------------------------

def get_playoff_state(points_table, completed_df):
    top4 = points_table.head(4)["Abbr"].tolist()
    playoff_df = completed_df[completed_df["match_number"] > 70]

    bracket = {
        "q1":    {"label": "Qualifier 1", "teams": (top4[0], top4[1]),
                  "winner": None, "loser": None, "match_number": 71},
        "elim":  {"label": "Eliminator", "teams": (top4[2], top4[3]),
                  "winner": None, "loser": None, "match_number": 72},
        "q2":    {"label": "Qualifier 2", "teams": (None, None),
                  "winner": None, "loser": None, "match_number": 73},
        "final": {"label": "Final", "teams": (None, None),
                  "winner": None, "loser": None, "match_number": 74},
    }

    # Q1
    q1 = playoff_df[playoff_df["match_number"] == 71]
    if len(q1) > 0:
        w = q1.iloc[0]["winner"]
        bracket["q1"]["winner"] = w
        bracket["q1"]["loser"] = top4[1] if w == top4[0] else top4[0]

    # Eliminator
    elim = playoff_df[playoff_df["match_number"] == 72]
    if len(elim) > 0:
        w = elim.iloc[0]["winner"]
        bracket["elim"]["winner"] = w
        bracket["elim"]["loser"] = top4[3] if w == top4[2] else top4[2]

    # Q2
    if bracket["q1"]["loser"] and bracket["elim"]["winner"]:
        bracket["q2"]["teams"] = (bracket["q1"]["loser"], bracket["elim"]["winner"])
        q2 = playoff_df[playoff_df["match_number"] == 73]
        if len(q2) > 0:
            bracket["q2"]["winner"] = q2.iloc[0]["winner"]

    # Final
    if bracket["q1"]["winner"] and bracket["q2"].get("winner"):
        bracket["final"]["teams"] = (bracket["q1"]["winner"], bracket["q2"]["winner"])
        final = playoff_df[playoff_df["match_number"] == 74]
        if len(final) > 0:
            bracket["final"]["winner"] = final.iloc[0]["winner"]

    return bracket


# ---------------------------------------------------------------------------
# Team colors
# ---------------------------------------------------------------------------

TEAM_COLORS = {
    "CSK": "#FFFF00", "MI": "#004BA0", "RCB": "#EC1C24", "KKR": "#3A225D",
    "DC": "#00008B", "PBKS": "#ED1B24", "RR": "#254AA5", "SRH": "#FF822A",
    "LSG": "#A4D4E4", "GT": "#1B2133",
}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("IPL 2026 Live Predictions")

    completed_df = load_completed()
    completed_count = len(completed_df)
    remaining = 70 - completed_count

    st.metric("Matches Completed", f"{completed_count} / 70",
              delta=f"{remaining} remaining")

    pred = load_predictions()
    if pred and "rankings" in pred:
        rankings = pred["rankings"]
        winner = rankings[0]
        st.success(f"Predicted Winner: **{winner['team_name']}** ({winner['win_probability']}%)")

        # Bar chart
        chart_df = pd.DataFrame(rankings)
        chart_df = chart_df.sort_values("win_probability", ascending=True)
        st.bar_chart(chart_df.set_index("team_name")["win_probability"],
                     horizontal=True, height=400)

    # Points table
    if len(completed_df) > 0:
        st.subheader("Points Table")
        pts = build_points_table(completed_df)
        st.dataframe(pts[["Team", "P", "W", "L", "NR", "Pts"]],
                     use_container_width=True, hide_index=False)

    # Playoff bracket
    league_matches = completed_df[completed_df["match_number"] <= 70] if "match_number" in completed_df.columns else completed_df
    if len(league_matches) >= 70:
        st.subheader("Playoff Bracket")
        pts = build_points_table(completed_df)
        bracket = get_playoff_state(pts, completed_df)
        for key in ["q1", "elim", "q2", "final"]:
            b = bracket[key]
            t1, t2 = b["teams"]
            if t1 and t2:
                w = b["winner"]
                if w:
                    st.write(f"**{b['label']}**: {TEAMS.get(t1,t1)} vs {TEAMS.get(t2,t2)} "
                             f"-> **{TEAMS.get(w,w)}** won")
                else:
                    st.write(f"**{b['label']}**: {TEAMS.get(t1,t1)} vs {TEAMS.get(t2,t2)} "
                             f"-- *upcoming*")
            else:
                st.write(f"**{b['label']}**: TBD vs TBD")

        if bracket["final"]["winner"]:
            champ = bracket["final"]["winner"]
            st.balloons()
            st.success(f"IPL 2026 CHAMPION: **{TEAMS.get(champ, champ)}**")


def page_matches():
    st.title("Match-by-Match Predictions")

    match_df = load_match_predictions()
    if match_df is None:
        st.warning("No match predictions found. Run the pipeline first.")
        return

    # Filter
    status_filter = st.selectbox("Filter", ["All", "Actual", "Predicted", "No Result"])
    if status_filter != "All":
        match_df = match_df[match_df["Status"] == status_filter]

    st.dataframe(match_df, use_container_width=True, hide_index=True, height=600)

    # Summary
    if status_filter == "All":
        full_df = load_match_predictions()
        actual = len(full_df[full_df["Status"] == "Actual"])
        predicted = len(full_df[full_df["Status"] == "Predicted"])
        nr = len(full_df[full_df["Status"] == "No Result"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Actual", actual)
        col2.metric("Predicted", predicted)
        col3.metric("No Result", nr)


def page_predictor():
    st.title("Match Predictor")
    st.write("Pick any two teams to see the model's prediction.")

    team_names = {v: k for k, v in TEAMS.items()}
    col1, col2 = st.columns(2)
    with col1:
        t1_name = st.selectbox("Team 1", list(TEAMS.values()), index=0)
    with col2:
        t2_name = st.selectbox("Team 2", list(TEAMS.values()), index=1)

    t1 = team_names[t1_name]
    t2 = team_names[t2_name]

    if t1 == t2:
        st.warning("Pick two different teams.")
        return

    if st.button("Predict"):
        with st.spinner("Running model..."):
            from src.prediction.match_predictor import predict_match
            result = predict_match(t1, t2)

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric(result["team1_name"], f"{result['team1_win_prob']}%")
        col2.metric(result["team2_name"], f"{result['team2_win_prob']}%")

        winner_name = result["predicted_winner_name"]
        confidence = result["confidence"]
        st.success(f"Predicted Winner: **{winner_name}** ({confidence}% confidence)")


def page_history():
    st.title("Prediction History")

    # --- Section 1: Model Accuracy Tracker ---
    st.subheader("Model Accuracy")
    st.caption("How well did the model predict completed matches?")

    match_df = load_match_predictions()
    completed_df = load_completed()

    if match_df is not None and len(completed_df) > 0:
        # Build accuracy table from the EARLIEST snapshot that predicted each match
        # If no historical snapshots, use current predictions
        history_files = sorted(glob.glob(os.path.join(HISTORY_DIR, "*_matches.csv")))

        # Load the earliest prediction for each match
        earliest_predictions = {}
        for f in history_files:
            snap_df = pd.read_csv(f)
            for _, row in snap_df.iterrows():
                mn = int(row["Match"])
                if mn not in earliest_predictions and row["Status"] == "Predicted":
                    earliest_predictions[mn] = {
                        "predicted_winner": row["Predicted Winner"],
                        "probability": row["Win Probability"],
                        "predicted_on": os.path.basename(f).replace("_matches.csv", ""),
                    }

        # Fall back to current predictions for matches without history
        if match_df is not None:
            for _, row in match_df.iterrows():
                mn = int(row["Match"])
                if mn not in earliest_predictions and row["Status"] == "Predicted":
                    earliest_predictions[mn] = {
                        "predicted_winner": row["Predicted Winner"],
                        "probability": row["Win Probability"],
                        "predicted_on": "current",
                    }

        # Compare predictions vs actual results
        accuracy_rows = []
        correct = 0
        total = 0
        for _, row in completed_df.iterrows():
            mn = int(row["match_number"])
            actual_winner = str(row.get("winner", ""))
            if not actual_winner or actual_winner == "NR" or actual_winner not in ACTIVE_TEAMS_2026:
                accuracy_rows.append({
                    "Match": mn,
                    "Home": row["team1"],
                    "Away": row["team2"],
                    "Predicted": "-",
                    "Prob": "-",
                    "Actual": "No Result",
                    "Correct": "-",
                })
                continue

            pred = earliest_predictions.get(mn)
            if pred:
                predicted = pred["predicted_winner"]
                prob = pred["probability"]
                is_correct = predicted == actual_winner
                total += 1
                if is_correct:
                    correct += 1
                accuracy_rows.append({
                    "Match": mn,
                    "Home": row["team1"],
                    "Away": row["team2"],
                    "Predicted": predicted,
                    "Prob": prob,
                    "Actual": actual_winner,
                    "Correct": "Yes" if is_correct else "No",
                })
            else:
                accuracy_rows.append({
                    "Match": mn,
                    "Home": row["team1"],
                    "Away": row["team2"],
                    "Predicted": "N/A",
                    "Prob": "-",
                    "Actual": actual_winner,
                    "Correct": "-",
                })

        if accuracy_rows:
            acc_df = pd.DataFrame(accuracy_rows)
            st.dataframe(acc_df, use_container_width=True, hide_index=True)

            if total > 0:
                pct = correct / total * 100
                col1, col2, col3 = st.columns(3)
                col1.metric("Correct", f"{correct}/{total}")
                col2.metric("Accuracy", f"{pct:.1f}%")
                col3.metric("Matches Tracked", total)
    else:
        st.info("No completed matches yet. Accuracy tracking starts after the first pipeline run.")

    # --- Section 2: Tournament Prediction Trends ---
    st.markdown("---")
    st.subheader("Tournament Prediction Trends")
    st.caption("How each team's win probability changed over time as real matches were added.")

    history = load_prediction_history()
    if len(history) < 2:
        st.info("Need at least 2 days of data to show trends. Run the pipeline daily to build history.")
        if len(history) == 1:
            st.write(f"First snapshot: {history[0]['date']}")
        return

    rows = []
    for h in history:
        row = {"Date": h["date"]}
        for team in ACTIVE_TEAMS_2026:
            row[TEAMS.get(team, team)] = h.get(team, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    st.line_chart(df, height=400)


# ---------------------------------------------------------------------------
# Admin page (separate full-width page, password protected)
# ---------------------------------------------------------------------------

def page_admin():
    st.title("Admin Panel")

    # Password gate
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        password = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Wrong password")
        return

    st.success("Authenticated")

    completed_df = load_completed()
    completed_count = len(completed_df)

    # ---- Layout: 3 columns ----
    col_add, col_manage, col_pipeline = st.columns(3)

    # ---- Column 1: Add Match Result ----
    with col_add:
        st.subheader("Add Match Result")

        league_done = False
        league_matches = completed_df[completed_df["match_number"] <= 70] if "match_number" in completed_df.columns and len(completed_df) > 0 else completed_df
        if len(league_matches) >= 70:
            league_done = True

        if league_done:
            # Playoff mode
            pts = build_points_table(completed_df)
            bracket = get_playoff_state(pts, completed_df)

            next_match = None
            for key in ["q1", "elim", "q2", "final"]:
                b = bracket[key]
                if b["teams"][0] and b["teams"][1] and not b["winner"]:
                    next_match = (key, b)
                    break

            if next_match:
                key, b = next_match
                t1, t2 = b["teams"]
                st.info(f"**{b['label']}** (Match {b['match_number']})")
                st.write(f"{TEAMS.get(t1,t1)} vs {TEAMS.get(t2,t2)}")
                winner = st.selectbox("Winner", [t1, t2],
                                      format_func=lambda x: TEAMS.get(x, x),
                                      key="po_winner")
                toss_winner = st.selectbox("Toss Winner", [t1, t2],
                                           format_func=lambda x: TEAMS.get(x, x),
                                           key="po_toss")
                toss_decision = st.selectbox("Toss Decision", ["bat", "field"],
                                             key="po_td")

                if st.button("Add Playoff Result"):
                    _add_result(b["match_number"], t1, t2, winner, toss_winner, toss_decision, "Playoff")
                    st.success(f"Added {b['label']} result!")
                    st.rerun()
            elif bracket["final"]["winner"]:
                champ = bracket["final"]["winner"]
                st.success(f"Tournament complete! Champion: {TEAMS.get(champ, champ)}")
            else:
                st.info("Waiting for previous playoff results.")
        else:
            # League mode
            played = set(completed_df["match_number"].astype(int).tolist()) if len(completed_df) > 0 else set()
            next_num = 1
            for i in range(1, 71):
                if i not in played:
                    next_num = i
                    break

            match_number = st.number_input("Match Number", min_value=1, max_value=70,
                                            value=next_num)

            team_list = list(TEAMS.keys())
            t1 = st.selectbox("Team 1 (Home)", team_list,
                               format_func=lambda x: TEAMS.get(x, x), key="add_t1")
            t2 = st.selectbox("Team 2 (Away)", team_list, index=1,
                               format_func=lambda x: TEAMS.get(x, x), key="add_t2")

            result_type = st.radio("Result", ["Winner", "No Result (Rain)"])

            if result_type == "Winner":
                winner = st.selectbox("Winner", [t1, t2],
                                      format_func=lambda x: TEAMS.get(x, x), key="add_w")
                toss_winner = st.selectbox("Toss Winner", [t1, t2],
                                           format_func=lambda x: TEAMS.get(x, x), key="add_tw")
                toss_decision = st.selectbox("Toss Decision", ["bat", "field"], key="add_td")
            else:
                winner = "NR"
                toss_winner = ""
                toss_decision = ""

            if st.button("Add Result"):
                if t1 == t2:
                    st.error("Pick two different teams.")
                elif match_number in played:
                    st.error(f"Match {match_number} already has a result.")
                else:
                    _add_result(match_number, t1, t2, winner, toss_winner, toss_decision, "League")
                    st.success(f"Match {match_number} added!")
                    st.rerun()

    # ---- Column 2: Manage Results ----
    with col_manage:
        st.subheader("Manage Results")

        if len(completed_df) > 0:
            st.dataframe(
                completed_df[["match_number", "team1", "team2", "winner", "toss_winner", "toss_decision"]],
                use_container_width=True, hide_index=True, height=300,
            )

            delete_options = []
            for _, row in completed_df.iterrows():
                mn = int(row["match_number"])
                winner_str = row.get("winner", "")
                label = f"Match {mn}: {row['team1']} vs {row['team2']} -> {winner_str}"
                delete_options.append((mn, label))

            selected = st.selectbox("Select match to delete",
                                     delete_options,
                                     format_func=lambda x: x[1],
                                     key="del_select")

            if st.button("Delete Selected Result"):
                match_num_to_delete = selected[0]
                completed_df = completed_df[completed_df["match_number"] != match_num_to_delete]
                completed_df.to_csv(COMPLETED_2026_CSV, index=False)
                st.success(f"Deleted match {match_num_to_delete}")
                st.rerun()
        else:
            st.info("No results added yet.")

    # ---- Column 3: Run Pipeline ----
    with col_pipeline:
        st.subheader("Run Pipeline")
        st.write(f"Matches in system: **{completed_count}**")

        st.caption("This re-runs the full ML pipeline: appends new matches to training data, "
                   "recomputes features, and regenerates all predictions.")

        if st.button("Run ML Pipeline", type="primary"):
            with st.spinner("Running pipeline... (this takes ~1-2 minutes)"):
                _run_pipeline()
            save_prediction_snapshot()
            st.success("Pipeline complete! Predictions updated.")
            st.rerun()

        st.markdown("---")
        st.subheader("Retrain Models")
        st.caption("Retrains all 6 ML models on updated data. Recommended after ~30 matches.")

        if st.button("Retrain + Run Pipeline"):
            with st.spinner("Retraining models... (this takes ~3-5 minutes)"):
                _run_pipeline_retrain()
            save_prediction_snapshot()
            st.success("Retrained and predictions updated!")
            st.rerun()


def _add_result(match_number, t1, t2, winner, toss_winner, toss_decision, stage):
    """Add a match result to completed_2026.csv."""
    completed_df = load_completed()

    new_row = pd.DataFrame([{
        "match_number": match_number,
        "date": str(date.today()),
        "team1": t1,
        "team2": t2,
        "winner": winner,
        "win_by_runs": 0,
        "win_by_wickets": 0,
        "venue": "",
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
    }])

    completed_df = pd.concat([completed_df, new_row], ignore_index=True)
    completed_df = completed_df.sort_values("match_number").reset_index(drop=True)
    os.makedirs(LIVE_DIR, exist_ok=True)
    completed_df.to_csv(COMPLETED_2026_CSV, index=False)


def _run_pipeline():
    """Run the full ML pipeline: append training data, preprocess, features, predict."""
    from src.live.updater import (
        load_completed_results, append_to_training_data, run_pipeline_refresh,
    )
    from src.prediction.predict_2026 import predict_2026_partial, save_predictions
    from src.prediction.match_predictor import predict_all_2026_matches

    completed_df = load_completed_results()
    append_to_training_data(completed_df)
    run_pipeline_refresh(retrain=False)

    rankings = predict_2026_partial(completed_df)
    save_predictions(rankings)
    predict_all_2026_matches()


def _run_pipeline_retrain():
    """Run pipeline with full model retraining."""
    from src.live.updater import (
        load_completed_results, append_to_training_data, run_pipeline_refresh,
    )
    from src.prediction.predict_2026 import predict_2026_partial, save_predictions
    from src.prediction.match_predictor import predict_all_2026_matches

    completed_df = load_completed_results()
    append_to_training_data(completed_df)
    run_pipeline_refresh(retrain=True)

    rankings = predict_2026_partial(completed_df)
    save_predictions(rankings)
    predict_all_2026_matches()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="IPL 2026 Predictions", page_icon="🏏", layout="wide")

    # Auto-setup on first deploy
    ensure_setup()

    # Navigation
    page = st.sidebar.radio("Navigate", [
        "Tournament Overview",
        "Match Predictions",
        "Match Predictor",
        "Prediction History",
        "Admin",
    ])

    if page == "Tournament Overview":
        page_overview()
    elif page == "Match Predictions":
        page_matches()
    elif page == "Match Predictor":
        page_predictor()
    elif page == "Prediction History":
        page_history()
    elif page == "Admin":
        page_admin()


if __name__ == "__main__":
    main()
