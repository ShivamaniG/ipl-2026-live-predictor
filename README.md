# IPL 2026 Live Predictor

A machine learning system that predicts IPL 2026 outcomes and **updates predictions in real-time** as matches complete. Built on 17 years of ball-by-ball IPL data (2008-2025).

## What Makes This Different

Most IPL prediction models run once and give static results. This system is **live** — after every match, real results feed back into the model, features update (form, H2H, venue stats), and predictions get more accurate as the season progresses.

```
Match 1 result comes in → model re-predicts with updated data
Match 2 result comes in → predictions shift again
...
Match 50 → predictions are mostly based on reality, not guesswork
```

## Features

- **Live Score Integration** — Fetches results from CricketData.org API automatically
- **Partial-Season Prediction** — Completed matches locked in, remaining predicted by ML
- **Dynamic Bayesian Weights** — Model trust increases as season progresses (30% early → 85% in playoffs)
- **Match-by-Match Predictions** — Every match in the 70-match schedule with winner + probability
- **Streamlit Dashboard** — Interactive web UI with admin panel
- **Playoff Bracket** — Auto-detects Q1/Eliminator/Q2/Final matchups from points table
- **5 ML Models + Ensemble** — Random Forest, XGBoost, LightGBM, Neural Network, ExtraTrees with stacking

## Latest Predictions (Live)

### Tournament Winner Ranking

| Rank | Team | Win Probability |
|------|------|-----------------|
| 1 | Royal Challengers Bengaluru | 11.58% |
| 2 | Rajasthan Royals | 11.40% |
| 3 | Delhi Capitals | 10.75% |
| 4 | Chennai Super Kings | 10.39% |
| 5 | Kolkata Knight Riders | 10.18% |
| 6 | Mumbai Indians | 10.12% |
| 7 | Lucknow Super Giants | 9.66% |
| 8 | Gujarat Titans | 8.95% |
| 9 | Sunrisers Hyderabad | 8.77% |
| 10 | Punjab Kings | 8.22% |

*Updated after 12 matches (11 results + 1 no-result)*

### Points Table

| # | Team | P | W | L | NR | Pts |
|---|------|---|---|---|-----|-----|
| 1 | PBKS | 3 | 2 | 0 | 1 | 5 |
| 2 | RCB | 2 | 2 | 0 | 0 | 4 |
| 3 | DC | 2 | 2 | 0 | 0 | 4 |
| 4 | RR | 2 | 2 | 0 | 0 | 4 |
| 5 | MI | 2 | 1 | 1 | 0 | 2 |
| 6 | SRH | 3 | 1 | 2 | 0 | 2 |
| 7 | LSG | 2 | 1 | 1 | 0 | 2 |
| 8 | CSK | 3 | 0 | 3 | 0 | 0 |
| 9 | KKR | 3 | 0 | 2 | 1 | 1 |
| 10 | GT | 2 | 0 | 2 | 0 | 0 |

## Model Performance

| Model | CV Accuracy | Test Accuracy | Test AUC |
|-------|-------------|---------------|----------|
| Random Forest | 0.6350 | 0.6711 | 0.6995 |
| XGBoost | 0.6284 | 0.6600 | 0.7133 |
| LightGBM | 0.6477 | 0.6600 | 0.7138 |
| Neural Network | 0.6080 | 0.6049 | 0.6141 |
| ExtraTrees | 0.6444 | 0.6512 | 0.7083 |
| Ensemble | - | 0.6446 | 0.7058 |

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### First-Time Setup (builds models from scratch)

```bash
python main.py --mode all
```

### Daily Usage (after each match)

```bash
# Option 1: Auto-fetch from API
python main.py --mode live --api-key YOUR_CRICAPI_KEY

# Option 2: Use the dashboard (recommended)
streamlit run app.py
```

### Dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501. Public pages show predictions, points table, and match results. Admin panel (password: `ipl2026admin`) lets you add results and run the pipeline from the UI.

## How It Works

### Data Pipeline

```
IPL.csv (278K ball-by-ball deliveries, 2008-2025)
  → Extract 1,146 matches
  → SQLite database (6 tables)
  → Preprocess + mirror for class balance
  → Engineer 31 features per match
  → Train 5 models + stacking ensemble
  → Predict 2026 with Bayesian blending
```

### 31 Features

| Category | Features | What It Captures |
|----------|----------|-----------------|
| Toss | 2 | Toss winner, bat/field decision |
| Win Rates | 6 | All-time + last 3 seasons (Bayesian smoothed) |
| Form | 5 | Last 5 matches + current season form |
| Head-to-Head | 1 | H2H win rate over last 3 seasons |
| Venue | 8 | Venue win rate, home advantage, pitch conditions |
| Titles | 3 | Recent championship wins |
| Team Strength | 6 | Batting + bowling strength from player stats |

### Live Update Flow

```
1. Fetch completed match from CricAPI
2. Save to data/live/completed_2026.csv
3. Append to training data (data/raw/matches.csv)
4. Re-run preprocessing → features recomputed with 2026 data
5. Partial-season prediction:
   - Completed matches = actual results (locked in)
   - Remaining matches = model predictions (with updated features)
6. Dynamic Bayesian weights shift toward model as season progresses
7. Save updated rankings + match-by-match CSV
```

### Dynamic Bayesian Weights

| Season Stage | Matches | Squad Prior | Form Prior | Model Signal | Playoff Prior |
|---|---|---|---|---|---|
| Early | 0-20 | 35% | 30% | 30% | 5% |
| Mid | 21-45 | 20% | 20% | 55% | 5% |
| Late | 46-56 | 10% | 10% | 75% | 5% |
| Playoffs | 57+ | 5% | 5% | 85% | 5% |

## Project Structure

```
ipl-2026-live-predictor/
├── app.py                          # Streamlit dashboard
├── config.py                       # Central configuration
├── main.py                         # CLI entry point
├── IPL.csv                         # Ball-by-ball data (2008-2025)
├── ipl-2026-UTC.csv                # 2026 fixture schedule
├── data/
│   ├── raw/                        # Extracted matches, player stats
│   ├── processed/                  # Features, preprocessed matches
│   ├── db/                         # SQLite database
│   └── live/                       # Live score data
│       ├── completed_2026.csv      # Real match results
│       ├── api_cache.json          # API response cache
│       └── history/                # Daily prediction snapshots
├── src/
│   ├── data/                       # ETL pipeline
│   ├── features/                   # Feature engineering
│   ├── models/                     # 5 base models + ensemble
│   ├── prediction/                 # Prediction + visualization
│   └── live/                       # Live score integration
│       ├── fetch_scores.py         # CricAPI + ESPN client
│       ├── updater.py              # Pipeline refresh engine
│       └── scheduler.py            # CLI runner
├── outputs/
│   ├── models/                     # Trained model pickles
│   └── results/                    # Predictions + charts
├── notebooks/                      # Analysis notebook
└── tests/                          # 65 tests (22 for live module)
```

## API Setup

1. Sign up at [cricketdata.org](https://cricketdata.org) for a free API key (100 requests/day)
2. Set it as an environment variable: `export CRICAPI_KEY=your_key_here`
3. Or pass it directly: `python main.py --mode live --api-key your_key_here`

## CLI Reference

```bash
python main.py --mode setup        # Extract data + build features
python main.py --mode train        # Train all models
python main.py --mode predict      # Static 2026 prediction
python main.py --mode visualize    # Generate charts
python main.py --mode live         # Fetch scores + update predictions
python main.py --mode live --retrain  # Also retrain models (after ~30 matches)
python main.py --mode all          # Full pipeline end-to-end
```

## Tests

```bash
python -m pytest tests -v
```

65 tests covering data extraction, features, models, predictions, and live integration.

## Credits

- Original prediction pipeline by [manpatell](https://github.com/manpatell/IPL-Winner-Prediction-2026)
- Live score integration, Streamlit dashboard, and match-by-match predictions by [ShivamaniG](https://github.com/ShivamaniG)
- Data source: Real IPL ball-by-ball data (2008-2025)

## License

MIT
