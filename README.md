# IPL 2026 Live Predictor

A machine learning system that predicts IPL 2026 outcomes and updates predictions in real-time as matches complete. Built on 17 years of ball-by-ball IPL data (2008–2025).

---

## What Makes This Different

Most IPL prediction models run once and give static results. This system is live — after every match, real results feed back into the model, features update (form, head-to-head, venue stats), and predictions improve as the season progresses.

* Match 1 completes → model updates predictions
* Match 2 completes → predictions adjust again
* Later stages → predictions rely more on real outcomes than assumptions

---

## Features

* Live Score Integration via CricketData/CricAPI workflow
* Partial-Season Prediction with completed matches locked
* Dynamic Bayesian Weights adjusting across season stages
* Match-by-Match Predictions for all fixtures
* Streamlit Dashboard with interactive UI and admin controls
* Playoff Bracket Logic for Q1, Eliminator, Q2, and Final
* Ensemble of 5 ML Models: Random Forest, XGBoost, LightGBM, Neural Network, ExtraTrees

---

## Latest Predictions

### Tournament Winner Ranking

| Rank | Team                        | Win Probability |
| ---- | --------------------------- | --------------- |
| 1    | Royal Challengers Bengaluru | 11.58%          |
| 2    | Rajasthan Royals            | 11.40%          |
| 3    | Delhi Capitals              | 10.75%          |
| 4    | Chennai Super Kings         | 10.39%          |
| 5    | Kolkata Knight Riders       | 10.18%          |
| 6    | Mumbai Indians              | 10.12%          |
| 7    | Lucknow Super Giants        | 9.66%           |
| 8    | Gujarat Titans              | 8.95%           |
| 9    | Sunrisers Hyderabad         | 8.77%           |
| 10   | Punjab Kings                | 8.22%           |

---

## 2026 Validation Snapshot

| Metric                   |                                Value |
| ------------------------ | -----------------------------------: |
| Verified results         |                              41 / 70 |
| Scored matches           |                                   40 |
| No results               |                                    1 |
| Correct picks            |                                   17 |
| Prediction accuracy      |                                42.5% |
| Current predicted winner | Royal Challengers Bengaluru (14.25%) |

Note: Accuracy is computed only on completed matches with a winner. No-result matches are excluded.

---

## Key Findings

* The system performs better at tournament-level prediction than individual match prediction
* Real-time season prediction is significantly harder than offline evaluation
* Live updates and context shifts improve usefulness even with moderate accuracy
* The project is best viewed as a live cricket intelligence system rather than a single accuracy metric

---

## Model Performance

| Model          | CV Accuracy | Test Accuracy | Test AUC |
| -------------- | ----------: | ------------: | -------: |
| Random Forest  |      0.6350 |        0.6711 |   0.6995 |
| XGBoost        |      0.6284 |        0.6600 |   0.7133 |
| LightGBM       |      0.6477 |        0.6600 |   0.7138 |
| Neural Network |      0.6080 |        0.6049 |   0.6141 |
| ExtraTrees     |      0.6444 |        0.6512 |   0.7083 |
| Ensemble       |           — |        0.6446 |   0.7058 |

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### First-Time Setup

```bash
python main.py --mode all
```

### Daily Usage

Option 1: Auto-fetch from API

```bash
python main.py --mode live --api-key YOUR_CRICAPI_KEY
```

Option 2: Dashboard

```bash
streamlit run app.py
```

---

## Dashboard

```bash
streamlit run app.py
```

Opens at: [http://localhost:8501](http://localhost:8501)

---

## How It Works

### Data Pipeline

```
IPL.csv (278K deliveries, 2008–2025)
→ Extract 1,146 matches
→ SQLite database (6 tables)
→ Preprocessing + class balancing
→ 31 feature engineering
→ Train 5 models + ensemble
→ Predict 2026 with Bayesian blending
```

---

### Feature Engineering (31 Features)

| Category      | Count | Description                       |
| ------------- | ----: | --------------------------------- |
| Toss          |     2 | Toss winner and decision          |
| Win Rates     |     6 | Historical and recent performance |
| Form          |     5 | Last 5 matches and current season |
| Head-to-Head  |     1 | Recent matchup strength           |
| Venue         |     8 | Venue performance and conditions  |
| Titles        |     3 | Recent championships              |
| Team Strength |     6 | Batting and bowling metrics       |

---

### Live Update Flow

1. Fetch completed match from API
2. Store in `data/live/completed_2026.csv`
3. Append to training dataset
4. Recompute features with updated data
5. Predict remaining matches
6. Apply dynamic Bayesian weighting
7. Save updated rankings and predictions

---

### Dynamic Bayesian Weights

| Stage    | Matches | Squad | Form | Model | Playoff |
| -------- | ------: | ----: | ---: | ----: | ------: |
| Early    |    0–20 |   35% |  30% |   30% |      5% |
| Mid      |   21–45 |   20% |  20% |   55% |      5% |
| Late     |   46–56 |   10% |  10% |   75% |      5% |
| Playoffs |     57+ |    5% |   5% |   85% |      5% |

---

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

---

## Credits

* Original tournament prediction pipeline: [https://github.com/manpatell/IPL-Winner-Prediction-2026](https://github.com/manpatell/IPL-Winner-Prediction-2026)
* Live system, dashboard, and evaluation: [https://github.com/ShivamaniG](https://github.com/ShivamaniG)

---

## License

MIT