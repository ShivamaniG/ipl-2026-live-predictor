# IPL 2026 Winner Prediction

A complete machine learning pipeline to predict IPL 2026 outcomes using real ball-by-ball IPL data.

The project supports two prediction modes:
- Tournament winner probabilities (all 10 teams)
- Match-by-match fixture predictions for the full 2026 schedule

## Latest Prediction Snapshot

### Tournament Winner Ranking (2026)

| Rank | Team | Win Probability |
|------|------|-----------------|
| 1 | Royal Challengers Bengaluru | 11.43% |
| 2 | Kolkata Knight Riders | 10.68% |
| 3 | Rajasthan Royals | 10.66% |
| 4 | Chennai Super Kings | 10.49% |
| 5 | Delhi Capitals | 10.41% |
| 6 | Mumbai Indians | 9.73% |
| 7 | Lucknow Super Giants | 9.59% |
| 8 | Sunrisers Hyderabad | 9.54% |
| 9 | Gujarat Titans | 9.50% |
| 10 | Punjab Kings | 7.97% |

Source: `outputs/results/prediction_2026.json`

### Fixture-level Summary (2026 Schedule)

Predicted league-stage wins from `outputs/results/ipl_2026_match_predictions.csv`:
- MI: 11
- DC: 11
- LSG: 10
- GT: 9
- PBKS: 9
- SRH: 7
- CSK: 5
- RCB: 3
- RR: 3
- KKR: 2

## Model Performance

| Model | CV Accuracy | Test Accuracy | Test AUC |
|------|-------------|---------------|----------|
| Random Forest | 0.6350 | 0.6711 | 0.6995 |
| XGBoost | 0.6311 | 0.6534 | 0.7111 |
| LightGBM | 0.6477 | 0.6600 | 0.7138 |
| Neural Network | 0.6080 | 0.6049 | 0.6141 |
| ExtraTrees | 0.6444 | 0.6512 | 0.7083 |
| Ensemble | - | 0.6490 | 0.7054 |

Source: `outputs/results/model_results.json`

## Data

Primary training source:
- `IPL.csv` (real IPL ball-by-ball data, 2008-2025)

Generated pipeline artifacts:
- `data/raw/matches.csv`
- `data/raw/player_stats.csv`
- `data/raw/teams.json`
- `data/processed/matches_processed.csv`
- `data/processed/features.csv`
- `data/db/ipl.db`

2026 fixture input:
- `ipl-2026-UTC.csv`

## Project Structure

```
IPL-Winner-Prediction-2026/
├── config.py
├── main.py
├── IPL.csv
├── ipl-2026-UTC.csv
├── data/
│   ├── raw/
│   ├── processed/
│   └── db/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── prediction/
├── outputs/
│   ├── models/
│   └── results/
├── notebooks/
├── tests/
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Pipeline

```bash
python main.py --mode setup
python main.py --mode train
python main.py --mode predict
python main.py --mode visualize
```

Or run all in one command:

```bash
python main.py --mode all
```

## Generate Match-by-Match Predictions (2026 schedule)

Input file must exist at `ipl-2026-UTC.csv`.

Output file:
- `outputs/results/ipl_2026_match_predictions.csv`

## Tests

```bash
python -m pytest tests -v
```

## Output Visualizations

Generated in `outputs/results/`:
- `win_probability.png`
- `model_comparison.png`
- `historical_win_rates.png`

## License

MIT
