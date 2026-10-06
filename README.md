# Live NBA Stats Predictor

![tests](https://github.com/DanFang1/Live-NBA-Predictor/actions/workflows/test.yml/badge.svg)

Real-time NBA player stat predictor for in-game sports betting decisions. Select any of the top 150 NBA players and get a live predicted final points total with an 80% confidence interval — updated every 60 seconds during live games.

**Use case:** You placed a LeBron over 27.5 pts parlay. At halftime he has 10 points and the predictor shows he's trending toward 22 — you cash out early instead of riding it out.

---

## Architecture

```
NBA live API (scoreboard + box scores)
       ↑
       │ every 60s
       │
APScheduler (background thread)
       │
       │ predictions for all live players
       ↓
    Redis cache  (key: live:{player_id}, TTL: 90s)
       │
       │ instant reads
       ↓
FastAPI backend  (/live/{player_id})
       │
       │ Server-Sent Events
       ↓
Next.js API route  (/api/live-stream/[id])
       │
       │ EventSource stream
       ↓
  Browser (React)
```

One scoreboard call plus one box score per live game each cycle serves all concurrent users. Each user receives predictions via a persistent SSE connection — no per-user polling.

**Live prediction:** final points = points so far + pre-game forecast × fraction of expected minutes remaining. The interval narrows as the game goes on.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, Tailwind CSS, Recharts |
| Backend | Python, FastAPI, Uvicorn |
| ML | XGBoost, scikit-learn, pandas |
| Cache | Redis |
| Scheduler | APScheduler |
| Data | nba_api |
| Deploy | Railway (3 services: frontend, backend, Redis) |

---

## ML Model

### Training Data
- **50,284 game logs** across top 150 NBA players by minutes played
- **5 seasons** (2020–2026) fetched from the NBA stats API
- Time-based 80/20 train/test split (no data leakage)

### Features
- Rolling 5-game averages: points, assists, rebounds, minutes
- Days of rest
- Home/away
- Opponent team (encoded)
- Historical average points vs. that specific opponent (expanding window, no leakage)

### Models
Three XGBoost models trained separately:
- **Point estimate** — predicts final points
- **Lower bound** — 10th percentile (quantile regression, `alpha=0.1`)
- **Upper bound** — 90th percentile (quantile regression, `alpha=0.9`)

### Accuracy
Measured on 10,027 held-out games (the latest 20%, 2025-03 to 2026-04):
- **MAE: 5.5 points** (5.7 for a last-5-average baseline)
- **78% coverage** of the 80% interval
