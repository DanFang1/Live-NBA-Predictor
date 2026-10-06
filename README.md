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

---

## Local Setup

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Redis is optional locally — without `REDIS_URL` set, the `/live/{player_id}` endpoint falls back to direct NBA API calls.

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Requires `BACKEND_URL=http://localhost:8000` in `frontend/.env.local`.

### Tests
```bash
cd backend
pip install -r requirements-dev.txt
pytest   # 32 tests, run in CI on every push
```

### Retrain the model
```bash
cd backend
source .venv/bin/activate
python data/fetch_historical.py   # fetches raw game logs
python data/features.py           # builds features.csv
python model/train.py             # trains and saves the 3 models
```

---

## Key Engineering Decisions

**Redis + APScheduler instead of per-request NBA API calls**
The NBA API rate-limits aggressively. With multiple concurrent users each polling every 30s, rate limiting kicks in quickly. APScheduler runs a single background job every 60s that caches predictions for all live players in Redis. User requests read from cache in microseconds — NBA API calls scale with the number of live games, not the number of users.

**Server-Sent Events instead of client-side polling**
The frontend opens one persistent EventSource connection per player selection. The Next.js SSE route polls the backend every 30s and pushes events to the browser. This moves polling off the browser and onto the server, where it can be controlled and observed.

**Quantile regression for confidence intervals**
Rather than a single point prediction, three separate XGBoost models are trained at different quantiles (p10, p50, p90). This gives a calibrated uncertainty range — useful for betting decisions where knowing the downside scenario matters as much as the expected value.

**Next.js API routes as runtime proxy**
Railway Docker builds don't support build-time environment variables for the frontend. All backend calls go through Next.js API routes (`/api/live`, `/api/players`, `/api/live-stream`) that read `BACKEND_URL` at runtime, avoiding the baked-in URL problem.

**Time-based train/test split**
Using a random split on time-series game data would leak future information into the training set (a player's game from March would train on a game from April). The dataset is split chronologically: earliest 80% of games for training, latest 20% for evaluation.
