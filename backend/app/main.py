import os
import json
import redis
from contextlib import asynccontextmanager
from functools import lru_cache
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.predictor import predict_pts, predict_with_interval
from app.live import get_live_features, _features_df
from app.ingestion import fetch_and_cache_all_live
from nba_api.stats.static import players as nba_players

redis_client = None
scheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, scheduler
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            fetch_and_cache_all_live,
            "interval",
            seconds=60,
            args=[redis_client],
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)
    if redis_client:
        redis_client.close()

app = FastAPI(lifespan=lifespan)

_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    last5_avg_pts: float
    last5_avg_ast: float
    last5_avg_reb: float
    last5_avg_min: float
    days_rest: float
    is_home: int
    avg_pts_vs_opponent: float
    opponent_encoded: int


@lru_cache(maxsize=1)
def _player_list():
    result = []
    for pid in _features_df["PLAYER_ID"].unique().tolist():
        info = nba_players.find_player_by_id(pid)
        if info:
            result.append({"id": int(pid), "name": info["full_name"]})
    result.sort(key=lambda x: x["name"])
    return result


@app.get("/players")
def list_players():
    return _player_list()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    predicted_pts = predict_pts(request.model_dump())
    return {"predicted_pts": round(predicted_pts, 1)}


@app.get("/live/{player_id}")
def live_predict(player_id: int):
    if redis_client is None:
        features = get_live_features(player_id)
        if features is None:
            return {"error": "Player not found in any live game today"}
        result = predict_with_interval(features)
        return {"player_id": player_id, **result}

    try:
        cached = redis_client.get(f"live:{player_id}")
    except redis.RedisError:
        return JSONResponse(
            status_code=503,
            content={"error": "Prediction cache unavailable"},
        )
    if cached is None:
        return {"error": "Player not found in any live game today"}
    return json.loads(cached)