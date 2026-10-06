import json
import logging
from app.live import fetch_live_player_stats, get_live_features
from app.predictor import predict_live


logger = logging.getLogger(__name__)


def fetch_and_cache_all_live(redis_client):
    live_stats = fetch_live_player_stats()

    for player_id, stats in live_stats.items():
        try:
            features = get_live_features(player_id, stats)
            if features is None:
                continue
            result = predict_live(features)
            payload = json.dumps({"player_id": player_id, **result})
            redis_client.set(f"live:{player_id}", payload, ex=90)
        except Exception as e:
            logger.warning(f"Failed to cache player {player_id}: {e}")
