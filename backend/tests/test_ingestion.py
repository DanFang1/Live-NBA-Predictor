import json
from unittest.mock import MagicMock
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2
import app.ingestion as ingestion
from tests.conftest import FakeRedis

PREDICTION = {"predicted_pts": 20.0, "pts_low": 12.0, "pts_high": 28.0}


def stub_nba(monkeypatch, games, boxes):
    """games: {game_id: status_id}, boxes: {game_id: [player_ids]}"""
    board = MagicMock()
    board.game_header.get_data_frame.return_value = pd.DataFrame(
        {"GAME_ID": list(games), "GAME_STATUS_ID": list(games.values())}
    )
    monkeypatch.setattr(ingestion.scoreboardv2, "ScoreboardV2", lambda **kw: board)

    def fake_box(game_id, timeout):
        box = MagicMock()
        box.player_stats.get_data_frame.return_value = pd.DataFrame(
            {"PLAYER_ID": boxes[game_id]}
        )
        return box

    monkeypatch.setattr(boxscoretraditionalv2, "BoxScoreTraditionalV2", fake_box)
    monkeypatch.setattr(ingestion, "get_live_features", lambda pid: {"x": 1})
    monkeypatch.setattr(ingestion, "predict_with_interval", lambda f: PREDICTION)


def test_caches_every_player_in_live_games_with_ttl(monkeypatch):
    stub_nba(monkeypatch, {"G1": 2, "G2": 1}, {"G1": [1, 2], "G2": [3]})
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)

    assert set(fake.data) == {"live:1", "live:2"}
    assert fake.ttls == {"live:1": 90, "live:2": 90}
    assert json.loads(fake.data["live:1"]) == {"player_id": 1, **PREDICTION}


def test_no_live_games_writes_nothing(monkeypatch):
    stub_nba(monkeypatch, {"G1": 1}, {"G1": [1]})
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)
    assert fake.data == {}


def test_scoreboard_failure_is_swallowed(monkeypatch):
    def boom(**kw):
        raise TimeoutError("nba api down")
    monkeypatch.setattr(ingestion.scoreboardv2, "ScoreboardV2", boom)
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)   # must not raise
    assert fake.data == {}


def test_one_bad_player_does_not_block_others(monkeypatch):
    stub_nba(monkeypatch, {"G1": 2}, {"G1": [1, 2]})
    monkeypatch.setattr(
        ingestion, "get_live_features",
        lambda pid: None if pid == 1 else {"x": 1},
    )
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)
    assert set(fake.data) == {"live:2"}
