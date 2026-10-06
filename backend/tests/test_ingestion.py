import json
import app.ingestion as ingestion
from tests.conftest import FakeRedis

PREDICTION = {"predicted_pts": 20.0, "pts_low": 12.0, "pts_high": 28.0}
STATS = {"pts_so_far": 10, "min_so_far": 17.0}


def stub_pipeline(monkeypatch, player_ids):
    monkeypatch.setattr(
        ingestion, "fetch_live_player_stats", lambda: {pid: STATS for pid in player_ids}
    )
    monkeypatch.setattr(ingestion, "get_live_features", lambda pid, stats: {"x": 1})
    monkeypatch.setattr(ingestion, "predict_live", lambda f: PREDICTION)


def test_caches_every_live_player_with_ttl(monkeypatch):
    stub_pipeline(monkeypatch, [1, 2])
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)

    assert set(fake.data) == {"live:1", "live:2"}
    assert fake.ttls == {"live:1": 90, "live:2": 90}
    assert json.loads(fake.data["live:1"]) == {"player_id": 1, **PREDICTION}


def test_no_live_players_writes_nothing(monkeypatch):
    stub_pipeline(monkeypatch, [])
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)
    assert fake.data == {}


def test_player_without_history_is_skipped(monkeypatch):
    stub_pipeline(monkeypatch, [1, 2])
    monkeypatch.setattr(
        ingestion, "get_live_features",
        lambda pid, stats: None if pid == 1 else {"x": 1},
    )
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)
    assert set(fake.data) == {"live:2"}


def test_one_failing_player_does_not_block_others(monkeypatch):
    stub_pipeline(monkeypatch, [1, 2])

    def flaky(f):
        if "boom" in f:
            raise ValueError("model error")
        return PREDICTION

    monkeypatch.setattr(
        ingestion, "get_live_features",
        lambda pid, stats: {"boom": 1} if pid == 1 else {"x": 1},
    )
    monkeypatch.setattr(ingestion, "predict_live", flaky)
    fake = FakeRedis()
    ingestion.fetch_and_cache_all_live(fake)
    assert set(fake.data) == {"live:2"}
