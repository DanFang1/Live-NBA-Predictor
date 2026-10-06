import json
from tests.conftest import FakeRedis

PAYLOAD = {"player_id": 1, "predicted_pts": 20.0, "pts_low": 12.0, "pts_high": 28.0}


def test_health(client):
    assert client().get("/health").json() == {"status": "ok"}


def test_players_returns_sorted_list(client):
    players = client().get("/players").json()
    assert len(players) > 100
    names = [p["name"] for p in players]
    assert names == sorted(names)


def test_live_cache_hit(client):
    fake = FakeRedis({"live:1": json.dumps(PAYLOAD)})
    r = client(fake).get("/live/1")
    assert r.status_code == 200
    assert r.json() == PAYLOAD


def test_live_cache_miss_returns_200_with_error(client):
    r = client(FakeRedis()).get("/live/1")
    assert r.status_code == 200
    assert "error" in r.json()


def test_live_redis_down_returns_503(client):
    r = client(FakeRedis(fail=True)).get("/live/1")
    assert r.status_code == 503
    assert "error" in r.json()
