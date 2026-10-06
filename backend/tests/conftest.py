import pytest
import redis
from fastapi.testclient import TestClient
import app.main as main


class FakeRedis:
    def __init__(self, data=None, fail=False):
        self.data = data or {}
        self.fail = fail

    def get(self, key):
        if self.fail:
            raise redis.ConnectionError("down")
        return self.data.get(key)

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    def make(fake=None):
        monkeypatch.setattr(main, "redis_client", fake)
        return TestClient(main.app)
    return make
