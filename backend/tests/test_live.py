from unittest.mock import MagicMock
import pandas as pd
import app.live as live
from app.predictor import predict_live

PLAYER_ID = int(live._features_df["PLAYER_ID"].iloc[0])


def game(game_id, status, home="BOS", away="NYK"):
    return {"gameId": game_id, "gameStatus": status,
            "homeTeam": {"teamId": 1, "teamTricode": home},
            "awayTeam": {"teamId": 2, "teamTricode": away}}


def player(person_id, points, minutes):
    return {"personId": person_id,
            "statistics": {"points": points, "minutesCalculated": minutes}}


def stub_nba(monkeypatch, games, boxes):
    """boxes: {game_id: (home_players, away_players)}"""
    sb = MagicMock()
    sb.games.get_dict.return_value = games
    monkeypatch.setattr(live.scoreboard, "ScoreBoard", lambda: sb)

    def fake_box(game_id):
        home, away = boxes[game_id]
        box = MagicMock()
        box.home_team_player_stats.get_dict.return_value = home
        box.away_team_player_stats.get_dict.return_value = away
        return box

    monkeypatch.setattr(live.boxscore, "BoxScore", fake_box)


def stub_history(monkeypatch, rows):
    """rows: list of (date, opponent, pts) for player 7, oldest first."""
    df = pd.DataFrame(
        {
            "PLAYER_ID": 7,
            "GAME_DATE": pd.to_datetime([r[0] for r in rows]),
            "opponent": [r[1] for r in rows],
            "PTS": [r[2] for r in rows],
            "AST": [r[2] / 2 for r in rows],
            "REB": [r[2] / 4 for r in rows],
            "MIN": [r[2] + 10 for r in rows],
        }
    )
    monkeypatch.setattr(live, "_features_df", df)
    monkeypatch.setattr(live, "_opponent_codes", {"AAA": 0, "BBB": 1})


# --- fetch_live_player_stats -------------------------------------------------

def test_fetch_collects_players_from_both_teams_of_live_games(monkeypatch):
    stub_nba(
        monkeypatch,
        [game("G1", 2), game("G2", 1)],
        {"G1": ([player(1, 10, "PT17M")], [player(2, 4, "PT09M")]),
         "G2": ([player(3, 0, "PT00M")], [])},
    )
    assert live.fetch_live_player_stats() == {
        1: {"pts_so_far": 10, "min_so_far": 17.0, "is_home": 1, "opponent": "NYK"},
        2: {"pts_so_far": 4, "min_so_far": 9.0, "is_home": 0, "opponent": "BOS"},
    }


def test_fetch_skips_players_without_stats(monkeypatch):
    stub_nba(
        monkeypatch,
        [game("G1", 2)],
        {"G1": ([{"personId": 1, "statistics": {}}, player(2, 6, "PT12M")], [])},
    )
    assert set(live.fetch_live_player_stats()) == {2}


def test_fetch_survives_scoreboard_failure(monkeypatch):
    def boom():
        raise TimeoutError("nba down")
    monkeypatch.setattr(live.scoreboard, "ScoreBoard", boom)
    assert live.fetch_live_player_stats() == {}


def test_fetch_skips_a_game_whose_box_score_fails(monkeypatch):
    stub_nba(monkeypatch, [game("G1", 2), game("G2", 2)],
             {"G1": ([player(1, 10, "PT17M")], []), "G2": ([], [])})

    real = live.boxscore.BoxScore

    def flaky(game_id):
        if game_id == "G2":
            raise TimeoutError("box score down")
        return real(game_id)

    monkeypatch.setattr(live.boxscore, "BoxScore", flaky)
    assert set(live.fetch_live_player_stats()) == {1}


# --- get_player_history ------------------------------------------------------

def test_last5_uses_the_five_most_recent_games_including_the_latest(monkeypatch):
    rows = [(f"2026-01-0{d}", "AAA", d) for d in range(1, 8)]   # PTS 1..7
    stub_history(monkeypatch, rows)
    h = live.get_player_history(7, "AAA", today=pd.Timestamp("2026-01-10"))
    assert h["last5_avg_pts"] == 5.0          # mean(3..7), not mean(2..6)
    assert h["last5_avg_min"] == 15.0         # mean(13..17)


def test_avg_vs_opponent_uses_full_history_with_fallback(monkeypatch):
    stub_history(monkeypatch, [
        ("2026-01-01", "BBB", 10), ("2026-01-02", "AAA", 20),
        ("2026-01-03", "BBB", 30), ("2026-01-04", "AAA", 40),
    ])
    today = pd.Timestamp("2026-01-10")
    assert live.get_player_history(7, "BBB", today)["avg_pts_vs_opponent"] == 20.0
    stub_history(monkeypatch, [("2026-01-01", "AAA", 10), ("2026-01-02", "AAA", 30)])
    h = live.get_player_history(7, "BBB", today)           # never played BBB
    assert h["avg_pts_vs_opponent"] == h["last5_avg_pts"] == 20.0


def test_days_rest_counts_from_last_game_to_today(monkeypatch):
    stub_history(monkeypatch, [("2026-01-01", "AAA", 10), ("2026-01-05", "AAA", 12)])
    h = live.get_player_history(7, "BBB", today=pd.Timestamp("2026-01-08"))
    assert h["days_rest"] == 3


def test_opponent_is_encoded_like_training(monkeypatch):
    stub_history(monkeypatch, [("2026-01-01", "AAA", 10)])
    assert live.get_player_history(7, "BBB", pd.Timestamp("2026-01-08"))["opponent_encoded"] == 1


def test_unknown_opponent_or_player_has_no_history(monkeypatch):
    stub_history(monkeypatch, [("2026-01-01", "AAA", 10)])
    today = pd.Timestamp("2026-01-08")
    assert live.get_player_history(7, "ZZZ", today) == {}      # e.g. a preseason foreign team
    assert live.get_player_history(999, "AAA", today) == {}


def test_opponent_codes_match_the_training_encoding():
    df = live._features_df
    assert (df["opponent"].map(live._opponent_codes) == df["opponent_encoded"]).all()


# --- get_live_features -------------------------------------------------------

def test_live_features_carry_live_stats_and_context():
    stats = {"pts_so_far": 10, "min_so_far": 17.0, "is_home": 0, "opponent": "BOS"}
    f = live.get_live_features(PLAYER_ID, stats)
    assert (f["pts_so_far"], f["min_so_far"], f["is_home"]) == (10, 17.0, 0)
    assert f["opponent_encoded"] == live._opponent_codes["BOS"]


def test_unknown_player_has_no_features():
    stats = {"pts_so_far": 0, "min_so_far": 0.0, "is_home": 1, "opponent": "BOS"}
    assert live.get_live_features(-1, stats) is None


def test_scoring_more_raises_the_live_prediction(monkeypatch):
    stub_nba(monkeypatch, [game("G1", 2)],
             {"G1": ([player(PLAYER_ID, 10, "PT17M")], [])})
    stats = live.fetch_live_player_stats()[PLAYER_ID]
    features = live.get_live_features(PLAYER_ID, stats)
    low = predict_live({**features, "pts_so_far": 5})
    high = predict_live({**features, "pts_so_far": 25})
    assert high["predicted_pts"] > low["predicted_pts"]
