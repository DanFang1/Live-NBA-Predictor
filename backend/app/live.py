import pandas as pd
import os
from nba_api.live.nba.endpoints import scoreboard, boxscore

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/features.csv")
_features_df = pd.read_csv(FEATURES_PATH, parse_dates=["GAME_DATE"])
_opponent_codes = {
    team: code for code, team in enumerate(sorted(_features_df["opponent"].unique()))
}


def fetch_live_player_stats() -> dict[int, dict]:
    try:
        games = scoreboard.ScoreBoard().games.get_dict()
    except Exception:
        return {}

    stats_by_player = {}
    for game in games:
        if game["gameStatus"] != 2:
            continue
        home = game["homeTeam"]["teamTricode"]
        away = game["awayTeam"]["teamTricode"]
        try:
            box = boxscore.BoxScore(game["gameId"])
            home_players = box.home_team_player_stats.get_dict()
            away_players = box.away_team_player_stats.get_dict()
        except Exception:
            continue
        for players, is_home, opponent in (
            (home_players, 1, away),
            (away_players, 0, home),
        ):
            for p in players:
                s = p.get("statistics") or {}
                if "points" not in s or "minutesCalculated" not in s:
                    continue
                stats_by_player[p["personId"]] = {
                    "pts_so_far": s["points"],
                    "min_so_far": float(
                        s["minutesCalculated"].replace("PT", "").replace("M", "")
                    ),
                    "is_home": is_home,
                    "opponent": opponent,
                }
    return stats_by_player


def get_player_history(player_id: int, opponent: str, today=None) -> dict:
    rows = _features_df[_features_df["PLAYER_ID"] == player_id].sort_values("GAME_DATE")
    if rows.empty or opponent not in _opponent_codes:
        return {}
    today = today if today is not None else pd.Timestamp.today().normalize()
    recent = rows.tail(5)
    vs_opponent = rows.loc[rows["opponent"] == opponent, "PTS"]
    return {
        "last5_avg_pts": recent["PTS"].mean(),
        "last5_avg_ast": recent["AST"].mean(),
        "last5_avg_reb": recent["REB"].mean(),
        "last5_avg_min": recent["MIN"].mean(),
        "avg_pts_vs_opponent": (
            vs_opponent.mean() if len(vs_opponent) else recent["PTS"].mean()
        ),
        "opponent_encoded": _opponent_codes[opponent],
        "days_rest": (today - rows["GAME_DATE"].iloc[-1]).days,
    }


def get_live_features(player_id: int, live_stats: dict) -> dict | None:
    history = get_player_history(player_id, live_stats["opponent"])
    if not history:
        return None
    return {
        **history,
        "pts_so_far": live_stats["pts_so_far"],
        "min_so_far": live_stats["min_so_far"],
        "is_home": live_stats["is_home"],
    }
