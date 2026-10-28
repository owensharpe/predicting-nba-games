"""
Name: sequences.py
Author: Shreeya Adhikari
Description: turns the cleaned tables into model inputs. Each game gets the home team's and the away team's
last K games as sequences, plus pre-game context. Shared by the LSTM and the baselines.
"""

# import libraries
from pathlib import Path

import numpy as np
import pandas as pd

# global variables
CLEAN_DIR = Path(__file__).parent.parent / "data" / "cleaned_data"

# stats from each past game that the LSTM sees at one timestep
# these are game results, so they only ever come from games before the one being predicted
SEQ_FEATS = [
    "win", "margin", "home", "rest_days",
    "pace", "off_rtg", "def_rtg",
    "efg", "tov_pct", "orb_pct", "ft_rate",
    "opp_efg", "opp_tov_pct", "opp_orb_pct", "opp_ft_rate",
    "three_pa_rate", "ts_pct",
    "new_season",          # 1 if this is the team's first game of a season (the offseason sits right before it)
    "opp_prior_net_rtg",   # how good the opponent was going into that game, so a win over a bad team counts for less
]

# rest days above this are all treated the same (first game of a season, All-Star break, etc.)
MAX_REST = 7

# pre-game context for the game being predicted; all known before tip-off
CTX_FEATS = ["home_rest", "away_rest", "home_b2b", "away_b2b", "neutral", "playoffs"]


def load_tables():
    """Load team_games (sorted so each team's games are in order) and games."""
    team_games = (pd.read_parquet(CLEAN_DIR / "team_games.parquet")
                    .sort_values(["team_id", "game_date", "game_id"])
                    .reset_index(drop=True))
    games = pd.read_parquet(CLEAN_DIR / "games.parquet")
    return team_games, games


def add_derived(team_games, k):
    """Add rest_days filling, the new_season flag and the opponent's form going into each game."""
    tg = team_games.copy()
    tg["rest_days"] = tg.rest_days.fillna(MAX_REST).clip(upper=MAX_REST)
    tg["new_season"] = (tg.groupby("team_id").season.diff() != 0).astype(float)

    # each team's average net rating over its k games before this one (shift(1) excludes the game itself)
    prior = tg.groupby("team_id").net_rtg.transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean())
    lookup = tg[["game_id", "team_id"]].assign(opp_prior_net_rtg=prior)
    tg = tg.merge(lookup.rename(columns={"team_id": "opp_id"}), on=["game_id", "opp_id"], how="left")
    return tg


def build_sequences(tg, k):
    """
    Return seq, where seq[r] holds the k games the team in row r played before row r, and a boolean
    has_hist marking the rows that have k earlier games. Sequences run across seasons.
    """
    feats = tg[SEQ_FEATS].astype(float)

    # standardize with train rows only, so val/test never leak into the scaling
    train = feats[tg.split == "train"]
    x = ((feats - train.mean()) / train.std()).fillna(0).to_numpy(np.float32)  # the 2 Cup finals have no stats -> mean

    seq = np.zeros((len(tg), k, len(SEQ_FEATS)), np.float32)
    has_hist = np.zeros(len(tg), bool)
    for rows in tg.groupby("team_id").indices.values():
        for j in range(k, len(rows)):
            seq[rows[j]] = x[rows[j - k:j]]
            has_hist[rows[j]] = True
    return seq, has_hist


def context(g):
    """Pre-game context features for each game in g."""
    home_rest = g.home_rest_days.fillna(MAX_REST).clip(upper=MAX_REST)
    away_rest = g.away_rest_days.fillna(MAX_REST).clip(upper=MAX_REST)
    return np.column_stack([
        home_rest / MAX_REST,
        away_rest / MAX_REST,
        home_rest.eq(0),
        away_rest.eq(0),
        g.neutral,
        g.game_type.eq("playoffs"),
    ]).astype(np.float32)


def build_dataset(k=10):
    """
    Build model inputs for every split.

    Returns (data, g). data[split] is a tuple (home_seq, away_seq, ctx, y) of numpy arrays, and g is the
    games table, one row per game, lined up with those arrays (it has a split column).
    Games where either team has fewer than k earlier games are dropped, which only affects the start of 2004-05.
    """
    team_games, games = load_tables()
    tg = add_derived(team_games, k)
    seq, has_hist = build_sequences(tg, k)

    # attach the row number of each side of the game
    tg["row"] = np.arange(len(tg))
    key = tg[["game_id", "team_id", "row"]]
    g = (games.merge(key.rename(columns={"team_id": "home_id", "row": "home_row"}), on=["game_id", "home_id"])
              .merge(key.rename(columns={"team_id": "away_id", "row": "away_row"}), on=["game_id", "away_id"]))
    g = g[has_hist[g.home_row] & has_hist[g.away_row]].reset_index(drop=True)

    data = {}
    for split in ["train", "val", "test"]:
        d = g[g.split == split]
        data[split] = (seq[d.home_row], seq[d.away_row], context(d), d.home_win.to_numpy(np.float32))
    return data, g
