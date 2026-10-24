"""
Name: data_cleaning.py
Author: Owen Sharpe
Description: cleaning the pulled data from Kaggle to be prepared for model training.
"""

# import libraries
from pathlib import Path

import numpy as np
import pandas as pd

# global variables
RAW_VERSION = 520  # again, can be changed, but try to keep same version as 'data_pulling.py'
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw_data" / f"v{RAW_VERSION}"
CLEAN_DIR = DATA_DIR / "cleaned_data"

# this is the first season where we don't have to deal with expansion changes
START_SEASON = 2004

# encoding game type
GAME_TYPES = {
    1: "preseason",
    2: "regular",
    3: "allstar",
    4: "playoffs",
    5: "playin",
    6: "cup_final",
}
KEEP_TYPES = ["regular", "playin", "playoffs", "cup_final"]

# making our data splits (this is what makes sense in my mind)
# can be group decision though
SPLITS = {
    "train": range(2004, 2022),
    "val": range(2022, 2024),
    "test": range(2024, 2026),
}

# changes made for COVID 'bubble' season in the latter half of 2019-2020
BUBBLE_SEASON = 2019
BUBBLE_START = pd.Timestamp("2020-07-30")

# an arena counts as a team's home if it hosted at least this many of the team's home games in some season; arena IDs change on renames
# real neutral sites (international games, Vegas, Austin) never host more than a couple a season
MIN_HOME_GAMES = 5

# regular season games
EXPECTED_GAMES = {2011: 66, 2020: 72}  # irregularities from lockout & COVID
SKIP_COUNT_SEASONS = {2019}  # game count varies by team due to COVID
KNOWN_SHORT_COUNT = {
    # BOS at IND, April 2013, canceled and never made up
    (2012, 1610612738): 81,
    (2012, 1610612754): 81,
    # 2024-25: seven games missing from the v520 source (two 0-0 placeholders, five absent)
    **{(2024, team): 81 for team in [1610612737, 1610612740, 1610612745, 1610612746,
                                     1610612749, 1610612753, 1610612762, 1610612764]},
    **{(2024, team): 80 for team in [1610612747, 1610612759, 1610612766]},
}

# team stats (renamed)
TEAM_STATS = {
    "possessions": "poss",
    "pace": "pace",
    "offensiveRating": "off_rtg",
    "defensiveRating": "def_rtg",
    "netRating": "net_rtg",
    "effectiveFieldGoalPercentage": "efg",
    "teamTurnoverPercentage": "tov_pct",
    "offensiveReboundPercentage": "orb_pct",
    "freeThrowAttemptRate": "ft_rate",
    "opponentEffectiveFieldGoalPercentage": "opp_efg",
    "opponentTurnoverPercentage": "opp_tov_pct",
    "opponentOffensiveReboundPercentage": "opp_orb_pct",
    "opponentFreeThrowAttemptRate": "opp_ft_rate",
    "percentFieldGoalAttempts3Point": "three_pa_rate",
    "trueShootingPercentage": "ts_pct",
    "numMinutes": "minutes",
}

# various helper functions
def decode_game_id(df, id_col):
    df['game_type'] = (df[id_col] // 10000000).map(GAME_TYPES).fillna("other")
    df["season"] = 2000 + (df[id_col] // 100000) % 100
    return df


def assign_data_split(season):
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    return "live"


def flag_neutral(df, arena_col, exclude=None):
 
    # find every arena that hosted enough of a team's home games in some season to count as home
    counted = df if exclude is None else df[~exclude]
    counts = counted.groupby(["home_id", "season", arena_col]).size()
    home_arenas = counts[counts >= MIN_HOME_GAMES].reset_index()[["home_id", arena_col]].drop_duplicates()
    home_arenas["is_home_arena"] = True
 
    # a game is neutral if its arena is known and isn't one of the home team's arenas
    matched = df[["home_id", arena_col]].merge(home_arenas, on=["home_id", arena_col], how="left")
    return (df[arena_col].notna().to_numpy() & matched["is_home_arena"].isna().to_numpy())
 

# data table builders
def build_games(raw):
    df = raw.rename(columns={
        "gameId": "game_id",
        "hometeamId": "home_id",
        "awayteamId": "away_id",
        "homeScore": "home_score",
        "awayScore": "away_score",
        "arenaId": "arena_id",
        "gameType": "source_game_type",
    })

    df["game_date"] = pd.to_datetime(df["gameDateTimeEst"]).dt.normalize()
    df = df[df["game_date"] >= pd.Timestamp(f"{START_SEASON}-08-01")].copy()
    df = decode_game_id(df, "game_id")

    # assumption about Cup final's ID prefix
    print("\nGame types decoded from gameId vs the dataset's gameType column:")
    print(pd.crosstab(df["game_type"], df["source_game_type"]))

    # only keep game types we care about (i.e., get rid of preseason and all-star)
    df = df[df["game_type"].isin(KEEP_TYPES)].copy()

    # drop unplayed or any postponed games
    unplayed = (df['home_score'].isna() | df['away_score'].isna() | (df['home_score'] + df['away_score'] == 0))
    if unplayed.any():
        print(f"Dropping {unplayed.sum()} unplayed games.")
    df = df[~unplayed].copy()

    # drop duplicate game IDs, and keep the first ones
    duplicates = df['game_id'].duplicated()
    if duplicates.any():
        print(f"Dropping {duplicates.sum()} duplicates from game IDs.")
    df = df[~duplicates].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_win"] = (df["margin"] > 0).astype(int)

    # flag those irregulars mentioned in the global variables
    df['neutral'] = flag_neutral(df, "arena_id")
    df['bubble'] = (df['season'] == BUBBLE_SEASON) & (df['game_date'] >= BUBBLE_START)
    df['neutral'] = flag_neutral(df, "arena_id", exclude=df["bubble"]) | df["bubble"]
    df['split'] = df['season'].map(assign_data_split)

    return df.sort_values(['game_id', 'game_id']).reset_index(drop=True)


def build_team_games(raw_extended, games):
    # start from the games table so every game gets exactly two rows, even with no source stats
    base = ["game_id", "game_date", "season", "split", "game_type", "neutral", "bubble"]
    home = games[base + ["home_id", "away_id", "home_score", "away_score"]].rename(columns={
        "home_id": "team_id", "away_id": "opp_id", "home_score": "pts", "away_score": "opp_pts"})
    away = games[base + ["away_id", "home_id", "away_score", "home_score"]].rename(columns={
        "away_id": "team_id", "home_id": "opp_id", "away_score": "pts", "home_score": "opp_pts"})
    home["home"] = 1
    away["home"] = 0
    df = pd.concat([home, away], ignore_index=True)
    df["win"] = (df["pts"] > df["opp_pts"]).astype(int)
    df["margin"] = df["pts"] - df["opp_pts"]

    # source stats, plus the source's own scores and home flag for a consistency check
    cols = ["gameId", "teamId", "home", "teamScore", "opponentScore",
            "reboundsOffensive", "reboundsDefensive", *TEAM_STATS]
    stats = raw_extended[cols].rename(columns={
        "gameId": "game_id",
        "teamId": "team_id",
        "home": "src_home",
        "teamScore": "src_pts",
        "opponentScore": "src_opp_pts",
        "reboundsOffensive": "oreb",
        "reboundsDefensive": "dreb",
        **TEAM_STATS,
    })

    # look for duplicate team rows
    duplicates = stats.duplicated(["game_id", "team_id"])
    if duplicates.any():
        print(f"Dropping {duplicates.sum()} duplicate team-game rows.")
    stats = stats[~duplicates]
 
    df = df.merge(stats, on=["game_id", "team_id"], how="left")
 
    # where the source has stats, its scores and home flag should match Games.csv
    has_stats = df["src_pts"].notna()
    off = has_stats & ((df["src_pts"] != df["pts"])
                       | (df["src_opp_pts"] != df["opp_pts"])
                       | (df["src_home"] != df["home"]))
    if off.any():
        print(f"WARNING: {df.loc[off, 'game_id'].nunique()} games where Games.csv and team stats disagree.")


    # recompute offensive rebound % from counts for every season, using the standard
    # four-factors definition: OREB / (OREB + opponent DREB), player rebounds only.
    # the source is missing it before 2017 and appears to count team rebounds after,
    # so one definition across all seasons beats mixing two
    # this updated notion will fix final flaws
    opp = df[["game_id", "team_id", "oreb", "dreb"]].rename(
        columns={"team_id": "opp_id", "oreb": "opp_oreb", "dreb": "opp_dreb"})
    df = df.merge(opp, on=["game_id", "opp_id"], how="left")
    orb = df["oreb"] / (df["oreb"] + df["opp_dreb"])
    opp_orb = df["opp_oreb"] / (df["opp_oreb"] + df["dreb"])
 
    # same concept, different definition: expect a high correlation, not an exact match
    both = df["orb_pct"].notna() & orb.notna()
    corr = orb[both].corr(df.loc[both, "orb_pct"])
    print(f"\nORB% vs source on {both.sum():,} rows: correlation {corr:.3f}, "
          f"mean ours {orb[both].mean():.3f}, mean source {df.loc[both, 'orb_pct'].mean():.3f}")
    print("Our ORB% by season:")
    print(orb.groupby(df["season"]).mean().round(3).to_string())
    df["orb_pct"] = orb
    df["opp_orb_pct"] = opp_orb

    # rest days are known before tip-off
    df = df.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    by_team_season = df.groupby(["team_id", "season"])
    prev_date = by_team_season["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - prev_date).dt.days - 1
    df["back_to_back"] = df["rest_days"].eq(0)
    df["game_num"] = by_team_season.cumcount() + 1  # this team's nth game of the season
 
    main_features = ["game_id", "game_date", "season", "split", "game_type", "team_id", "opp_id", "home", "neutral", "bubble", "game_num", "rest_days", "back_to_back", "win", "pts", "opp_pts", "margin"]
    return df[main_features + list(TEAM_STATS.values())]


def build_teams(raw_history, games):

    # mapping each team ID to an index for the model with its name
    ids = sorted(set(games["home_id"]) | set(games["away_id"]))
    latest = raw_history.sort_values("seasonActiveTill").groupby("teamId").tail(1)
    teams = latest[latest["teamId"].isin(ids)].rename(columns={
        "teamId": "team_id", "teamCity": "city", "teamName": "name", "teamAbbrev": "abbrev",
    })
    teams = teams.sort_values("team_id").reset_index(drop=True)
    teams["team_idx"] = np.arange(len(teams))

    # in case teams are missing
    missing = set(ids) - set(teams["team_id"])
    if missing:
        print(f"WARNING: team IDs missing from TeamHistories: {sorted(missing)}")
    return teams[["team_idx", "team_id", "city", "name", "abbrev"]]


def add_game_context(games, team_games, teams):

    # adding additional context around the main game table
    rest = team_games[["game_id", "team_id", "rest_days"]]
    for side in ["home", "away"]:
        games = games.merge(rest.rename(columns={"team_id": f"{side}_id", "rest_days": f"{side}_rest_days"}), on=["game_id", f"{side}_id"], how="left")

    # tracking OT games; 240 team minutes is regulation, each overtime adds 25
    ot_ids = team_games.loc[team_games["minutes"] > 241, "game_id"]
    games["overtime"] = games["game_id"].isin(ot_ids)
 
    to_idx = dict(zip(teams["team_id"], teams["team_idx"]))
    games["home_idx"] = games["home_id"].map(to_idx).astype("Int64")
    games["away_idx"] = games["away_id"].map(to_idx).astype("Int64")
 
    cols = ["game_id", "game_date", "season", "split", "game_type",
            "home_id", "away_id", "home_idx", "away_idx",
            "home_score", "away_score", "margin", "home_win",
            "home_rest_days", "away_rest_days",
            "neutral", "bubble", "overtime", "attendance", "arena_id"]
    return games[cols]


def build_comprehensive_schedule(raw_schedule, teams):

    # building upcoming games in the same format as the games table without results
    df = raw_schedule.rename(columns={"gameId": "game_id", "homeTeamId": "home_id", "awayTeamId": "away_id", "arenaName": "arena"})
    df["game_date"] = pd.to_datetime(df["gameDateTimeEst"]).dt.normalize()
    df = decode_game_id(df, "game_id")
    df = df[df["game_type"].isin(KEEP_TYPES)].copy()

    # the schedule has no arena IDs, so use arena names
    df["neutral"] = flag_neutral(df, "arena")

    # Cup knockout games can be listed before the teams are known
    to_idx = dict(zip(teams["team_id"], teams["team_idx"]))
    df["home_idx"] = df["home_id"].map(to_idx).astype("Int64")
    df["away_idx"] = df["away_id"].map(to_idx).astype("Int64")
 
    cols = ["game_id", "game_date", "season", "game_type", "home_id", "away_id", "home_idx", "away_idx", "neutral", "arena"]
    return df[cols].sort_values(["game_date", "game_id"]).reset_index(drop=True)


# checking tables
def validate(games, team_games):
    issues = []
 
    if games["game_id"].duplicated().any():
        issues.append("duplicate game IDs in games")
 
    # every game should have exactly one home row and away row
    per_game = team_games.groupby("game_id")["home"].agg(["size", "sum"])
    bad = per_game[(per_game["size"] != 2) | (per_game["sum"] != 1)]
    if len(bad):
        issues.append(f"{len(bad)} games without exactly one home and one away row.")
 
    # 30 teams every season
    n_teams = games[games["game_type"] == "regular"].groupby("season")["home_id"].nunique()
    for season, n in n_teams[n_teams != 30].items():
        issues.append(f"season {season}: {n} teams, expected 30.")
 
    # regular season games per team
    last_complete = max(SPLITS["test"])
    reg = team_games[team_games["game_type"] == "regular"]
    for (season, team), n in reg.groupby(["season", "team_id"]).size().items():
        if season in SKIP_COUNT_SEASONS or season > last_complete:
            continue
        expected = KNOWN_SHORT_COUNT.get((season, team), EXPECTED_GAMES.get(season, 82))
        if n != expected:
            issues.append(f"season {season}, team {team}: {n} regular season games, expected {expected}.")
 
    if (team_games["rest_days"] < 0).any():
        issues.append("negative rest days, so some team plays twice on one date.")
 
    # season from the gameId should match the calendar everywhere but the bubble
    year = games["game_date"].dt.year
    calendar_season = np.where(games["game_date"].dt.month >= 8, year, year - 1)
    off_season = (games["season"] != calendar_season) & ~games["bubble"]
    if off_season.any():
        issues.append(f"{off_season.sum()} games where gameId season and date disagree.")
 
    if games["home_idx"].isna().any() or games["away_idx"].isna().any():
        issues.append("some team IDs have no model index.")
 
    print("\nValidation:")
    if issues:
        for issue in issues:
            print(f" - {issue}")
    else:
        print("  all checks passed")
    return issues

# helper summary for the data README
def summarize(games, team_games):
    print("\nGames per season by type:")
    print(pd.crosstab(games["season"], games["game_type"], margins=True))
 
    print("\nFlags per season (neutral, bubble, overtime):")
    print(games.groupby("season")[["neutral", "bubble", "overtime"]].sum())
 
    no_stats = team_games.loc[team_games["poss"].isna(), ["game_id", "game_type"]].drop_duplicates()
    print(f"\nGames with no source stats: {len(no_stats)}")
    if len(no_stats):
        print(no_stats["game_type"].value_counts().to_string())
 
    print("\nMissing share of each team stat:")
    stats = list(TEAM_STATS.values())
    missing = team_games[stats].isna().mean().round(4)
    print(missing[missing > 0] if (missing > 0).any() else "  none missing")


def main():
    CLEAN_DIR.mkdir(exist_ok=True)

    raw_games = pd.read_csv(RAW_DIR / "Games.csv", low_memory=False)
    raw_extended = pd.read_csv(RAW_DIR / "TeamStatisticsExtended.csv", low_memory=False)
    raw_history = pd.read_csv(RAW_DIR / "TeamHistories.csv")
    raw_schedule = pd.read_csv(RAW_DIR / "LeagueSchedule26_27.csv")

    games = build_games(raw_games)
    team_games = build_team_games(raw_extended, games)
    teams = build_teams(raw_history, games)
    games = add_game_context(games, team_games, teams)
    schedule = build_comprehensive_schedule(raw_schedule, teams)

    validate(games, team_games)
    summarize(games, team_games)

    # if everything went fine
    outputs = { "games": games, "team_games": team_games, "teams": teams, "schedule": schedule}
    for name, dataframe in outputs.items():
        dataframe.to_parquet(CLEAN_DIR / f"{name}.parquet", index=False)
        print(f"wrote {name}.parquet: {len(dataframe):,} rows...")

if __name__ == '__main__':
    main()
