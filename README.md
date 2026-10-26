# Predicting NBA Games

## Group 9: Shreeya Adhikari, Claire Hendershot & Owen Sharpe

## Course: DS 4420, Fall 2026

This project's task will be to predict NBA games using a rolling window of games and a myriad of other statistics using time-series forecasting and Bayesian modeling.

### Data Origination & Preprocessing

Our data comes from the free and open-source [Kaggle dataset](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data) by Eoin A. Moore. All credit goes to him for developing and simplifying the data. From Kaggle, the description of the data:

##### 1. `PlayerStatistics.csv`

- The centerpiece of the dataset, this file contains detailed box scores for every individual player in NBA history, game by game.

##### 2. `TeamStatistics.csv`

- A complete record of team-level box scores for every game played in NBA history.

##### 3. `PlayerStatisticsExtended.csv`

- This file contains all the advanced game-by-game statistic for each player in almost game since 1996 when advanced statistics records start on nba.com. It also includes all the data in PlayerStatistics.csv. Each row is an extension of the corresponding PlayerStatistics.csv row, with over 100 columns of data.

##### 4. `TeamStatisticsExtended.csv`

- This is the same idea as PlayerStatisticsExtended.csv, but at a team level.

##### 5. `PlaybyPlay.parquet`

- Detailed play-by-play information for almost every game since 1996, when this data collection began.

##### 6. `Games.csv`

- Comprehensive records of all NBA games, including home and away teams, game outcomes, and, where available, arena and attendance details. Recent entries include expanded data, with plans to retroactively update historical entries.

##### 7. `LeagueSchedule24_25.csv`, `LeagueSchedule25_26.csv`

- The full schedule for the 2024–2025 and 2025-2026 NBA seasons, including game locations and times.

##### 8. `Players.csv`

- Biographical information for almost every player, including height, weight, and position, enabling detailed player analysis across eras.

##### 9. `TeamHistories.csv`

- A historical account of NBA franchises, tracking name changes, relocations, and expansions over the league’s history.

We currently use four of these files: Games.csv, TeamStatisticsExtended.csv, TeamHistories.csv and LeagueSchedule26_27.csv. The player files and the play-by-play file are not used yet.

### Reproducing the data

```bash
python data/data_pulling.py    # downloads version 520 into data/raw_data/v520/ ; you can specify which version of the data within the script
python data/data_cleaning.py   # builds the tables in data/cleaned_data/
```

Raw data is not committed, since `data/raw_data/` is in `.gitignore`. The cleaned tables are committed, so you only need to run these scripts to change the cleaning.

### Cleaning decisions

**Seasons.** We use 2004-05 through 2025-26. 2004-05 is the first season with 30 teams. Seasons are labeled by start year, so `season = 2024` means 2024-25.

**Game types.** We keep regular season, play-in, playoff and NBA Cup final games. Preseason and All-Star games are dropped. Both the game type and the season are read from the game ID. The first digit is the type (2 regular season, 4 playoffs, 5 play-in, 6 Cup final), and the next two digits are the season. We use the ID instead of the dataset's `gameType` labels because the labels are inconsistent (e.g., the Cup is spelled three different ways, and one Cup final is labeled "Regular Season"). We use the ID instead of the date because the 2020 bubble playoffs ran into October.

**Neutral sites.** A game is neutral if it was played away from the home team's usual arenas. An arena counts as a team's home if it hosted at least 5 of that team's home games in some season. This rule handles arena renames, which give the same building a new ID partway through a season. International games, the Las Vegas Cup games and the Spurs' games in Austin all get flagged. Most seasons have between 0 and 8 neutral games.

**Bubble.** Every game from the 2019-20 restart in Orlando, July 30, 2020 onward, is flagged both `bubble` and `neutral`.

**Rest days.** Computed from each team's schedule within a season. They're known before tip-off, so they're safe to use as features.

**Team indices.** Each franchise keeps one team ID through relocations and renames. `teams.parquet` maps those IDs to indices 0 to 29 for the models.

### Splits

| Split   | Seasons            |
| ------- | ------------------ |
| `train` | 2004-05 to 2021-22 |
| `val`   | 2022-23 to 2023-24 |
| `test`  | 2024-25 to 2025-26 |
| `live`  | 2026-27 onward     |

The split is stored as a column in `games` and `team_games`. To change it, edit `SPLITS` in `data_cleaning.py` and rerun.

### Known data issues

**Seven 2024-25 games are missing.** The season has 1,223 of its 1,230 regular season games. Two are 0-0 placeholders in every source file: Jazz vs. Wizards on March 19, 2025, and Spurs vs. Magic on April 1, 2025. The other five game IDs don't appear in the source at all. Eleven teams show 80 or 81 games as a result, and their rest days run long around the gaps. The missing IDs are 22400524, 22400532, 22400537, 22400538, 22400617, 22400627 and 22400988.

**The 2023-24 and 2024-25 Cup finals have no team stats.** The final doesn't count in official statistics. Both games keep their scores, and their stat columns are empty.

**Offensive rebound percentage is recomputed.** The source is missing it for about 15% of rows from 2004-05 through 2016-17, and it appears to count team rebounds after that. We compute `orb_pct` for every season with the standard four-factors definition: offensive rebounds / (offensive rebounds + opponent defensive rebounds). On the 51,709 rows where both exist, ours has a 0.926 correlation with the source and averages 0.249 against the source's 0.289.

**Arena names and cities are blank before 2025-26.** Arena IDs exist for every season, and those are what the neutral flag uses.

**The 2012-13 Celtics vs. Pacers game was canceled** after the Boston Marathon bombing and never made up. This is real and not a data error.

## Tables

All tables are saved as Parquet in `data/cleaned_data/`.

### `games.parquet`

One row per game, 28,375 rows.

| Column                             | Description                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| `game_id`                          | NBA game ID                                                                                    |
| `game_date`                        | Date of the game, Eastern time                                                                 |
| `season`                           | Season start year                                                                              |
| `split`                            | `train`, `val`, `test` or `live`                                                               |
| `game_type`                        | `regular`, `playin`, `playoffs` or `cup_final`                                                 |
| `home_id`, `away_id`               | Team IDs                                                                                       |
| `home_idx`, `away_idx`             | Model indices 0 to 29, from `teams.parquet`                                                    |
| `home_score`, `away_score`         | Final scores                                                                                   |
| `margin`                           | `home_score` minus `away_score`                                                                |
| `home_win`                         | 1 if the home team won                                                                         |
| `home_rest_days`, `away_rest_days` | Days off before this game. 0 means a back-to-back. Empty for a team's first game of the season |
| `neutral`                          | Played away from the home team's usual arenas                                                  |
| `bubble`                           | Played in the 2020 Orlando bubble                                                              |
| `overtime`                         | Went to overtime                                                                               |
| `attendance`                       | Reported attendance                                                                            |
| `arena_id`                         | Arena ID from the source                                                                       |

### `team_games.parquet`

Two rows per game, one from each team's side, 56,750 rows. It shares `game_id`, `game_date`, `season`, `split`, `game_type`, `neutral` and `bubble` with `games`.

**Everything from `win` onward is a result of the game.** Only use it through features built from earlier games. The columns before `win` are known before tip-off.

| Column                                                 | Description                                                       |
| ------------------------------------------------------ | ----------------------------------------------------------------- |
| `team_id`, `opp_id`                                    | This team and its opponent                                        |
| `home`                                                 | 1 if this team was the listed home team                           |
| `game_num`                                             | This team's nth game of the season, counting all kept game types  |
| `rest_days`, `back_to_back`                            | Days off before this game, and whether that's 0                   |
| `win`, `pts`, `opp_pts`, `margin`                      | The result from this team's side                                  |
| `poss`, `pace`                                         | Possessions, and possessions per 48 minutes                       |
| `off_rtg`, `def_rtg`, `net_rtg`                        | Points scored and allowed per 100 possessions, and the difference |
| `efg`, `tov_pct`, `orb_pct`, `ft_rate`                 | This team's four factors, as decimals                             |
| `opp_efg`, `opp_tov_pct`, `opp_orb_pct`, `opp_ft_rate` | The opponent's four factors, as decimals                          |
| `three_pa_rate`                                        | Share of field goal attempts that were threes                     |
| `ts_pct`                                               | True shooting percentage                                          |
| `minutes`                                              | Team minutes: 240 in regulation, plus 25 per overtime             |

### `teams.parquet`

One row per franchise, 30 rows.

| Column                   | Description                         |
| ------------------------ | ----------------------------------- |
| `team_idx`               | Model index, 0 to 29                |
| `team_id`                | NBA team ID                         |
| `city`, `name`, `abbrev` | Current city, name and abbreviation |

### `schedule.parquet`

Upcoming 2026-27 games in the same format as `games.parquet`, without results, 1,207 rows. Cup knockout games are listed before the teams are known, so their team columns are empty. The NBA adds the remaining Cup-dependent games in December, so the schedule needs a fresh pull before live predictions.
