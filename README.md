# Predicting NBA Games

This project's task will be to predict NBA games using a rolling window of games and a myriad of other statistics using time-series forecasting and Bayesian modeling. 

### Data Origination & Preprocessing (In-Progress)

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
