"""
Name:  bayes.py
Author: Claire Hendershot
Description: trains a Bayesian model that predicts the probability the home team wins from their point margin
(home court + home team rating - away team rating + rest effects + normal noise). Uses a gaussian prior and
posterior that updates after each game date.

Usage: python models/bayes.py [--sigma 12] [--rho 0.7] [--tau 2] [--drift 0.06]
"""

# import libraries
import argparse
from math import erf
from pathlib import Path

import numpy as np
import pandas as pd

from sequences import CLEAN_DIR, MAX_REST

# global variables
OUT_DIR = Path(__file__).parent / "output"
PREDICTIONS = OUT_DIR / "bayes_predictions.csv"
N_TEAMS = 30

# coefficient layout: [home team, 30 team ratings, back-to-back difference, rest difference]
HOME, B2B, REST = 0, N_TEAMS + 1, N_TEAMS + 2
TEAMS = np.arange(1, N_TEAMS + 1)
N_COEF = N_TEAMS + 3

# prior means and standard deviation
PRIOR_MEAN = np.array([3.0] + [0.0] * N_TEAMS + [0.0, 0.0])
PRIOR_STD = np.array([2.0] + [5.0] * N_TEAMS + [2.0, 2.0])

erf_array = np.vectorize(erf)

class Gaussian:
    def __init__(self, sigma):
        """ Start with prior beliefs. """
        self.mu = PRIOR_MEAN.copy()
        self.V = np.diag(PRIOR_STD ** 2)
        self.sigma2 = sigma ** 2

    def predict(self, X):
        """
        Predict the probability that the home team wins.
        Return expected point margin between two teams and chance that the home team wins (prob that margin > 0).
        """
        mean = X @ self.mu
        sd = np.sqrt(self.sigma2 + np.sum((X @ self.V) * X, axis=1))
        return mean, 0.5 * (1 + erf_array(mean / sd / np.sqrt(2)))

    def update(self, X, y):
        """ Learn from one day of results and update hyperparameters. """
        lam0 = np.linalg.inv(self.V)
        lam_n = lam0 + X.T @ X / self.sigma2
        self.mu = np.linalg.solve(lam_n, lam0 @ self.mu + X.T @ y / self.sigma2)
        self.V = np.linalg.inv(lam_n)

    def add_rating_variance(self, var):
        """ Makes teams' rating a little less certain so ratings can change. """
        self.V[TEAMS, TEAMS] += var

    def new_season(self, rho, tau):
        """ Adjust for teams' changes with every new season. """
        self.mu[TEAMS] *= rho
        self.V[TEAMS, :] *= rho
        self.V[:, TEAMS] *= rho
        self.add_rating_variance(tau ** 2)

def design_matrix(g):
    """
    Turn games data into rows of numbers that the model can understand.
        home court: 1, or 0 if neutral site
        team rating: +1 in home team's column, -1 in away team's column, 0 for all other teams
        back-to-back: home on a back-to-back minus away on a back-to-back (-1, 0, 1)
        rest: home rest days minus away rest days (capped at 7), divided by 7
    """
    X = np.zeros((len(g), N_COEF))
    rows = np.arange(len(g))
    X[:, HOME] = ~g.neutral.to_numpy(bool)
    X[rows, 1 + g.home_idx.to_numpy(int)] = 1
    X[rows, 1 + g.away_idx.to_numpy(int)] = -1

    home_rest = g.home_rest_days.fillna(MAX_REST).clip(upper=MAX_REST)
    away_rest = g.away_rest_days.fillna(MAX_REST).clip(upper=MAX_REST)
    X[:, B2B] = home_rest.eq(0).astype(float) - away_rest.eq(0).astype(float)
    X[:, REST] = (home_rest - away_rest) / MAX_REST

    return X

def run(hp, games):
    """
    Go through the games one day at a time. Predict that day's games first, then update the model from those results.
    Return predictions and model at the end.
    """
    model = Gaussian(hp["sigma"])
    preds = []
    prev_season = None

    for(season, _), day in games.groupby(["season", "game_date"]):
        X, y = design_matrix(day), day.margin.to_numpy(float)
        if prev_season is not None and season != prev_season:
            model.new_season(hp["rho"], hp["tau"])
        prev_season = season
        model.add_rating_variance(hp["drift"])

        mean, p_home = model.predict(X)
        preds.append(day[["game_id", "split", "home_win"]].assign(pred_margin=mean, p_bayes=p_home))
        model.update(X, y)

    return pd.concat(preds, ignore_index=True), model

def val_log_loss(preds):
    """ Log loss on the validation split (lower is better)."""
    val = preds[preds.split == "val"]
    p, y = val.p_bayes, val.home_win
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def train(hp):
    """
    Run the model over every game, print a summary, save the predictions for evaluate.py.
    """
    games = pd.read_parquet(CLEAN_DIR / "games.parquet").sort_values(["season", "game_date", "game_id"])
    print(f"hyperparameters: {hp}")

    preds, model = run(hp, games)
    print(f"val log loss: {val_log_loss(preds)}")
    print(f"home court {model.mu[HOME]}, back to back {model.mu[B2B]}, rest {model.mu[REST]}")

    OUT_DIR.mkdir(exist_ok=True)
    preds.to_csv(PREDICTIONS, index=False)
    print(f"saved {PREDICTIONS.name} to {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=12.0)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--drift", type=float, default=0.06)
    train(vars(parser.parse_args()))
