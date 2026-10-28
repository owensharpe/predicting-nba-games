"""
Name: evaluate.py
Author: Shreeya Adhikari
Description: evaluates the trained LSTM against two baselines on the val and test splits, and saves
the metrics table, the test predictions and a calibration plot to models/output/.

Baselines:
  home team    - always pick the home team, with the train home win rate as the probability
  logistic     - logistic regression on each team's average over the same last K games, plus the same context
If the LSTM can't beat the logistic baseline, the sequence modeling isn't adding anything.

Usage: python models/evaluate.py   (run models/lstm.py first)
"""

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from lstm import OUT_DIR, load_model, predict
from sequences import build_dataset

# global variables
CALIBRATION_BINS = 10
COLORS = {"LSTM": "#2a78d6", "Logistic": "#eb6834"}


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p) if np.ptp(p) > 0 else 0.5,  # a constant prediction has no ranking
    }


def pooled(split_data):
    """Flatten a split into the logistic baseline's features: each team's mean over its last K games, plus context."""
    home_seq, away_seq, ctx, _ = split_data
    return np.hstack([home_seq.mean(axis=1), away_seq.mean(axis=1), ctx])


def calibration_plot(y, preds, path):
    """Predicted vs observed home win rate in equal-width bins, one line per model."""
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#9a9a96", linestyle="--", linewidth=1, label="Perfect calibration")

    for name, p in preds.items():
        bins = np.clip(np.digitize(p, edges) - 1, 0, CALIBRATION_BINS - 1)
        df = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"),
                                                                             n=("y", "size"))
        df = df[df.n >= 20]  # bins with only a handful of games are just noise
        ax.plot(df.p, df.y, color=COLORS[name], linewidth=2, marker="o", markersize=6, label=name)

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted home win probability", ylabel="Observed home win rate",
           title="Calibration on the test split")
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    model, hp = load_model()
    data, g = build_dataset(hp["k"])
    print(f"LSTM hyperparameters: {hp}")

    logistic = LogisticRegression(max_iter=2000).fit(pooled(data["train"]), data["train"][3])
    home_rate = data["train"][3].mean()

    rows, test_preds = [], {}
    for split in ["val", "test"]:
        y = data[split][3]
        preds = {
            "Home team": np.full(len(y), home_rate),
            "Logistic": logistic.predict_proba(pooled(data[split]))[:, 1],
            "LSTM": predict(model, data[split]),
        }
        for name, p in preds.items():
            rows.append({"split": split, "model": name, "games": len(y), **metrics(y, p)})
        if split == "test":
            test_preds = preds

    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False, float_format="%.4f"))

    OUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUT_DIR / "metrics.csv", index=False)

    test_games = g[g.split == "test"][["game_id", "game_date", "home_id", "away_id", "home_win"]].reset_index(drop=True)
    test_games["p_lstm"] = test_preds["LSTM"]
    test_games["p_logistic"] = test_preds["Logistic"]
    test_games.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    calibration_plot(data["test"][3], {k: test_preds[k] for k in COLORS}, OUT_DIR / "calibration.png")
    print(f"\nsaved metrics.csv, test_predictions.csv and calibration.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
