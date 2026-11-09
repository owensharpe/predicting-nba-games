"""
Name: mlp.py
Author: Owen Sharpe
Description: trains a vanilla feed-forward neural network that predicts the probability the home team wins.
"""

# import libraries
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from sequences import build_dataset

# global variables
OUT_DIR = Path(__file__).parent / "output"
CHECKPOINT = OUT_DIR / "mlp.pt"
METRICS = OUT_DIR / "mlp_metrics.csv"
PREDICTIONS = OUT_DIR / "mlp_predictions.csv"
SEED = 4420  # same seed as the LSTM

# model settings; these can be changed, but tune them on val and leave test for the final comparison
K = 10              # past games per team
HIDDEN = [64, 32]   # size of each hidden layer
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 5        # stopping after 'PATIENCE' epochs if not better val log loss


# mlp network
class MLP(nn.Module):
    def __init__(self, n_inputs, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()

        # each hidden layer is linear to ReLU to dropout, then one output for the home win logit
        layers = []
        for size in hidden:
            layers += [nn.Linear(n_inputs, size), nn.ReLU(), nn.Dropout(dropout)]
            n_inputs = size
        layers.append(nn.Linear(n_inputs, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # condensing to vector


# various helper functions
def pooled(split_data):

    # same features as the logistic baseline
    home_seq, away_seq, ctx, _ = split_data
    return np.hstack([home_seq.mean(axis=1), away_seq.mean(axis=1), ctx]).astype(np.float32)


def standardize(features):

    # scale every input with train stats only, so val and test never leak
    mean = features["train"].mean(axis=0)
    std = features["train"].std(axis=0)
    std[std == 0] = 1
    scaled = {split: ((x - mean) / std).astype(np.float32) for split, x in features.items()}
    return scaled, mean, std


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def metrics(y, p):

    # the same four metrics evaluate.py reports
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
    }


# training
def train(x, y, hidden=HIDDEN, dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = MLP(x["train"].shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # using Adam optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    x_train, y_train = torch.from_numpy(x["train"]), torch.from_numpy(y["train"])

    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(y_train)).split(BATCH_SIZE):
            opt.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)

        # early stopping on val log loss (same as LSTM)
        val_loss = log_loss(y["val"], predict(model, x["val"]))
        if verbose:
            print(f"epoch {epoch:3d}  train loss {total / len(y_train):.4f}  val log loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {name: w.clone() for name, w in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs == PATIENCE:  # where the cutoff point is for seeing if loss not improving
                if verbose:
                    print(f"stopping early, best val log loss {best_val:.4f}")
                break

    model.load_state_dict(best_state)
    return model, best_val


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # same games and same splits as the LSTM and baselines
    data, g = build_dataset(K)
    features = {split: pooled(data[split]) for split in data}
    x, mean, std = standardize(features)
    y = {split: data[split][3] for split in data}
    print(f"games: train {len(y['train']):,}, val {len(y['val']):,}, test {len(y['test']):,} "
          f"({x['train'].shape[1]} inputs per game)\n")

    model, best_val = train(x, y)

    # save everything needed to rebuild the model later (tensors only, so torch.load works with its defaults)
    torch.save({
        "state_dict": model.state_dict(),
        "hparams": {"k": K, "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY},
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }, CHECKPOINT)

    # score val and test, keeping each game's prediction, so it can be merged with the other models by game_id
    rows, preds = [], []
    for split in ["val", "test"]:
        p = predict(model, x[split])
        rows.append({"split": split, "model": "MLP", "games": len(p), **metrics(y[split], p)})
        games = g[g["split"] == split][["game_id", "game_date", "split", "home_id", "away_id", "home_win"]]
        preds.append(games.assign(p_mlp=p))

    results = pd.DataFrame(rows)
    results.to_csv(METRICS, index=False)
    pd.concat(preds, ignore_index=True).to_csv(PREDICTIONS, index=False)

    # print next to the other models if evaluate.py has been run
    print()
    others = OUT_DIR / "metrics.csv"
    table = pd.concat([pd.read_csv(others), results]) if others.exists() else results
    table["split"] = pd.Categorical(table["split"], ["val", "test"])
    print(table.sort_values("split", kind="stable").to_string(index=False, float_format="%.4f"))

    print(f"\nwrote {CHECKPOINT.name}, {METRICS.name} and {PREDICTIONS.name} to {OUT_DIR}")


if __name__ == "__main__":
    main()
