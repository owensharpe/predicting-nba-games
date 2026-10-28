"""
Name: lstm.py
Author: Shreeya Adhikari
Description: trains an LSTM that predicts the probability the home team wins. A shared LSTM encodes each team's
last K games, and a small dense head combines both encodings with pre-game context.
Early stopping uses log loss on the val split. The test split is only touched in evaluate.py.

Usage: python models/lstm.py [--k 10] [--hidden 64] [--dropout 0.3] [--lr 1e-3] [--epochs 100] [--patience 5]
"""

# import libraries
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss

from sequences import CTX_FEATS, SEQ_FEATS, build_dataset

# global variables
OUT_DIR = Path(__file__).parent / "output"
CHECKPOINT = OUT_DIR / "lstm.pt"
BATCH_SIZE = 256
SEED = 4420


class GameLSTM(nn.Module):
    def __init__(self, n_feat, n_ctx, hidden=64, dropout=0.3):
        super().__init__()
        # one encoder for both teams, so "recent form" means the same thing on either side
        self.encoder = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_ctx, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, home_seq, away_seq, ctx):
        _, (h_home, _) = self.encoder(home_seq)
        _, (h_away, _) = self.encoder(away_seq)
        return self.head(torch.cat([h_home[-1], h_away[-1], ctx], dim=1)).squeeze(1)  # logit of home win


def to_tensors(split_data):
    return [torch.from_numpy(a) for a in split_data]


def predict(model, split_data):
    """Home win probabilities for one split."""
    model.eval()
    home_seq, away_seq, ctx, _ = to_tensors(split_data)
    with torch.no_grad():
        return torch.sigmoid(model(home_seq, away_seq, ctx)).numpy()


def load_model(path=CHECKPOINT):
    """Load a trained model and the hyperparameters it was trained with."""
    ckpt = torch.load(path)
    hp = ckpt["hparams"]
    model = GameLSTM(len(SEQ_FEATS), len(CTX_FEATS), hp["hidden"], hp["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model, hp


def train(hp):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data, _ = build_dataset(hp["k"])
    train_t = to_tensors(data["train"])
    print(f"games: train {len(data['train'][3])}, val {len(data['val'][3])}, test {len(data['test'][3])}")

    model = GameLSTM(len(SEQ_FEATS), len(CTX_FEATS), hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(train_t[3])).split(BATCH_SIZE):
            home_seq, away_seq, ctx, y = (t[batch] for t in train_t)
            opt.zero_grad()
            loss = loss_fn(model(home_seq, away_seq, ctx), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)

        val_loss = log_loss(data["val"][3], predict(model, data["val"]))
        print(f"epoch {epoch:3d}  train loss {total / len(train_t[3]):.4f}  val log loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {name: w.clone() for name, w in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs == hp["patience"]:
                print(f"stopping early, best val log loss {best_val:.4f}")
                break

    OUT_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "hparams": hp}, CHECKPOINT)
    print(f"saved {CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="number of past games per team")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    train(vars(parser.parse_args()))
